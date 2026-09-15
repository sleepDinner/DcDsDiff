"""Pixel-space reference denoiser with an explicit, enforceable frozen boundary.

The only data inputs are the noisy RGB observation and its own log-SNR.
Stage A trains epsilon prediction; after ``freeze()`` even an enclosing
``model.train()`` cannot enable gradients or mutable training state here.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


REFERENCE_ARCHITECTURE_V1 = "tect-reference-pixel-unet-v1"
REFERENCE_ARCHITECTURE_V2 = "tect-reference-pixel-unet-v2-stable-paths"
RESIDUAL_MERGE_SCALE = 1 / math.sqrt(2)


def time_embedding(times: torch.Tensor, width: int = 256) -> torch.Tensor:
    times = times.float().reshape(-1)
    frequency = torch.exp(
        -math.log(10000) * torch.arange(width // 2, device=times.device).float()
        / (width // 2)
    )
    phase = times[:, None] * frequency[None]
    return torch.cat((phase.cos(), phase.sin()), dim=1)


class TimeResidual(nn.Module):
    """Two convolution residual block; GroupNorm has no running statistics."""

    def __init__(self, in_channels: int, out_channels: int, time_width: int = 256,
                 *, stabilized: bool = False):
        super().__init__()
        # Default stays v1 because the task Image adapter also uses this block.
        self.stabilized = bool(stabilized)
        self.projected_skip = in_channels != out_channels
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_width, 2 * out_channels))
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        hidden = self.conv1(F.silu(normalized))
        scale, shift = self.time(time).chunk(2, dim=1)
        hidden = self.norm2(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        # A channel-changing skip must not bypass all normalization. Identity
        # skips preserve the signal; scaled merges reduce repeated residual
        # amplification across the reference's deep path.
        skip_input = normalized if self.stabilized and self.projected_skip else x
        merged = self.skip(skip_input) + self.conv2(F.silu(hidden))
        return merged * RESIDUAL_MERGE_SCALE if self.stabilized else merged


class ReferenceDenoiser(nn.Module):
    """Unshuffle(4), four U-Net levels, and RGB epsilon prediction.

    ``forward(noisy_image, image_time)`` returns epsilon [B,3,H,W] and
    features [B,256,H/4,W/4]. ``image_time`` is the Image log-SNR, not
    the independently sampled Mask time. There is no clean-image, mask,
    target-noise, or conditioning-feature argument.
    """

    architecture_version = REFERENCE_ARCHITECTURE_V1

    def __init__(self, gradient_checkpointing: bool = False,
                 architecture_version: str = REFERENCE_ARCHITECTURE_V1):
        super().__init__()
        if architecture_version not in (REFERENCE_ARCHITECTURE_V1, REFERENCE_ARCHITECTURE_V2):
            raise ValueError(f"Unknown reference architecture: {architecture_version!r}")
        self.architecture_version = architecture_version
        self.stabilized = architecture_version == REFERENCE_ARCHITECTURE_V2
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.register_buffer("frozen", torch.tensor(False), persistent=True)
        self.unshuffle = nn.PixelUnshuffle(4)
        self.entry = nn.Conv2d(48, 64, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        widths = (64, 128, 256, 256)
        self.encoder = nn.ModuleList([
            nn.ModuleList([TimeResidual(width, width, stabilized=self.stabilized),
                           TimeResidual(width, width, stabilized=self.stabilized)])
            for width in widths
        ])
        self.downsample = nn.ModuleList([
            nn.Conv2d(widths[index], widths[index + 1], 3, stride=2, padding=1)
            for index in range(3)
        ])
        self.upsample = nn.ModuleList([
            nn.Conv2d(widths[index + 1], widths[index], 3, padding=1)
            for index in range(2, -1, -1)
        ])
        self.decoder = nn.ModuleList([
            nn.ModuleList([TimeResidual(2 * widths[index], widths[index], stabilized=self.stabilized),
                           TimeResidual(widths[index], widths[index], stabilized=self.stabilized)])
            for index in range(2, -1, -1)
        ])
        self.feature = nn.Conv2d(64, 256, 1)
        self.output_norm = nn.GroupNorm(8, 256)
        self.output = nn.Conv2d(256, 48, 3, padding=1)
        self.shuffle = nn.PixelShuffle(4)
        # v1 registers no extra parameters: historical state dictionaries still
        # load strictly. v2 breaks the unnormalized down/up/projection chain
        # identified in the collapsed reference, preserving its output shapes.
        self.downsample_norm = nn.ModuleList([
            nn.GroupNorm(8, widths[index]) for index in range(3)
        ] if self.stabilized else [])
        self.upsample_norm = nn.ModuleList([
            nn.GroupNorm(8, widths[index + 1]) for index in range(2, -1, -1)
        ] if self.stabilized else [])

    def freeze(self) -> "ReferenceDenoiser":
        self.frozen.fill_(True)
        self.requires_grad_(False)
        super().train(False)
        return self

    def train(self, mode: bool = True) -> "ReferenceDenoiser":
        if bool(self.frozen):
            self.requires_grad_(False)
            return super().train(False)
        return super().train(mode)

    def _block(self, module: nn.Module, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, time, use_reentrant=False)
        return module(x, time)

    def forward(self, noisy_image: torch.Tensor, image_time: torch.Tensor):
        if noisy_image.ndim != 4 or noisy_image.shape[1] != 3:
            raise ValueError("Reference requires noisy RGB [B,3,H,W]")
        if any(dimension % 4 for dimension in noisy_image.shape[-2:]):
            raise ValueError("Reference dimensions must be divisible by 4")
        if image_time.numel() not in (1, noisy_image.shape[0]):
            raise ValueError("One Image log-SNR per observation is required")
        # no_grad, not inference_mode: downstream trainable modules may save
        # these ordinary detached tensors for their parameter gradients.
        context = torch.no_grad() if bool(self.frozen) else nullcontext()
        with context:
            time = self.time(time_embedding(image_time).expand(noisy_image.shape[0], -1))
            hidden = self.entry(self.unshuffle(noisy_image))
            skips = []
            for level, blocks in enumerate(self.encoder):
                for block in blocks:
                    hidden = self._block(block, hidden, time)
                skips.append(hidden)
                if level < len(self.downsample):
                    if self.stabilized:
                        hidden = self.downsample_norm[level](hidden)
                    hidden = self.downsample[level](hidden)
            decoder_levels = zip(self.upsample, self.decoder, reversed(skips[:-1]))
            for level, (up, blocks, skip) in enumerate(decoder_levels):
                if self.stabilized:
                    hidden = self.upsample_norm[level](hidden)
                hidden = up(F.interpolate(hidden, size=skip.shape[-2:], mode="nearest"))
                hidden = torch.cat((hidden, skip), dim=1)
                for block in blocks:
                    hidden = self._block(block, hidden, time)
            # This feature projection is on the epsilon prediction path and
            # therefore receives real Stage A supervision.
            features = F.silu(self.output_norm(self.feature(hidden)))
            epsilon = self.shuffle(self.output(features))
            return epsilon, features
