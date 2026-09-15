"""TECT task network: original RGB/HF conditions and explicit Image/Mask streams.

The baseline file is imported, never modified. DIB and original MSIE are not
constructed. The Mask input/output blocks and PVT time tokens are retained.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn import ConvModule
from model.net import (MMFF, MSFF, ResnetBlock, Upsample1, net as BaselineNet,
                       pvt_v2_b2, timestep_embedding)
from model.tect_diff.reference import TimeResidual, time_embedding
from model.tect_diff.normalization import (
    MAIN_ARCHITECTURE_V1, configure_main_normalization, require_main_normalization,
)


class WindowCrossAttention(nn.Module):
    """Query-owned local cross attention with explicit padding-key exclusion."""

    def __init__(self, channels: int = 256, heads: int = 4, window: int = 8):
        super().__init__()
        if channels % heads:
            raise ValueError("Channels must be divisible by heads")
        self.channels, self.heads, self.window = channels, heads, window
        self.query_norm = nn.GroupNorm(8, channels)
        self.context_norm = nn.GroupNorm(8, channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def _windows(self, x: torch.Tensor):
        batch, channels, height, width = x.shape
        side = self.window
        return (x.view(batch, channels, height // side, side, width // side, side)
                .permute(0, 2, 4, 3, 5, 1).reshape(-1, side * side, channels))

    def forward(self, query_features: torch.Tensor, context_features: torch.Tensor):
        if query_features.shape != context_features.shape:
            raise ValueError("Window query/context must have the same [B,C,H,W] grid")
        batch, channels, height, width = query_features.shape
        pad_h, pad_w = -height % self.window, -width % self.window
        query = F.pad(self.query(self.query_norm(query_features)), (0, pad_w, 0, pad_h))
        context = self.context_norm(context_features)
        key = F.pad(self.key(context), (0, pad_w, 0, pad_h))
        value = F.pad(self.value(context), (0, pad_w, 0, pad_h))
        query, key, value = [self._windows(item).view(-1, self.window ** 2, self.heads,
                                                      channels // self.heads).transpose(1, 2)
                             for item in (query, key, value)]
        valid = F.pad(torch.ones(batch, 1, height, width, device=query.device,
                                 dtype=torch.bool), (0, pad_w, 0, pad_h), value=False)
        key_valid = self._windows(valid).squeeze(-1)[:, None, None, :]
        attended = F.scaled_dot_product_attention(query, key, value,
                                                  attn_mask=key_valid, dropout_p=0.0)
        padded_h, padded_w = height + pad_h, width + pad_w
        attended = attended.transpose(1, 2).reshape(-1, self.window ** 2, channels)
        attended = (attended.view(batch, padded_h // self.window, padded_w // self.window,
                                  self.window, self.window, channels)
                    .permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_h, padded_w))
        return self.output(attended[:, :, :height, :width])


class ImageTaskAdapter(nn.Module):
    """Predict epsilon for each replica; Mask feedback is only stopped shape."""

    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        self.input = nn.Conv2d(256, 256, 1)
        self.shape = nn.Sequential(nn.Conv2d(2, 64, 3, padding=1), nn.GroupNorm(8, 64),
                                   nn.SiLU(), nn.Conv2d(64, 256, 3, padding=1))
        self.mask_to_image = WindowCrossAttention()
        self.block = TimeResidual(256, 256)
        self.output = nn.Sequential(nn.GroupNorm(8, 256), nn.SiLU(),
                                    nn.Conv2d(256, 48, 3, padding=1), nn.PixelShuffle(4))

    def forward(self, features, epsilon_ref, image_logsnr, previous=None, has_previous=None):
        if features.ndim != 5 or features.shape[1:3] != (2, 256):
            raise ValueError("Expected reference features [B,2,256,H/4,W/4]")
        batch, replicas, channels, height, width = features.shape
        if epsilon_ref.shape != (batch, replicas, 3, height * 4, width * 4):
            raise ValueError("Reference epsilon and feature grids do not match")
        if previous is None:
            if has_previous is not None and bool(torch.as_tensor(has_previous).any()):
                raise ValueError("has_previous cannot be true without a predicted shape")
            previous = features.new_zeros(batch, 1, 32, 32)
            available = features.new_zeros(batch, 1, 1, 1)
        else:
            if previous.ndim != 4 or previous.shape[:2] != (batch, 1):
                raise ValueError("Previous prediction must have shape [B,1,H,W]")
            previous = (F.interpolate(previous.detach().float(), size=(32, 32),
                                      mode="bilinear", align_corners=False) >= 0.5).to(features.dtype)
            available = (features.new_ones(batch) if has_previous is None else
                         torch.as_tensor(has_previous, device=features.device, dtype=features.dtype))
            available = available.reshape(-1).expand(batch).reshape(batch, 1, 1, 1).detach()
            if bool(((available != 0) & (available != 1)).any()):
                raise ValueError("has_previous must be binary")
        shape = self.shape(torch.cat((previous * available,
                                      available.expand(-1, 1, 32, 32)), dim=1))
        shape = F.interpolate(shape, size=(height, width), mode="bilinear", align_corners=False)
        times = image_logsnr.reshape(-1)
        if times.numel() == 1:
            times = times.expand(batch * replicas)
        elif times.numel() == batch:
            times = times.repeat_interleave(replicas)
        elif times.numel() != batch * replicas:
            raise ValueError("Image log-SNR must be scalar, [B], or [B,2]")
        time = self.time(time_embedding(times))
        joint = self.input(features.detach().flatten(0, 1))
        shape = shape.repeat_interleave(replicas, dim=0)
        joint = joint + self.mask_to_image(joint, shape) * available.repeat_interleave(replicas, 0)
        joint = self.block(joint, time)
        epsilon_joint = epsilon_ref.detach() + self.output(joint).unflatten(0, (batch, replicas))
        return joint.unflatten(0, (batch, replicas)), epsilon_joint


class MaskInputOutput(nn.Module):
    """The original MIB down2/up2/pred2, with 256-channel inter-stream grid."""

    def __init__(self):
        super().__init__()
        norm = dict(type="BN", requires_grad=True)
        self.time_embed = nn.Sequential(nn.Linear(256, 1024), nn.SiLU(), nn.Linear(1024, 256))
        self.down2 = nn.ModuleList([
            ConvModule(1, 256, 7, padding=3, stride=4, norm_cfg=norm),
            ResnetBlock(256, 256, groups=8, time_emb_dim=256),
            ConvModule(256, 256, 3, padding=1, norm_cfg=norm),
        ])
        self.up2 = nn.Sequential(
            ConvModule(512, 256, 1, norm_cfg=norm), Upsample1(256, 64, factor=2),
            ConvModule(64, 64, 3, padding=1, norm_cfg=norm), Upsample1(64, 32, factor=2),
            ConvModule(32, 32, 3, padding=1, norm_cfg=norm),
        )
        self.pred2 = nn.Sequential(ConvModule(32, 32, 1, norm_cfg=norm), nn.Dropout(0.1),
                                   nn.Conv2d(32, 1, 1))

    def encode(self, noisy_mask, logsnr):
        time = self.time_embed(timestep_embedding(logsnr, 256))
        features = noisy_mask
        for block in self.down2:
            features = block(features, time) if isinstance(block, ResnetBlock) else block(features)
        return features

    def decode(self, features, condition):
        return self.pred2(self.up2(torch.cat((condition, features), dim=1)))


class TECTNetwork(nn.Module):
    architecture_version = MAIN_ARCHITECTURE_V1

    def __init__(self, pretrained_path, gradient_checkpointing: bool = False, *,
                 normalization: str = "batchnorm",
                 architecture_version: str = MAIN_ARCHITECTURE_V1):
        super().__init__()
        require_main_normalization(normalization, architecture_version)
        self.architecture_version = architecture_version
        self.normalization = normalization
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.backbone = pvt_v2_b2(in_chans=3, mask_chans=1)
        self.backbone_t = pvt_v2_b2(in_chans=3, mask_chans=1)
        # Retain baseline's fail-closed feature-key/shape/hash coverage gate.
        BaselineNet._init_weights(self, pretrained_path, False)
        for backbone in (self.backbone, self.backbone_t):
            # Upstream pvt_v2_b2 drops constructor kwargs. Its unused mask
            # projection remains present for provenance but is not optimized.
            backbone.patch_embed1.mask_proj.requires_grad_(False)
        self.mmff1, self.mmff2 = MMFF(64, 256), MMFF(128, 256)
        self.mmff3, self.mmff4 = MMFF(320, 256), MMFF(512, 256)
        self.msff = MSFF()
        self.mask = MaskInputOutput()
        self.image_task_adapter = ImageTaskAdapter()
        self.image_to_mask = WindowCrossAttention()
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        configure_main_normalization(self, normalization, architecture_version)

    def _pvt(self, backbone, logsnr, image):
        # PVT has LayerNorm/DropPath, no BN buffers; checkpointing preserves
        # its RNG. Keep MMFF/MIB BN outside recomputation to avoid double updates.
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(backbone, logsnr, image, use_reentrant=False)
        return backbone(logsnr, image)

    def forward(self, image01, noisy_mask, mask_logsnr, ref_features, epsilon_ref,
                image_logsnr, previous=None, has_previous=None, trace=None):
        if image01.ndim != 4 or image01.shape[1] != 3:
            raise ValueError("TECT conditions require observed RGB [B,3,H,W]")
        if noisy_mask.shape != (image01.shape[0], 1, *image01.shape[-2:]):
            raise ValueError("Mask and observation grids must match")
        if any(size % 32 for size in image01.shape[-2:]):
            raise ValueError("PVT/MSFF input dimensions must be divisible by 32")
        if trace is None or trace.shape != image01.shape:
            raise ValueError("Pass original-image FFT HFVG, resized/flipped with RGB, in [-1,1]")
        mask_logsnr = mask_logsnr.reshape(-1).expand(image01.shape[0])
        rgb = (image01 - self.rgb_mean) / self.rgb_std
        rgb_features = self._pvt(self.backbone, mask_logsnr, rgb)
        trace_features = self._pvt(self.backbone_t, mask_logsnr, trace)
        conditions = [fusion(rgb_feature, trace_feature) for fusion, rgb_feature, trace_feature in
                      zip((self.mmff1, self.mmff2, self.mmff3, self.mmff4), rgb_features, trace_features)]
        condition = self.msff(conditions)
        joint, epsilon_joint = self.image_task_adapter(ref_features, epsilon_ref, image_logsnr,
                                                        previous, has_previous)
        mask_features = self.mask.encode(noisy_mask, mask_logsnr)
        image_context = joint.mean(dim=1)
        mask_features = mask_features + self.image_to_mask(mask_features, image_context)
        logits = self.mask.decode(mask_features, condition)
        return {"logits_base": logits, "epsilon_joint": epsilon_joint, "F_joint": joint}
