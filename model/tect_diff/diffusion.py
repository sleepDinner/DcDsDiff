"""Observation-anchored image denoising and consistent x0 mask DDIM.

Training samples one uniform mask time per microbatch. The inverse of the
nonlinear sampling grid maps that time onto exactly the inference prefixes.
Image measurement never reads the mask, history, segmentation GT or task head.
"""
import hashlib
import math
import time

import torch
from torch import nn
import torch.nn.functional as F

from denoising_diffusion_pytorch.simple_diffusion import (
    logsnr_schedule_cosine, logsnr_schedule_shifted,
)
from model.loss import structure_loss
from .evidence import FixedTrajectoryEvidence
from .amp_context import self_condition_no_grad


def coefficients(logsnr):
    return logsnr.float().sigmoid().sqrt(), (-logsnr.float()).sigmoid().sqrt()


def ddim_step(noisy_mask, probability, logsnr_t, logsnr_s, final=False):
    y0 = 2 * probability.float() - 1
    if final:
        return y0
    a_t, sigma_t = coefficients(logsnr_t)
    a_s, sigma_s = coefficients(logsnr_s)
    a_t, sigma_t, a_s, sigma_s = [x.reshape(-1, 1, 1, 1) for x in (a_t, sigma_t, a_s, sigma_s)]
    eps = (noisy_mask.float() - a_t * y0) / sigma_t.clamp_min(1e-12)
    return a_s * y0 + sigma_s * eps


def stable_noise(image, ids, seed, purpose):
    tensors = []
    for sample_id in ids:
        digest = hashlib.sha256(f'{seed}|{sample_id}|{purpose}'.encode()).digest()
        generator = torch.Generator(device=image.device)
        generator.manual_seed(int.from_bytes(digest[:8], 'little') % (2**63 - 1))
        channels = 1 if purpose == 'mask' else 3
        tensors.append(torch.randn((channels, *image.shape[-2:]), device=image.device, generator=generator))
    return torch.stack(tensors)


class TECTDiffusion(nn.Module):
    def __init__(self, network, reference, calibration, config):
        super().__init__()
        self.network = network
        self.reference = reference.freeze()
        self.config = config
        self.evidence = FixedTrajectoryEvidence(calibration, device=next(network.parameters()).device)
        self.gamma_logit = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))
        self.schedule = logsnr_schedule_shifted(logsnr_schedule_cosine, config['resolution'], 64)
        self.lambdas = tuple(config['image_diffusion']['lambdas'])
        self.last_diagnostics = {}

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    @torch.no_grad()
    def reference_path(self, image, m, noises=None, keep_all_features=False):
        """R independent noises; each is reused at all visible scales only."""
        observed = image.float() * 2 - 1
        if noises is None:
            noises = torch.stack([torch.randn_like(observed), torch.randn_like(observed)])
        responses, features, estimates = [], [], []
        for k in range(m):
            times = torch.full((len(image),), float(self.lambdas[k]), device=image.device)
            a, sigma = coefficients(times)
            one_scale, one_features, one_estimates = [], [], []
            for r in range(2):
                noisy = a[:, None, None, None] * observed + sigma[:, None, None, None] * noises[r]
                epsilon, feature = self.reference(noisy, times)
                one_scale.append(self.evidence.measure(epsilon.float(), noises[r]))
                if keep_all_features or k == m - 1:
                    one_features.append(feature.detach())
                    one_estimates.append(epsilon.detach())
            responses.append(torch.stack(one_scale))
            if one_features:
                features.append(torch.stack(one_features, dim=1))
                estimates.append(torch.stack(one_estimates, dim=1))
        # [R,m,B,9,H,W], features[k]=[B,R,256,H/4,W/4].
        return torch.stack(responses, dim=1), features, estimates, noises

    def control(self, logits, evidence, j, m):
        gamma = 2 * self.gamma_logit.float().sigmoid() * (float(j) / 9)
        if m < 2:
            gamma = gamma * 0
        ell, q = evidence['ell'].float(), evidence['q'].float()
        if ell.ndim == 3:
            ell, q = ell[:, None], q[:, None]
        corrected = logits.float() + gamma * q * ell
        return corrected, gamma

    def task(self, image, trace, noisy_mask, logsnr, features, epsilon, m, previous=None):
        time_i = torch.full((len(image),), float(self.lambdas[m-1]), device=image.device)
        return self.network(image, noisy_mask, logsnr, features, epsilon, time_i,
                            previous=previous, trace=trace)

    def forward(self, image, trace, gt):
        batch = len(image)
        t = torch.rand((), device=image.device).clamp(1e-5, 1-1e-5)
        progress = 1 - 2 / math.pi * torch.asin(t)
        j = min(9, int((progress * 10).item()))
        m = min(4, 1 + 4 * j // 10)
        logsnr = self.schedule(t.expand(batch))
        a, sigma = coefficients(logsnr)
        epsilon_mask = torch.randn_like(gt)
        noisy_mask = a[:, None, None, None] * (2 * gt.float() - 1) + sigma[:, None, None, None] * epsilon_mask
        with torch.no_grad():
            responses, features, estimates, noise_image = self.reference_path(image, m)
            context = self.evidence.prepare(image * 2 - 1)
            measured = self.evidence.prefix(context, responses)
        previous = None
        if torch.rand((), device=image.device).item() < self.config['model']['self_condition_probability']:
            # No stochastic buffers, dropout or BatchNorm updates in this extra pass.
            was_training = self.network.training
            self.network.eval()
            with self_condition_no_grad():
                first = self.task(image, trace, noisy_mask, logsnr, features[-1], estimates[-1], m)
                first_ctrl, _ = self.control(first['logits_base'], measured, j, m)
                previous = (F.interpolate(first_ctrl.sigmoid(), (32, 32), mode='bilinear', align_corners=False) >= .5).float().detach()
            self.network.train(was_training)
        output = self.task(image, trace, noisy_mask, logsnr, features[-1], estimates[-1], m, previous)
        controlled, gamma = self.control(output['logits_base'], measured, j, m)
        with torch.autocast(device_type='cuda', enabled=False):
            mask_loss = structure_loss(controlled.float(), gt.float())
            image_loss = F.mse_loss(output['epsilon_joint'].float(), noise_image.permute(1, 0, 2, 3, 4).float())
            loss = mask_loss + self.config['training']['lambda_image'] * image_loss
        self.last_diagnostics = {
            'prefix': m, 'sampling_step': j, 'mask_loss': mask_loss.detach(),
            'image_loss': image_loss.detach(), 'gamma': gamma.detach(),
            'joint_ref_mse': (output['epsilon_joint'].float()-estimates[-1].float()).square().mean().detach(),
            'control_logit_change': (controlled-output['logits_base'].float()).abs().mean().detach(),
            'A_mean': measured['A'].mean(), 'A_negative_fraction': (measured['A'] < 0).float().mean(),
            'q_mean': measured['q'].mean(), 'q_zero_fraction': (measured['q'] == 0).float().mean(),
            'ell_mean': measured['ell'].mean(),
        }
        return loss

    @torch.no_grad()
    def sample(self, image, trace, ids, return_diagnostics=False):
        # The API has no GT argument. Noise is invariant to epoch/rank/batch order.
        if return_diagnostics:
            torch.cuda.synchronize(image.device)
            profile_start = time.perf_counter()
        noise_image = torch.stack([stable_noise(image, ids, self.config['evaluation']['seed'], f'image-{r}') for r in range(2)])
        noisy_mask = stable_noise(image, ids, self.config['evaluation']['seed'], 'mask')
        responses, features, estimates, _ = self.reference_path(image, 4, noise_image, keep_all_features=True)
        context = self.evidence.prepare(image * 2 - 1)
        stream = self.evidence.begin(context)
        measured = {m: self.evidence.append(stream, responses[:, m-1]) for m in range(1, 5)}
        if return_diagnostics:
            torch.cuda.synchronize(image.device)
            profile_reference = time.perf_counter()
        steps = torch.linspace(1., 0., 11, device=image.device).mul(math.pi/2).sin()
        previous = None
        for j in range(10):
            m = min(4, 1 + 4*j//10)
            logsnr_t = self.schedule(steps[j].expand(len(image)))
            logsnr_s = self.schedule(steps[j+1].expand(len(image)))
            output = self.task(image, trace, noisy_mask, logsnr_t, features[m-1], estimates[m-1], m, previous)
            controlled, gamma = self.control(output['logits_base'], measured[m], j, m)
            probability = controlled.sigmoid()
            noisy_mask = ddim_step(noisy_mask, probability, logsnr_t, logsnr_s, final=(j == 9))
            previous = (F.interpolate(probability, (32, 32), mode='bilinear', align_corners=False) >= .5).float()
        if not torch.isfinite(probability).all():
            raise FloatingPointError('Non-finite final mask probability')
        if return_diagnostics:
            torch.cuda.synchronize(image.device)
            profile_end = time.perf_counter()
            return probability, {'P_base': output['logits_base'].float().sigmoid(),
                                 'P_ctrl': probability, **measured[4], 'gamma': gamma,
                                 'sample_profile': {'cold_reference_and_measurement_seconds': profile_reference-profile_start,
                                    'mask_ten_steps_with_reference_reuse_seconds': profile_end-profile_reference,
                                    'total_seconds': profile_end-profile_start, 'reference_forwards': 8,
                                    'image_count': len(image), 'reference_feature_requests': 20,
                                    'reference_feature_reuse_hits': 12, 'cross_call_cache': False}}
        return probability
