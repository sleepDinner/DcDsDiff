"""Versioned TECT mask objectives; existing protocols retain the original loss."""
import torch
import torch.nn.functional as F

from model.loss import structure_loss


def tect_mask_loss(logits, gt, policy):
    """Keep the per-image mean; v1 omits IoU only for an exactly empty input target.

    Emptiness is measured on the supplied training grid, regardless of native
    annotation size or image authenticity. No threshold or class renormalization
    is applied. Callers keep the independent Image MSE term unchanged.
    """
    if not isinstance(policy, str) or policy not in ('structure_v1', 'empty_target_bce_v1'):
        raise ValueError(f'Unknown TECT mask loss policy: {policy!r}')
    if policy == 'structure_v1':
        return structure_loss(logits, gt)

    # Match the original structure_loss operations for every nonempty target.
    weight = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
    wbce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    probability = torch.sigmoid(logits)
    inter = ((probability * gt) * weight).sum(dim=(2, 3))
    union = ((probability + gt) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    nonempty = gt.ne(0).flatten(1).any(dim=1, keepdim=True)
    return (wbce + nonempty * wiou).mean()
