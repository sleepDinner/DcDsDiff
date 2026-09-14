"""The active DcDsDiff training and temporal-ensemble inference interface."""

import torch
import torch.nn.functional as F
from torch import nn


def normalize_to_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def modification_train_val_forward(model: nn.Module, gt=None, image=None, de=None,
                                   trace=None, seg=None, **kwargs):
    """Generate mask/detail jointly; inference never uses ground-truth content."""
    if image is None or trace is None:
        raise ValueError("RGB and HFVG conditions are required")
    if model.training:
        if gt is None or de is None:
            raise ValueError("Training requires mask and detail targets")
        return model(gt, de, image, trace, seg=seg, **kwargs)

    time_ensemble = kwargs.pop("time_ensemble", False)
    gt_sizes = kwargs.pop("gt_sizes", None)
    pred_gt, pred_de = model.sample(image, trace, **kwargs)
    if time_ensemble:
        steps = model.num_sample_steps
        if len(model.history) != 2 * steps or steps < 1:
            raise ValueError(f"Expected {2 * steps} alternating mask/detail states, got {len(model.history)}")
        batch_size = image.shape[0]
        if gt_sizes is None or len(gt_sizes) != batch_size:
            raise ValueError("Temporal ensemble requires one original output size per image")
        if any(state.ndim != 4 or state.shape[:2] != (batch_size, 1) for state in model.history):
            raise ValueError("Temporal states must have shape [batch, 1, height, width]")
        masks = torch.cat(model.history[0::2], dim=1).detach().cpu()
        details = torch.cat(model.history[1::2], dim=1).detach().cpu()
        mask_mean = masks.mean(dim=1, keepdim=True)
        detail_mean = details.mean(dim=1, keepdim=True)
        pred_gt, pred_de = [], []
        for index, size in enumerate(gt_sizes):
            # Preserve the public mask mean/minmax/positive-majority mathematics.
            mask = F.interpolate(mask_mean[index].unsqueeze(0), size=size,
                                 mode="bilinear", align_corners=False)
            mask = normalize_to_01(mask)
            steps_at_size = F.interpolate(masks[index].unsqueeze(0), size=size,
                                          mode="bilinear", align_corners=False)
            majority = ((steps_at_size > 0).float().mean(dim=1, keepdim=True) > 0.5).float()
            pred_gt.append(mask * majority)
            # The paper describes one averaged detail image. The previous
            # per-step multiplication broadcast this output to T channels.
            pred_de.append(F.interpolate(detail_mean[index].unsqueeze(0), size=size,
                                         mode="bilinear", align_corners=False))

    return {
        "image": image, "pred_gt": pred_gt, "pred_de": pred_de,
        "gt": gt, "de": de, "trace": trace,
    }
