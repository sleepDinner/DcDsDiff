"""Original-coordinate binary localization metrics and exact-coverage reducers."""
from __future__ import annotations

import math
from collections import Counter
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def per_image_metrics(probability, gt, valid=None, threshold=0.5, boundary_ratio=0.005):
    """Threshold probabilities once; ignore pixels never enter any count.

    Boundary is a one-pixel 8-connected inner contour. Matching uses Euclidean
    distance <= max(1, ceil(boundary_ratio * original-image diagonal)).
    """
    p = np.asarray(probability, dtype=np.float64).squeeze()
    target = np.asarray(gt).squeeze().astype(bool)
    keep = np.ones_like(target) if valid is None else np.asarray(valid).squeeze().astype(bool)
    if p.ndim != 2 or p.shape != target.shape or keep.shape != target.shape:
        raise ValueError(f"Metric shape mismatch: {p.shape}, {target.shape}, {keep.shape}")
    if not keep.any():
        raise ValueError("ALL_IGNORED_SAMPLE: no valid pixels")
    if not np.isfinite(p[keep]).all() or np.any((p[keep] < 0) | (p[keep] > 1)):
        raise ValueError("Expected finite probabilities in [0,1]")
    prediction = (p >= threshold) & keep
    target = target & keep
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target & keep))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target & keep))
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    # Ignore regions cannot create artificial contours or serve as matches.
    edge_valid = binary_erosion(keep, np.ones((3, 3), bool), border_value=1)
    pred_edge = (prediction & ~binary_erosion(prediction, np.ones((3, 3), bool), border_value=0)) & edge_valid
    true_edge = (target & ~binary_erosion(target, np.ones((3, 3), bool), border_value=0)) & edge_valid
    pred_count, true_count = int(pred_edge.sum()), int(true_edge.sum())
    tolerance = max(1, int(math.ceil(boundary_ratio * math.hypot(*target.shape))))
    if pred_count and true_count:
        matched_pred = int(np.count_nonzero(pred_edge & (distance_transform_edt(~true_edge) <= tolerance)))
        matched_true = int(np.count_nonzero(true_edge & (distance_transform_edt(~pred_edge) <= tolerance)))
        precision, recall = matched_pred / pred_count, matched_true / true_count
        bf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        matched_pred = matched_true = 0
        bf1 = float(pred_count == true_count)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "valid_pixels": int(keep.sum()),
        "ignored_pixels": int((~keep).sum()), "pixel_f1": f1, "iou": iou,
        "boundary_f1": bf1, "boundary_tolerance_pixels": tolerance,
        "boundary_pred_count": pred_count, "boundary_gt_count": true_count,
        "boundary_matched_pred": matched_pred, "boundary_matched_gt": matched_true,
        "gt_positive_pixels": tp + fn, "pred_positive_pixels": tp + fp,
        "is_positive": bool(tp + fn), "is_authentic": not bool(tp + fn),
        "authentic_false_positive": bool(fp) if not (tp + fn) else None,
        "authentic_pixel_false_positive_rate": fp / (fp + tn) if not (tp + fn) else None,
        "mae": float(np.abs(p[keep] - target[keep]).mean()),
    }


def aggregate_dataset(rows, expected_ids=None):
    """Aggregate per-image rows after gathering all ranks; never average rank means."""
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot aggregate an empty dataset")
    ids = [row["id"] for row in rows]
    duplicate = [key for key, value in Counter(ids).items() if value != 1]
    if duplicate:
        raise ValueError(f"Duplicate evaluation IDs: {duplicate[:8]}")
    if expected_ids is not None:
        expected = list(expected_ids)
        if len(set(expected)) != len(expected) or set(ids) != set(expected):
            raise ValueError(f"Evaluation coverage mismatch: got {len(ids)}, expected {len(expected)}")
    result = {"count": len(rows), "id_unique_count": len(set(ids))}
    for metric in ("pixel_f1", "iou", "boundary_f1", "mae"):
        values = [float(row[metric]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite {metric}")
        result["dataset_" + metric] = math.fsum(values) / len(rows)
        result[metric + "_sum"] = math.fsum(values)
    for metric in ("tp", "fp", "fn", "tn", "valid_pixels", "ignored_pixels"):
        result[metric] = sum(int(row[metric]) for row in rows)
    denominator = 2 * result["tp"] + result["fp"] + result["fn"]
    result["pooled_pixel_f1"] = 2 * result["tp"] / denominator if denominator else 1.0
    positives = [row for row in rows if row["is_positive"]]
    authentic = [row for row in rows if row["is_authentic"]]
    result["positive_count"] = len(positives)
    result["positive_only_pixel_f1"] = math.fsum(row["pixel_f1"] for row in positives) / len(positives) if positives else None
    result["authentic_count"] = len(authentic)
    result["authentic_false_positive_images"] = sum(row["authentic_false_positive"] for row in authentic)
    result["authentic_image_false_positive_rate"] = result["authentic_false_positive_images"] / len(authentic) if authentic else None
    result["authentic_pixel_false_positive_rate"] = math.fsum(row["authentic_pixel_false_positive_rate"] for row in authentic) / len(authentic) if authentic else None
    return result


def aggregate_all8(results, expected_names=None):
    names = list(expected_names or ("Casiav1", "Columbia", "NIST16", "IMD2020", "DSO-1", "wild", "coverage", "Korus"))
    if len(names) != 8 or set(results) != set(names):
        raise ValueError("All8 selection requires all eight complete dataset results")
    if any(results[name]["count"] <= 0 for name in names):
        raise ValueError("All8 includes an empty dataset")
    summary = {"selection_protocol": "test_selected", "dataset_count": 8,
               "count": sum(results[name]["count"] for name in names)}
    for metric in ("pixel_f1", "iou", "boundary_f1"):
        values = [results[name]["dataset_" + metric] for name in names]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite All8 metric")
        summary["all8_macro_" + metric] = math.fsum(values) / 8
    for key in ("tp", "fp", "fn", "tn"):
        summary[key] = sum(results[name][key] for name in names)
    denominator = 2 * summary["tp"] + summary["fp"] + summary["fn"]
    summary["all8_pooled_pixel_f1"] = 2 * summary["tp"] / denominator if denominator else 1.0
    return summary


def final_auc_ap(probability, gt, valid=None):
    """Optional final-only metrics; one-class targets are NA, never invented scores."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    target = np.asarray(gt).astype(bool)
    keep = np.ones_like(target) if valid is None else np.asarray(valid).astype(bool)
    truth, pred = target[keep], np.asarray(probability)[keep]
    if not truth.size or np.unique(truth).size != 2:
        return {"auc": None, "ap": None, "auc_ap_valid": False}
    return {"auc": float(roc_auc_score(truth, pred)), "ap": float(average_precision_score(truth, pred)), "auc_ap_valid": True}
