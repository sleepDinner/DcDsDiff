import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.collate_utils import collate  # noqa: E402
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str  # noqa: E402
from utils.init_utils import add_args  # noqa: E402
from utils.train_utils import set_random_seed  # noqa: E402
from utils.trainer import Trainer  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

try:
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - Pillow < 9
    BILINEAR = Image.BILINEAR


def normalize_to_unit(array):
    array = np.asarray(array, dtype=np.float32)
    if array.size == 0:
        return array
    max_value = float(array.max())
    if max_value > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def binary_f1_iou(pred, gt, threshold=0.5):
    pred = normalize_to_unit(pred) >= threshold
    gt = normalize_to_unit(gt) >= 0.5

    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.float64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.float64)

    f1_denominator = (2.0 * tp) + fp + fn
    union = tp + fp + fn
    f1 = 1.0 if f1_denominator == 0 else (2.0 * tp) / f1_denominator
    iou = 1.0 if union == 0 else tp / union
    return float(f1), float(iou)


def collect_mask_files(root):
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Mask/prediction folder does not exist: {root}")

    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files[path.stem] = path
    return files


def load_grayscale(path, size=None):
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, BILINEAR)
    return np.asarray(image, dtype=np.float32)


def evaluate_prediction_folder(gt_root, pred_root, threshold=0.5, sweep_thresholds=False, threshold_steps=256):
    gt_files = collect_mask_files(gt_root)
    pred_files = collect_mask_files(pred_root)
    common_stems = sorted(set(gt_files).intersection(pred_files))
    if not common_stems:
        raise SystemExit(f"No matching prediction/GT filenames under {pred_root} and {gt_root}")

    f1_values = []
    iou_values = []
    mae_values = []
    sweep = {
        "best_F1": 0.0,
        "best_F1_threshold": threshold,
        "best_IoU": 0.0,
        "best_IoU_threshold": threshold,
    }

    thresholds = np.linspace(0.0, 1.0, threshold_steps) if sweep_thresholds else []
    sweep_f1_sums = np.zeros(len(thresholds), dtype=np.float64)
    sweep_iou_sums = np.zeros(len(thresholds), dtype=np.float64)

    for stem in common_stems:
        gt_image = Image.open(gt_files[stem]).convert("L")
        gt = np.asarray(gt_image, dtype=np.float32)
        pred = load_grayscale(pred_files[stem], size=gt_image.size)

        pred_unit = normalize_to_unit(pred)
        gt_unit = normalize_to_unit(gt)
        f1, iou = binary_f1_iou(pred_unit, gt_unit, threshold=threshold)
        f1_values.append(f1)
        iou_values.append(iou)
        mae_values.append(float(np.mean(np.abs(pred_unit - (gt_unit >= 0.5).astype(np.float32)))))

        for idx, sweep_threshold in enumerate(thresholds):
            sweep_f1, sweep_iou = binary_f1_iou(pred_unit, gt_unit, threshold=float(sweep_threshold))
            sweep_f1_sums[idx] += sweep_f1
            sweep_iou_sums[idx] += sweep_iou

    results = {
        "num_images": len(common_stems),
        "threshold": float(threshold),
        "F1": float(np.mean(f1_values)),
        "IoU": float(np.mean(iou_values)),
        "MAE": float(np.mean(mae_values)),
        "missing_predictions": sorted(set(gt_files).difference(pred_files)),
        "extra_predictions": sorted(set(pred_files).difference(gt_files)),
    }

    if sweep_thresholds:
        mean_f1 = sweep_f1_sums / len(common_stems)
        mean_iou = sweep_iou_sums / len(common_stems)
        best_f1_idx = int(np.argmax(mean_f1))
        best_iou_idx = int(np.argmax(mean_iou))
        sweep.update(
            {
                "best_F1": float(mean_f1[best_f1_idx]),
                "best_F1_threshold": float(thresholds[best_f1_idx]),
                "best_IoU": float(mean_iou[best_iou_idx]),
                "best_IoU_threshold": float(thresholds[best_iou_idx]),
            }
        )
        results.update(sweep)

    return results


def build_test_loader(cfg):
    dataset_key = cfg.dataset_key
    if dataset_key not in cfg.test_dataset:
        available = ", ".join(cfg.test_dataset.keys())
        raise SystemExit(f"Unknown dataset key '{dataset_key}'. Available test datasets: {available}")

    test_dataset = instantiate_from_config(cfg.test_dataset[dataset_key])
    return DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )


def build_trainer(cfg):
    cond_uvit = instantiate_from_config(
        cfg.cond_uvit,
        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass),
    )
    model = recurse_instantiate_from_config(cfg.model, unet=cond_uvit)
    diffusion_model = instantiate_from_config(cfg.diffusion_model, model=model)
    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    return Trainer(
        diffusion_model,
        train_loader=None,
        test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )


def run_inference(cfg, pred_root):
    test_loader = build_test_loader(cfg)
    trainer = build_trainer(cfg)
    trainer.load(pretrained_path=cfg.checkpoint)
    test_loader = trainer.accelerator.prepare(test_loader)

    pred_root.mkdir(parents=True, exist_ok=True)
    if cfg.batch_ensemble:
        mae, _ = trainer.val_batch_ensemble(
            model=trainer.model,
            test_data_loader=test_loader,
            accelerator=trainer.accelerator,
            thresholding=False,
            save_to=pred_root,
        )
    elif cfg.time_ensemble:
        mae, _ = trainer.val_time_ensemble(
            model=trainer.model,
            test_data_loader=test_loader,
            accelerator=trainer.accelerator,
            thresholding=False,
            save_to=pred_root,
        )
    else:
        mae, _ = trainer.val(
            model=trainer.model,
            test_data_loader=test_loader,
            accelerator=trainer.accelerator,
            thresholding=False,
            save_to=pred_root,
        )
    trainer.accelerator.wait_for_everyone()
    return trainer, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./model-best.pt")
    parser.add_argument("--dataset-key", dest="dataset_key", type=str, default="Mix")
    parser.add_argument("--results_folder", type=str, default="./eval_results")
    parser.add_argument("--pred-root", dest="pred_root", type=str, default=None)
    parser.add_argument("--skip-inference", dest="skip_inference", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulate_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_sample_steps", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", dest="sweep_thresholds", action="store_true")
    parser.add_argument("--threshold-steps", dest="threshold_steps", type=int, default=256)
    parser.add_argument("--batch-ensemble", "--batch_ensemble", dest="batch_ensemble", action="store_true")
    parser.add_argument("--time-ensemble", "--time_ensemble", dest="time_ensemble", action="store_true", default=True)
    parser.add_argument("--no-time-ensemble", "--no_time_ensemble", dest="time_ensemble", action="store_false")

    cfg = add_args(parser)
    set_random_seed(7)

    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    if cfg.batch_ensemble and cfg.time_ensemble:
        raise SystemExit("Cannot use both --batch-ensemble and --time-ensemble")

    dataset_cfg = cfg.test_dataset[cfg.dataset_key]
    gt_root = Path(dataset_cfg.params.gt_root)
    pred_root = Path(cfg.pred_root) if cfg.pred_root else Path(cfg.results_folder) / cfg.dataset_key

    inference_mae = None
    if not cfg.skip_inference:
        trainer, inference_mae = run_inference(cfg, pred_root)
        if not trainer.accelerator.is_main_process:
            return
    elif not pred_root.is_dir():
        raise SystemExit(f"--skip-inference was set but prediction folder does not exist: {pred_root}")

    results = evaluate_prediction_folder(
        gt_root=gt_root,
        pred_root=pred_root,
        threshold=cfg.threshold,
        sweep_thresholds=cfg.sweep_thresholds,
        threshold_steps=cfg.threshold_steps,
    )
    if inference_mae is not None:
        results["inference_MAE"] = float(inference_mae)
    results["dataset_key"] = cfg.dataset_key
    results["gt_root"] = str(gt_root)
    results["pred_root"] = str(pred_root)
    results["checkpoint"] = str(cfg.checkpoint)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
