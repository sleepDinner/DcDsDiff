import argparse
import csv
import io
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import numpy as np
import torch
from omegaconf import OmegaConf
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
from tools.generate_git10k_aux import (  # noqa: E402
    build_generation_jobs,
    pair_image_mask_files,
    process_generation_job,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_DATASET_KEYS = ("BN", "PE", "IA", "PP")
PAPER_DATASET_SOURCE_CANDIDATES = {
    "BN": ("BN", "RBN"),
    "PE": ("PE", "EI", "Flux", "e", "t", "z"),
    "IA": ("IA",),
    "PP": ("PP",),
}
DATASET_SUBDIRS = {
    "image_root": "f",
    "gt_root": "m",
    "de_root": "d",
    "trace_root": "t",
}
EXTERNAL_IMAGE_DIR_NAMES = (
    "f",
    "image",
    "images",
    "img",
    "imgs",
    "jpegimages",
    "rgb",
)
EXTERNAL_MASK_DIR_NAMES = (
    "m",
    "mask",
    "masks",
    "gt",
    "gts",
    "groundtruth",
    "ground_truth",
    "annotation",
    "annotations",
    "label",
    "labels",
)

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


def binary_auc(pred, gt):
    pred = normalize_to_unit(pred).reshape(-1)
    gt = (normalize_to_unit(gt).reshape(-1) >= 0.5)
    pos_count = int(gt.sum())
    neg_count = int(gt.size - pos_count)
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    order = np.argsort(pred, kind="mergesort")
    sorted_pred = pred[order]
    ranks = np.empty(len(pred), dtype=np.float64)
    start = 0
    while start < len(pred):
        end = start + 1
        while end < len(pred) and sorted_pred[end] == sorted_pred[start]:
            end += 1
        # 1-based average rank for tied prediction scores.
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end

    pos_rank_sum = ranks[gt].sum(dtype=np.float64)
    auc = (pos_rank_sum - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count)
    return float(auc)


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
    auc_values = []
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
        auc_values.append(binary_auc(pred_unit, gt_unit))
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
        "AUC": float(np.mean(auc_values)),
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


def path_with_trailing_slash(path):
    return Path(path).as_posix().rstrip("/") + "/"


def safe_dataset_dir_name(name):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe_name.strip("_") or "dataset"


def infer_test_diff_root(cfg):
    template_key = "Mix" if "Mix" in cfg.test_dataset else next(iter(cfg.test_dataset.keys()))
    return Path(cfg.test_dataset[template_key].params.image_root).parent.parent


def get_dataset_cfg(cfg, dataset_key):
    if dataset_key not in cfg.test_dataset:
        template_key = "Mix" if "Mix" in cfg.test_dataset else next(iter(cfg.test_dataset.keys()))
        dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.test_dataset[template_key], resolve=True))
        test_diff_root = infer_test_diff_root(cfg)
        for root_name, subdir in DATASET_SUBDIRS.items():
            dataset_cfg.params[root_name] = path_with_trailing_slash(test_diff_root / dataset_key / subdir)
        return dataset_cfg
    return cfg.test_dataset[dataset_key]


def image_count(root):
    root = Path(root)
    if not root.is_dir():
        return 0
    return sum(1 for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def child_dirs(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def find_named_child_dir(root, names):
    root = Path(root)
    names_lower = {name.lower() for name in names}
    for path in child_dirs(root):
        if path.name.lower() in names_lower and image_count(path) > 0:
            return path
    return None


def infer_external_image_mask_roots(root):
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"External dataset root does not exist: {root}")

    image_root = find_named_child_dir(root, EXTERNAL_IMAGE_DIR_NAMES)
    mask_root = find_named_child_dir(root, EXTERNAL_MASK_DIR_NAMES)
    if image_root is not None and mask_root is not None:
        return image_root, mask_root

    child_summary = ", ".join(f"{path.name}({image_count(path)})" for path in child_dirs(root)) or "none"
    raise SystemExit(
        f"Cannot infer image/mask folders under external dataset root: {root}\n"
        f"Child folders: {child_summary}\n"
        "Use an explicit pair instead: --external-dataset NAME=/path/to/images,/path/to/masks"
    )


def parse_external_dataset_spec(spec):
    if "=" not in spec:
        raise SystemExit(
            f"Invalid --external-dataset value: {spec}\n"
            "Expected NAME=/dataset/root or NAME=/path/to/images,/path/to/masks"
        )
    name, path_spec = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"Invalid --external-dataset value with empty dataset name: {spec}")
    paths = [Path(item.strip()) for item in path_spec.split(",") if item.strip()]
    if len(paths) == 1:
        image_root, mask_root = infer_external_image_mask_roots(paths[0])
    elif len(paths) == 2:
        image_root, mask_root = paths
    else:
        raise SystemExit(
            f"Invalid --external-dataset paths for {name}: {path_spec}\n"
            "Expected one root path or two comma-separated paths."
        )
    return name, image_root, mask_root


def process_generation_jobs(jobs, num_workers):
    if num_workers <= 1:
        for job in jobs:
            process_generation_job(job)
        return
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in executor.map(process_generation_job, jobs, chunksize=8):
            pass


def prepare_external_dataset(
    name,
    image_root,
    mask_root,
    work_root,
    *,
    detail_radius,
    edge_kernel,
    cutoff_ratio,
    boost,
    overwrite,
    num_workers,
):
    pairs = pair_image_mask_files(Path(image_root), Path(mask_root))
    if not pairs:
        raise SystemExit(f"No matched image/mask pairs found for external dataset {name}: {image_root} <-> {mask_root}")

    dataset_root = Path(work_root) / safe_dataset_dir_name(name)
    jobs = build_generation_jobs(
        pairs,
        out_root=dataset_root,
        layout="flat",
        assignment={},
        detail_radius=detail_radius,
        edge_kernel=edge_kernel,
        cutoff_ratio=cutoff_ratio,
        boost=boost,
        overwrite=overwrite,
        copy_inputs=True,
        with_test_mix=False,
        mix_name="Mix",
    )
    process_generation_jobs(jobs, num_workers=num_workers)
    return dataset_root, len(pairs)


def configure_external_datasets(cfg):
    if not cfg.external_dataset:
        return None

    template_key = "Mix" if "Mix" in cfg.test_dataset else next(iter(cfg.test_dataset.keys()))
    template_cfg = OmegaConf.create(OmegaConf.to_container(cfg.test_dataset[template_key], resolve=True))
    work_root = Path(cfg.external_work_root) if cfg.external_work_root else Path(cfg.results_folder) / "external_datasets"

    external_dataset_cfg = OmegaConf.create({})
    dataset_keys = []
    prepared = {}
    for spec in cfg.external_dataset:
        name, image_root, mask_root = parse_external_dataset_spec(spec)
        dataset_root, num_images = prepare_external_dataset(
            name,
            image_root,
            mask_root,
            work_root,
            detail_radius=cfg.external_detail_radius,
            edge_kernel=cfg.external_edge_kernel,
            cutoff_ratio=cfg.external_cutoff_ratio,
            boost=cfg.external_boost,
            overwrite=cfg.external_overwrite,
            num_workers=cfg.external_num_workers,
        )
        dataset_cfg = OmegaConf.create(OmegaConf.to_container(template_cfg, resolve=True))
        for root_name, subdir in DATASET_SUBDIRS.items():
            dataset_cfg.params[root_name] = path_with_trailing_slash(dataset_root / subdir)
        external_dataset_cfg[name] = dataset_cfg
        dataset_keys.append(name)
        prepared[name] = {
            "source_image_root": str(image_root),
            "source_mask_root": str(mask_root),
            "prepared_root": str(dataset_root),
            "num_images": num_images,
        }

    cfg.test_dataset = external_dataset_cfg
    cfg.dataset_keys = dataset_keys
    return prepared


def dataset_has_images(cfg, dataset_key):
    dataset_cfg = get_dataset_cfg(cfg, dataset_key)
    return image_count(dataset_cfg.params.image_root) > 0


def available_dataset_rows(cfg):
    test_diff_root = infer_test_diff_root(cfg)
    rows = []
    if not test_diff_root.is_dir():
        return test_diff_root, rows
    for dataset_root in sorted(path for path in test_diff_root.iterdir() if path.is_dir()):
        rows.append(
            {
                "key": dataset_root.name,
                "f": image_count(dataset_root / "f"),
                "m": image_count(dataset_root / "m"),
                "d": image_count(dataset_root / "d"),
                "t": image_count(dataset_root / "t"),
            }
        )
    return test_diff_root, rows


def format_available_datasets(cfg):
    test_diff_root, rows = available_dataset_rows(cfg)
    lines = [f"Available test datasets under {test_diff_root}:"]
    if not rows:
        lines.append("  none")
        return "\n".join(lines)

    table_rows = [("Dataset", "f", "m", "d", "t")]
    for row in rows:
        table_rows.append((row["key"], str(row["f"]), str(row["m"]), str(row["d"]), str(row["t"])))
    widths = [max(len(row[index]) for row in table_rows) for index in range(len(table_rows[0]))]
    for row in table_rows:
        lines.append("  " + "  ".join(value.rjust(widths[index]) if index > 0 else value.ljust(widths[index])
                                      for index, value in enumerate(row)))
    return "\n".join(lines)


def resolve_dataset_sources(cfg, dataset_key):
    if dataset_key == "PE" and dataset_has_images(cfg, "PE"):
        return ["PE"]
    candidates = PAPER_DATASET_SOURCE_CANDIDATES.get(dataset_key, (dataset_key,))
    sources = [candidate for candidate in candidates if dataset_has_images(cfg, candidate)]
    return sources or [dataset_key]


def resolve_requested_sources(cfg, dataset_keys):
    return {dataset_key: resolve_dataset_sources(cfg, dataset_key) for dataset_key in dataset_keys}


def validate_dataset_roots(cfg, dataset_keys, *, skip_inference, multi_dataset):
    missing = []
    dataset_sources = resolve_requested_sources(cfg, dataset_keys)
    for dataset_key, source_keys in dataset_sources.items():
        if not source_keys:
            missing.append((dataset_key, "sources", "no matching source folders"))
            continue
        for source_key in source_keys:
            dataset_cfg = get_dataset_cfg(cfg, source_key)
            roots_to_check = {
                "image_root": dataset_cfg.params.image_root,
                "gt_root": dataset_cfg.params.gt_root,
                "de_root": dataset_cfg.params.de_root,
                "trace_root": dataset_cfg.params.trace_root,
            }
            if skip_inference:
                roots_to_check = {"gt_root": dataset_cfg.params.gt_root}
            for root_name, root_path in roots_to_check.items():
                if not Path(root_path).is_dir():
                    missing.append((dataset_key, f"{source_key}.{root_name}", str(root_path)))

            if skip_inference:
                pred_root = prediction_root_for_dataset(cfg, source_key, multi_dataset)
                if not pred_root.is_dir():
                    missing.append((dataset_key, f"{source_key}.pred_root", str(pred_root)))

    if not missing:
        return dataset_sources

    lines = ["Missing folders for requested evaluation datasets:"]
    for dataset_key, root_name, root_path in missing:
        lines.append(f"  {dataset_key}.{root_name}: {root_path}")
    lines.append("")
    lines.append(format_available_datasets(cfg))
    lines.append("")
    lines.append("The paper names are mapped to released folder prefixes as follows when those folders exist:")
    for logical_key, source_keys in dataset_sources.items():
        lines.append(f"  {logical_key}: {' + '.join(source_keys)}")
    lines.append("")
    lines.append("If a source folder is missing, inspect the source filenames and regenerate the auxiliary dataset:")
    lines.append("  find /data0/hl/DcDsDiff-and-GIT10K/GIT10K/Image -maxdepth 1 -type f | sed -E 's|.*/([A-Za-z]+).*|\\1|' | sort | uniq -c")
    lines.append("  python tools/generate_git10k_aux.py --image-root /data0/hl/DcDsDiff-and-GIT10K/GIT10K/Image --mask-root /data0/hl/DcDsDiff-and-GIT10K/GIT10K/Mask --out-root /data0/hl/Diff_dataset --layout project --with-test-mix --train-ratio 0.9 --overwrite")
    raise SystemExit("\n".join(lines))


def weighted_average_results(source_results, source_keys):
    total_images = int(sum(source_results[key]["num_images"] for key in source_keys))
    grouped = {
        "num_images": total_images,
        "sources": list(source_keys),
    }
    for metric in ("F1", "IoU", "AUC", "MAE", "inference_MAE"):
        weighted_sum = 0.0
        weight_sum = 0
        for source_key in source_keys:
            result = source_results[source_key]
            if metric not in result:
                continue
            weight = result["num_images"]
            weighted_sum += result[metric] * weight
            weight_sum += weight
        if weight_sum:
            grouped[metric] = float(weighted_sum / weight_sum)
    grouped["missing_predictions"] = {
        key: source_results[key]["missing_predictions"]
        for key in source_keys
        if source_results[key].get("missing_predictions")
    }
    grouped["extra_predictions"] = {
        key: source_results[key]["extra_predictions"]
        for key in source_keys
        if source_results[key].get("extra_predictions")
    }
    return grouped

def build_test_loader(cfg, dataset_key):
    test_dataset = instantiate_from_config(get_dataset_cfg(cfg, dataset_key))
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


def run_inference_for_dataset(trainer, cfg, dataset_key, pred_root):
    test_loader = build_test_loader(cfg, dataset_key)
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
    return mae


def average_dataset_results(dataset_results, dataset_keys):
    average = {"num_images": int(sum(dataset_results[key]["num_images"] for key in dataset_keys))}
    for metric in ("F1", "IoU", "AUC"):
        average[metric] = float(sum(dataset_results[key][metric] for key in dataset_keys) / len(dataset_keys))
    return average


def format_metric(value):
    if value != value:
        return "nan"
    return f"{value:.4f}"


def format_results_table(dataset_results, average_results, dataset_keys, threshold):
    rows = []
    rows.append(("Dataset", "Sources", "Images", "F1", "IoU", "AUC"))
    rows.append(("-" * 7, "-" * 7, "-" * 6, "-" * 6, "-" * 6, "-" * 6))
    for key in dataset_keys:
        result = dataset_results[key]
        rows.append(
            (
                key,
                "+".join(result.get("sources", [key])),
                str(result["num_images"]),
                format_metric(result["F1"]),
                format_metric(result["IoU"]),
                format_metric(result["AUC"]),
            )
        )
    rows.append(("-" * 7, "-" * 7, "-" * 6, "-" * 6, "-" * 6, "-" * 6))
    rows.append(
        (
            "Average",
            "macro",
            str(average_results["num_images"]),
            format_metric(average_results["F1"]),
            format_metric(average_results["IoU"]),
            format_metric(average_results["AUC"]),
        )
    )

    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = [f"Evaluation metrics (threshold={threshold:.3f})"]
    for row in rows:
        line = "  ".join(value.rjust(widths[index]) if index > 0 else value.ljust(widths[index])
                         for index, value in enumerate(row))
        lines.append(line)
    lines.append("Average = arithmetic mean of the listed datasets; no extra Mix evaluation is run.")
    return "\n".join(lines)


def format_csv_metric(value):
    if value != value:
        return "nan"
    return f"{value:.6f}"


def format_results_csv(dataset_results, average_results, dataset_keys):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["Dataset", "Sources", "Images", "F1", "IoU", "AUC"])
    for key in dataset_keys:
        result = dataset_results[key]
        writer.writerow(
            [
                key,
                "+".join(result.get("sources", [key])),
                result["num_images"],
                format_csv_metric(result["F1"]),
                format_csv_metric(result["IoU"]),
                format_csv_metric(result["AUC"]),
            ]
        )
    writer.writerow(
        [
            "Average",
            "macro",
            average_results["num_images"],
            format_csv_metric(average_results["F1"]),
            format_csv_metric(average_results["IoU"]),
            format_csv_metric(average_results["AUC"]),
        ]
    )
    return output.getvalue()


def make_results_payload(dataset_results, source_results, average_results, dataset_keys, threshold, checkpoint):
    return {
        "dataset_keys": list(dataset_keys),
        "threshold": float(threshold),
        "checkpoint": str(checkpoint),
        "datasets": dataset_results,
        "sources": source_results,
        "average": average_results,
        "average_note": "Average is computed from dataset rows only.",
        "source_mapping_note": "Paper dataset names are evaluated from the released source-prefix folders listed in each row.",
    }


def save_results_report(report_dir, table_text, payload, csv_text, prefix="evaluation_results"):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "txt": report_dir / f"{prefix}.txt",
        "json": report_dir / f"{prefix}.json",
        "csv": report_dir / f"{prefix}.csv",
    }
    paths["txt"].write_text(table_text + "\n", encoding="utf-8")
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["csv"].write_text(csv_text, encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def prediction_root_for_dataset(cfg, dataset_key, multi_dataset):
    if cfg.pred_root is not None:
        root = Path(cfg.pred_root)
        return root / dataset_key if multi_dataset else root
    return Path(cfg.results_folder) / dataset_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./model-best.pt")
    parser.add_argument(
        "--dataset-key",
        "--dataset-keys",
        dest="dataset_keys",
        nargs="+",
        default=list(DEFAULT_DATASET_KEYS),
        help="Dataset keys to evaluate. Defaults to BN PE IA PP.",
    )
    parser.add_argument("--results_folder", type=str, default="./eval_results")
    parser.add_argument("--report-dir", dest="report_dir", type=str, default=None)
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
    parser.add_argument(
        "--external-dataset",
        dest="external_dataset",
        action="append",
        default=None,
        help=(
            "Evaluate an external image/mask dataset. Use NAME=/dataset/root when it contains images/masks "
            "folders, or NAME=/path/to/images,/path/to/masks for explicit roots. Can be repeated."
        ),
    )
    parser.add_argument("--external-work-root", dest="external_work_root", type=str, default=None)
    parser.add_argument("--external-num-workers", dest="external_num_workers", type=int, default=1)
    parser.add_argument("--external-overwrite", dest="external_overwrite", action="store_true")
    parser.add_argument("--external-detail-radius", dest="external_detail_radius", type=float, default=15.0)
    parser.add_argument("--external-edge-kernel", dest="external_edge_kernel", type=int, default=3)
    parser.add_argument("--external-cutoff-ratio", dest="external_cutoff_ratio", type=float, default=0.5)
    parser.add_argument("--external-boost", dest="external_boost", type=float, default=10.0)
    parser.add_argument("--json", dest="print_json", action="store_true", help="Also print machine-readable JSON.")
    parser.add_argument("--list-datasets", dest="list_datasets", action="store_true", help="List available Test/Diff subsets and exit.")

    cfg = add_args(parser)
    set_random_seed(7)

    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    if cfg.batch_ensemble and cfg.time_ensemble:
        raise SystemExit("Cannot use both --batch-ensemble and --time-ensemble")

    external_prepared = configure_external_datasets(cfg)
    dataset_keys = list(cfg.dataset_keys)
    if cfg.list_datasets:
        print(format_available_datasets(cfg))
        return

    candidate_sources = resolve_requested_sources(cfg, dataset_keys)
    unique_source_keys = sorted({source_key for source_keys in candidate_sources.values() for source_key in source_keys})
    source_multi_dataset = len(unique_source_keys) > 1
    dataset_sources = validate_dataset_roots(
        cfg,
        dataset_keys,
        skip_inference=cfg.skip_inference,
        multi_dataset=source_multi_dataset,
    )

    trainer = None
    if not cfg.skip_inference:
        trainer = build_trainer(cfg)
        trainer.load(pretrained_path=cfg.checkpoint)

    dataset_results = {}
    source_results = {}
    for dataset_key, source_keys in dataset_sources.items():
        for source_key in source_keys:
            if source_key in source_results:
                continue
            dataset_cfg = get_dataset_cfg(cfg, source_key)
            gt_root = Path(dataset_cfg.params.gt_root)
            pred_root = prediction_root_for_dataset(cfg, source_key, source_multi_dataset)

            inference_mae = None
            if not cfg.skip_inference:
                inference_mae = run_inference_for_dataset(trainer, cfg, source_key, pred_root)
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
            results["dataset_key"] = source_key
            results["gt_root"] = str(gt_root)
            results["pred_root"] = str(pred_root)
            results["checkpoint"] = str(cfg.checkpoint)
            source_results[source_key] = results

        dataset_results[dataset_key] = weighted_average_results(source_results, source_keys)
        dataset_results[dataset_key]["dataset_key"] = dataset_key
        dataset_results[dataset_key]["checkpoint"] = str(cfg.checkpoint)

    average_results = average_dataset_results(dataset_results, dataset_keys)
    table = format_results_table(dataset_results, average_results, dataset_keys, threshold=cfg.threshold)
    payload = make_results_payload(
        dataset_results,
        source_results,
        average_results,
        dataset_keys,
        threshold=cfg.threshold,
        checkpoint=cfg.checkpoint,
    )
    if external_prepared is not None:
        payload["external_datasets"] = external_prepared
        payload["source_mapping_note"] = "External datasets were converted to DcDsDiff f/m/d/t folders before inference."
    csv_text = format_results_csv(dataset_results, average_results, dataset_keys)
    report_dir = Path(cfg.report_dir) if cfg.report_dir else Path(cfg.results_folder)
    saved_paths = save_results_report(report_dir, table, payload, csv_text)

    print(table)
    print("Saved evaluation reports:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")

    if cfg.print_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
