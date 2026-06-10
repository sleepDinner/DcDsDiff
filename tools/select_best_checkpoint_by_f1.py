"""Select the epoch checkpoint with the highest F1 on a test split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_checkpoint_f1_iou import (  # noqa: E402
    average_dataset_results,
    build_trainer,
    evaluate_prediction_folder,
    format_metric,
    get_dataset_cfg,
    prediction_root_for_dataset,
    resolve_requested_sources,
    run_inference_for_dataset,
    validate_dataset_roots,
    weighted_average_results,
)
from utils.init_utils import add_args  # noqa: E402
from utils.train_utils import set_random_seed  # noqa: E402


CHECKPOINT_RE = re.compile(r"^model-(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointCandidate:
    epoch: int
    path: Path


def parse_checkpoint_epoch(path: Path) -> int | None:
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else None


def list_checkpoint_candidates(
    checkpoint_dir: Path,
    *,
    start_epoch: int | None = None,
    end_epoch: int | None = None,
    every_n_epochs: int = 1,
    limit: int = 0,
) -> list[CheckpointCandidate]:
    if every_n_epochs < 1:
        raise SystemExit("--every-n-epochs must be >= 1")
    if not checkpoint_dir.is_dir():
        raise SystemExit(f"Checkpoint folder does not exist: {checkpoint_dir}")

    candidates = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        epoch = parse_checkpoint_epoch(path)
        if epoch is None:
            continue
        if start_epoch is not None and epoch < start_epoch:
            continue
        if end_epoch is not None and epoch > end_epoch:
            continue
        candidates.append(CheckpointCandidate(epoch=epoch, path=path))

    candidates.sort(key=lambda item: item.epoch)
    if every_n_epochs > 1:
        base_epoch = candidates[0].epoch if start_epoch is None and candidates else start_epoch
        candidates = [
            candidate
            for candidate in candidates
            if (candidate.epoch - int(base_epoch)) % every_n_epochs == 0
        ]
    if limit > 0:
        candidates = candidates[:limit]
    if not candidates:
        raise SystemExit(f"No model-<epoch>.pt checkpoints found under {checkpoint_dir}")
    return candidates


def default_best_output_path(checkpoint_dir: Path, epoch: int) -> Path:
    return checkpoint_dir / f"epoch{epoch}_best.pth"


def metric_or_nan(results: dict, metric: str) -> float:
    value = results.get(metric)
    return float(value) if value is not None else float("nan")


def row_from_results(candidate: CheckpointCandidate, results: dict, selection_metric: str) -> dict:
    return {
        "epoch": candidate.epoch,
        "checkpoint": str(candidate.path),
        "num_images": int(results["num_images"]),
        "F1": metric_or_nan(results, "F1"),
        "IoU": metric_or_nan(results, "IoU"),
        "AUC": metric_or_nan(results, "AUC"),
        "MAE": metric_or_nan(results, "MAE"),
        "inference_MAE": metric_or_nan(results, "inference_MAE"),
        "selection_metric": selection_metric,
        "selection_score": metric_or_nan(results, selection_metric),
    }


def better_row(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    candidate_score = candidate["selection_score"]
    incumbent_score = incumbent["selection_score"]
    if candidate_score != candidate_score:
        return False
    if incumbent_score != incumbent_score:
        return True
    if candidate_score != incumbent_score:
        return candidate_score > incumbent_score
    return int(candidate["epoch"]) < int(incumbent["epoch"])


def safe_rmtree(path: Path, allowed_root: Path) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if not path.exists():
        return
    if path == allowed_root or allowed_root not in path.parents:
        raise RuntimeError(f"Refusing to remove path outside scan output folder: {path}")
    shutil.rmtree(path)


def write_scan_reports(results_folder: Path, rows: list[dict], best_row: dict, best_output: Path | None) -> None:
    results_folder.mkdir(parents=True, exist_ok=True)
    csv_path = results_folder / "checkpoint_f1_scan.csv"
    json_path = results_folder / "checkpoint_f1_scan.json"
    txt_path = results_folder / "checkpoint_f1_scan.txt"

    fields = [
        "epoch",
        "checkpoint",
        "num_images",
        "F1",
        "IoU",
        "AUC",
        "MAE",
        "inference_MAE",
        "selection_metric",
        "selection_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "best": best_row,
        "best_output": str(best_output) if best_output else None,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "Checkpoint F1 scan",
        f"Best epoch: {best_row['epoch']}",
        f"Best checkpoint: {best_row['checkpoint']}",
        f"Best copied to: {best_output}" if best_output else "Best copied to: disabled",
        (
            "Best metrics: "
            f"F1={format_metric(best_row['F1'])}, "
            f"IoU={format_metric(best_row['IoU'])}, "
            f"AUC={format_metric(best_row['AUC'])}, "
            f"MAE={format_metric(best_row['MAE'])}"
        ),
        "",
        "epoch,F1,IoU,AUC,MAE,inference_MAE",
    ]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["epoch"]),
                    format_metric(row["F1"]),
                    format_metric(row["IoU"]),
                    format_metric(row["AUC"]),
                    format_metric(row["MAE"]),
                    format_metric(row["inference_MAE"]),
                ]
            )
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_candidate(
    trainer,
    cfg,
    candidate: CheckpointCandidate,
    dataset_sources: dict[str, list[str]],
    dataset_keys: list[str],
    *,
    predictions_root: Path,
    source_multi_dataset: bool,
) -> tuple[dict, dict]:
    epoch_prediction_root = predictions_root / f"model-{candidate.epoch}"
    cfg.results_folder = str(epoch_prediction_root)
    trainer.load(pretrained_path=str(candidate.path))

    dataset_results = {}
    source_results = {}
    for dataset_key, source_keys in dataset_sources.items():
        for source_key in source_keys:
            if source_key in source_results:
                continue
            dataset_cfg = get_dataset_cfg(cfg, source_key)
            gt_root = Path(dataset_cfg.params.gt_root)
            pred_root = prediction_root_for_dataset(cfg, source_key, source_multi_dataset)

            inference_mae = run_inference_for_dataset(trainer, cfg, source_key, pred_root)
            if not trainer.accelerator.is_main_process:
                return {}, {}

            results = evaluate_prediction_folder(
                gt_root=gt_root,
                pred_root=pred_root,
                threshold=cfg.threshold,
                sweep_thresholds=False,
                threshold_steps=cfg.threshold_steps,
            )
            results["inference_MAE"] = float(inference_mae)
            results["dataset_key"] = source_key
            results["gt_root"] = str(gt_root)
            results["pred_root"] = str(pred_root)
            results["checkpoint"] = str(candidate.path)
            source_results[source_key] = results

        dataset_results[dataset_key] = weighted_average_results(source_results, source_keys)
        dataset_results[dataset_key]["dataset_key"] = dataset_key
        dataset_results[dataset_key]["checkpoint"] = str(candidate.path)

    if len(dataset_keys) == 1:
        aggregate_results = dict(dataset_results[dataset_keys[0]])
    else:
        aggregate_results = average_dataset_results(dataset_results, dataset_keys)
        aggregate_results["dataset_key"] = "Average"
    return aggregate_results, source_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=str, default="/data0/hl/DcDsDiff_runs/main")
    parser.add_argument(
        "--dataset-key",
        "--dataset-keys",
        dest="dataset_keys",
        nargs="+",
        default=["Mix"],
        help="Dataset keys used for selection. Defaults to Mix.",
    )
    parser.add_argument("--results_folder", type=str, default="/data0/hl/DcDsDiff_runs/main_f1_scan")
    parser.add_argument("--best-output", dest="best_output", type=str, default=None)
    parser.add_argument("--no-copy-best", dest="copy_best", action="store_false", default=True)
    parser.add_argument("--keep-all-predictions", dest="keep_all_predictions", action="store_true")
    parser.add_argument("--start-epoch", dest="start_epoch", type=int, default=None)
    parser.add_argument("--end-epoch", dest="end_epoch", type=int, default=None)
    parser.add_argument("--every-n-epochs", dest="every_n_epochs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--selection-metric", dest="selection_metric", default="F1", choices=("F1", "IoU", "AUC"))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulate_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_sample_steps", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-steps", dest="threshold_steps", type=int, default=256)
    parser.add_argument("--batch-ensemble", "--batch_ensemble", dest="batch_ensemble", action="store_true")
    parser.add_argument("--time-ensemble", "--time_ensemble", dest="time_ensemble", action="store_true", default=True)
    parser.add_argument("--no-time-ensemble", "--no_time_ensemble", dest="time_ensemble", action="store_false")
    parser.add_argument("--pred-root", dest="pred_root", type=str, default=None, help=argparse.SUPPRESS)

    cfg = add_args(parser)
    set_random_seed(7)

    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps
    if cfg.batch_ensemble and cfg.time_ensemble:
        raise SystemExit("Cannot use both --batch-ensemble and --time-ensemble")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    candidates = list_checkpoint_candidates(
        checkpoint_dir,
        start_epoch=cfg.start_epoch,
        end_epoch=cfg.end_epoch,
        every_n_epochs=cfg.every_n_epochs,
        limit=cfg.limit,
    )
    if cfg.dry_run:
        for candidate in candidates:
            print(f"epoch={candidate.epoch} checkpoint={candidate.path}")
        return

    dataset_keys = list(cfg.dataset_keys)
    candidate_sources = resolve_requested_sources(cfg, dataset_keys)
    unique_source_keys = sorted({source_key for source_keys in candidate_sources.values() for source_key in source_keys})
    source_multi_dataset = len(unique_source_keys) > 1
    dataset_sources = validate_dataset_roots(
        cfg,
        dataset_keys,
        skip_inference=False,
        multi_dataset=source_multi_dataset,
    )

    results_folder = Path(cfg.results_folder)
    predictions_root = results_folder / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(cfg)
    rows = []
    best_row = None
    best_prediction_root = None

    print(f"Scanning {len(candidates)} checkpoints on dataset keys: {' '.join(dataset_keys)}", flush=True)
    for index, candidate in enumerate(candidates, 1):
        aggregate_results, _ = evaluate_candidate(
            trainer,
            cfg,
            candidate,
            dataset_sources,
            dataset_keys,
            predictions_root=predictions_root,
            source_multi_dataset=source_multi_dataset,
        )
        row = row_from_results(candidate, aggregate_results, cfg.selection_metric)
        rows.append(row)

        epoch_prediction_root = predictions_root / f"model-{candidate.epoch}"
        if better_row(row, best_row):
            if best_prediction_root is not None and not cfg.keep_all_predictions:
                safe_rmtree(best_prediction_root, predictions_root)
            best_row = row
            best_prediction_root = epoch_prediction_root
        elif not cfg.keep_all_predictions:
            safe_rmtree(epoch_prediction_root, predictions_root)

        print(
            f"[{index}/{len(candidates)}] epoch={candidate.epoch} "
            f"F1={format_metric(row['F1'])} IoU={format_metric(row['IoU'])} "
            f"AUC={format_metric(row['AUC'])} best_epoch={best_row['epoch']}",
            flush=True,
        )

    if best_row is None:
        raise SystemExit("No checkpoint was evaluated.")

    best_output = None
    if cfg.copy_best:
        best_output = Path(cfg.best_output) if cfg.best_output else default_best_output_path(checkpoint_dir, best_row["epoch"])
        best_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_row["checkpoint"], best_output)

    write_scan_reports(results_folder, rows, best_row, best_output)

    print("")
    print(f"Best epoch by {cfg.selection_metric}: {best_row['epoch']}")
    print(f"Best checkpoint: {best_row['checkpoint']}")
    if best_output:
        print(f"Copied best checkpoint to: {best_output}")
    print(f"Best F1={format_metric(best_row['F1'])}, IoU={format_metric(best_row['IoU'])}, AUC={format_metric(best_row['AUC'])}")
    print(f"Saved scan reports under: {results_folder}")


if __name__ == "__main__":
    main()
