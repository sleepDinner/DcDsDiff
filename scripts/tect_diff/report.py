"""Generate and publish small TECT reports from machine-readable run receipts.

Publication uses an isolated worktree at the latest authorized remote branch.
The active source snapshot is never changed by Git operations.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts.tect_diff.common import (atomic_json, git, inside, read_json,
                                      runtime_environment, timestamp)
from tools.resource_locks import acquire_file

SELECTION_NOTICE = ("按八测试集逐 epoch F1 选择 checkpoint；这些测试集参与模型选择，"
                    "不是独立泛化评估。selection_protocol=test_selected。"
                    "固定终点 final.pth 与 test-selected best.pth 分开报告。")
LEDGER_FIELDS = ("run_id", "group_id", "parent_run", "status", "stage", "commit", "config_hash",
                 "resolution", "seed", "reference_source_mode", "reference_hash", "calibration_hash",
                 "best_epoch", "best_all8_macro_pixel_f1", "final_epoch", "final_all8_macro_pixel_f1",
                 "elapsed_seconds", "report_path", "failure_reason")


def optional_json(path):
    return read_json(path) if Path(path).is_file() else {}


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _number(value):
    return "NA" if value is None or value == "" else f"{value:.8g}" if isinstance(value, float) else str(value)


def _score(record):
    if not record:
        return None
    metrics = record.get("metrics", {})
    return metrics.get("all8_macro_pixel_f1", metrics.get("summary", {}).get("all8_macro_pixel_f1"))


def _checkpoint_table(label, record):
    if not record:
        return f"### {label}\n\n尚无此终点的完整 checkpoint / 八集结果。\n"
    metrics = record.get("metrics", {})
    summary = metrics.get("summary", metrics.get("all8", metrics))
    datasets = metrics.get("datasets", metrics.get("per_dataset", {}))
    lines = [f"### {label}", "", f"Epoch: {record.get('epoch', 'NA')}。",
             f"All8 macro Pixel-F1: {_number(summary.get('all8_macro_pixel_f1'))}。", "",
             "| Dataset | Images | Pixel-F1 | IoU | Boundary-F1 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, values in datasets.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"| {name} | {_number(values.get('count'))} | "
                     f"{_number(values.get('dataset_pixel_f1'))} | {_number(values.get('dataset_iou'))} | "
                     f"{_number(values.get('dataset_boundary_f1'))} |")
    lines += ["", f"Server checkpoint: `{record.get('path', 'NA')}`",
              f"SHA256: `{record.get('sha256', 'NA')}`。权重仅保存在服务器。", ""]
    return "\n".join(lines)


def summarize_diagnostics(run):
    """Summarize observed low-frequency records, not unlogged training batches."""
    files = [run / "training_metrics.jsonl"]
    files += sorted(run.glob("training_metrics.rank*.jsonl"))
    streams = {}
    for path in files:
        if not path.is_file():
            continue
        metrics, records, invalid_lines = {}, 0, 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue
                records += 1
                for key, value in row.items():
                    if not isinstance(value, (int, float)):
                        continue
                    entry = metrics.setdefault(key, {"count": 0, "nonfinite_count": 0, "mean": 0.0,
                                                     "minimum": None, "maximum": None, "last": None,
                                                     "kind": "boolean_frequency" if isinstance(value, bool) else "scalar"})
                    value = int(value) if isinstance(value, bool) else value
                    if not math.isfinite(value):
                        entry["nonfinite_count"] += 1
                        continue
                    entry["count"] += 1
                    entry["mean"] = (entry["mean"] * (1 - 1 / entry["count"]) + value / entry["count"])
                    entry["minimum"] = value if entry["minimum"] is None else min(entry["minimum"], value)
                    entry["maximum"] = value if entry["maximum"] is None else max(entry["maximum"], value)
                    entry["last"] = value
        for entry in metrics.values():
            if not entry["count"] or not math.isfinite(entry["mean"]):
                entry["mean"] = None
        streams[path.name] = {"records": records, "malformed_lines": invalid_lines, "metrics": metrics}
    preflight = optional_json(run / "preflight_receipt.json")
    main_probe = optional_json(run / "main_probe_receipt.json")
    ranks = {str(rank): optional_json(run / f"progress_rank{rank}.json") for rank in (0, 1)}
    peak_by_rank = {}
    for rank, progress in ranks.items():
        candidates = []
        peak = progress.get("peak_memory_bytes")
        if isinstance(peak, (int, float)) and math.isfinite(peak):
            candidates.append(peak)
        entries = streams.get(f"training_metrics.rank{rank}.jsonl", {}).get("metrics", {})
        maximum = entries.get("peak_memory_bytes", {}).get("maximum")
        if maximum is not None:
            candidates.append(maximum)
        peak_by_rank[rank] = max(candidates) if candidates else None
    statistics = streams.get("training_metrics.jsonl", {}).get("metrics", {})
    def all_observed(key, predicate):
        entry = statistics.get(key)
        if not entry or not entry["count"]:
            return None
        return predicate(entry)
    result = {
        "scope": "Low-frequency sampled training records; not an estimate over all unlogged batches.",
        "streams": streams, "peak_allocated_bytes_by_rank": peak_by_rank,
        "preflight": preflight, "main_probe": main_probe,
        "sample_profiles": main_probe.get("sample_profile", main_probe.get("sample_profiles", {})),
        "flags": {
            "all_observed_q_zero": all_observed("q_zero_fraction", lambda item: item["minimum"] == 1),
            "all_observed_gamma_below_1e_minus4": all_observed("gamma", lambda item: item["maximum"] < 1e-4),
            "all_observed_image_loss_below_1e_minus12": all_observed("image_loss", lambda item: item["maximum"] < 1e-12),
            "all_observed_joint_ref_mse_below_1e_minus12": all_observed("joint_ref_mse", lambda item: item["maximum"] < 1e-12),
        },
        "control_effect_boundary": "A nonzero logit change/finite gradient is not proof of benefit. Compare registered training-probe base/control diagnostics where available; no checkpoint/output reselection.",
        "cache_scope": "Per-image within one ten-step call; no claimed cross-call or cross-epoch cache.",
    }
    atomic_json(run / "diagnostics_summary.json", result)
    return result


def exclusion_summary(root, bundle):
    result = {"excluded_train_count": bundle.get("excluded_train_count"),
              "rule": bundle.get("leakage_rule", "Preparation not complete")}
    if bundle.get("summary_path"):
        path = inside(root, Path(bundle["summary_path"]).parent / "exclusions.json")
        if path.is_file():
            exclusions = read_json(path)
            result.update(exact_rgb_overlap_count=sum(bool(item.get("exact_rgb_test_ids")) for item in exclusions),
                          known_source_overlap_count=sum(bool(item.get("known_source_test_ids")) for item in exclusions),
                          records=len(exclusions), counts_may_overlap=True)
    return result


def _merge_ledger(target, row):
    ledger_path = target / "analysis_reports/experiment_ledger.csv"
    existing = []
    if ledger_path.is_file():
        with ledger_path.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
    records = {entry["run_id"]: entry for entry in existing}
    records[row["run_id"]] = row
    records = [records[key] for key in sorted(records)]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    atomic_text(ledger_path, buffer.getvalue())
    lines = ["# TECT-Diff experiment ledger", "", SELECTION_NOTICE, "",
             "| Run | Status | Stage | Resolution | Best epoch / All8 F1 | Final epoch / All8 F1 |",
             "| --- | --- | --- | ---: | --- | --- |"]
    for entry in records:
        lines.append(f"| [{entry['run_id']}](runs/{entry['run_id']}/report.md) | {entry['status']} | "
                     f"{entry.get('stage', '')} | {entry.get('resolution', '')} | "
                     f"{entry.get('best_epoch', '')} / {entry.get('best_all8_macro_pixel_f1', '')} | "
                     f"{entry.get('final_epoch', '')} / {entry.get('final_all8_macro_pixel_f1', '')} |")
    atomic_text(target / "analysis_reports/experiment_ledger.md", "\n".join(lines) + "\n")
    return records


def generate(run_dir, target_root=None):
    run = Path(run_dir).resolve()
    config = read_json(run / "resolved_config.json")
    root = Path(config["project_root"]).resolve()
    inside(root, run)
    marker = read_json(run / ".tect-created.json")
    if marker.get("run_id") != run.name or marker.get("created_by") != "tect-diff":
        raise ValueError("Report run ownership mismatch")
    target = root if target_root is None else inside(root, target_root)
    state = optional_json(run / "controller_status.json")
    progress = optional_json(run / "progress.json")
    provenance = read_json(run / "provenance.json")
    bundle = optional_json(run / "data_bundle.json")
    reference = optional_json(run / "reference_receipt.json")
    calibration = optional_json(run / "calibration_receipt.json")
    main = optional_json(run / "main_receipt.json")
    diagnostics = summarize_diagnostics(run)
    exclusions = exclusion_summary(root, bundle)
    # Checkpoint choice is recorded by the trainer, never reselected here.
    selection = optional_json(run / "selection.json") or optional_json(run / "epoch_summary.json")
    best = main.get("best", selection.get("best", {})) or {}
    final = main.get("final", {}) or {}
    report = {
        "run_id": run.name, "protocol_id": config["protocol_id"], "group_id": config["group_id"],
        "status": state.get("status", "REGISTERED"), "stage": state.get("stage", "REGISTERED"),
        "commit": provenance["commit"], "config_hash": provenance["config_hash"],
        "resolution": config["resolution"], "seed": config["seed"], "gpus": config["gpus"],
        "environment": config["environment"], "selection_protocol": "test_selected",
        "selection_notice": SELECTION_NOTICE, "reference_source_mode": bundle.get("reference_source_mode"),
        "manifest_hashes": bundle.get("manifest_hashes", {}),
        "data_counts": {key: bundle.get(key) for key in ("train_count", "reference_count", "calibration_count", "test_counts", "excluded_train_count")},
        "reference": reference, "calibration": calibration, "best": best, "final": final,
        "costs": {**state.get("stage_costs", {}), **main.get("costs", {})}, "progress": progress,
        "pretrained_load_report": main.get("pretrained_load_report", {}),
        "stage_budgets": {"reference_epochs": config["reference"]["epochs"],
                          "calibration_max_images": config["data"]["calibration_max_images"],
                          "main_epochs": config["training"]["epochs"]},
        "diagnostics": diagnostics, "train_exclusions": exclusions,
        "failure_reason": state.get("failure_reason"), "updated_at": timestamp(),
        "baseline_comparison": "No complete baseline under this protocol; no superiority claim.",
        "checkpoints_local": "Not downloaded; server-authoritative weights only.",
        "source_snapshot": str(run / "source"),
    }
    row = dict(run_id=run.name, group_id=config["group_id"], parent_run=config.get("parent_run", ""),
               status=report["status"], stage=report["stage"], commit=report["commit"],
               config_hash=report["config_hash"], resolution=config["resolution"], seed=config["seed"],
               reference_source_mode=report["reference_source_mode"], reference_hash=reference.get("sha256", ""),
               calibration_hash=calibration.get("sha256", ""), best_epoch=best.get("epoch", ""),
               best_all8_macro_pixel_f1=_score(best), final_epoch=final.get("epoch", ""),
               final_all8_macro_pixel_f1=_score(final), elapsed_seconds=state.get("elapsed_seconds", ""),
               report_path=f"analysis_reports/runs/{run.name}/report.md", failure_reason=state.get("failure_reason", ""))
    status_text = ("训练及固定终点评估已结束；报告发布结果见服务器 publish_receipt.json。"
                   if report["status"] == "COMPLETED" else "尚未完成正式主训练与固定终点汇总。")
    lines = [f"# {run.name}", "", f"状态：**{report['status']}**；阶段：{report['stage']}。{status_text}", "",
             SELECTION_NOTICE, "", "## Provenance", "", f"- Source commit: `{report['commit']}`",
             f"- Config SHA256: `{report['config_hash']}`", f"- Environment: `{config['environment']}`",
             f"- GPUs: {config['gpus']}; input {config['resolution']} × {config['resolution']}; seed {config['seed']}.",
             f"- Reference source mode: `{report['reference_source_mode']}`.",
             "- 全部有效训练图参加定位训练；参考拟合和校准仅使用训练内角色，不划分选模验证集。",
             "- 观测锚定的 Image 加噪—去噪与 Mask 反向生成；不声称恢复篡改者的实际生成轨迹。", "",
             "## Data and training-only fitting", "", "```json",
             json.dumps({"counts": report["data_counts"], "exclusions": exclusions,
                         "manifest_hashes": report["manifest_hashes"],
                         "stage_budgets": report["stage_budgets"]}, indent=2, ensure_ascii=False), "```", "",
             "## Actual progress", "", "```json", json.dumps(progress, indent=2, ensure_ascii=False), "```", "",
             _checkpoint_table("Test-selected best", best), _checkpoint_table("Fixed final endpoint", final),
             "## Phase receipts and costs", "", "```json",
             json.dumps({"reference": reference, "calibration": calibration, "costs": report["costs"]},
                        indent=2, ensure_ascii=False), "```", "", "## Mechanisms and limitations", "",
             "参考参数与训练内统计冻结；Image loss 必须非零启用。梯度/异常存在仅说明工程路径可运行，不代表方法有效。",
             "q 全零、gamma 退化、Image 退化或控制损害定位以 diagnostics_summary.json 记录的实际诊断为准；缺失项仍待完成。",
             "以下为固定低频诊断样本的统计，不能冒充所有训练像素的总体统计。", "", "```json",
             json.dumps({"flags": diagnostics["flags"], "peak_allocated_bytes_by_rank": diagnostics["peak_allocated_bytes_by_rank"],
                         "sample_profiles": diagnostics["sample_profiles"],
                         "sampled_training": diagnostics["streams"].get("training_metrics.jsonl", {})},
                        indent=2, ensure_ascii=False), "```", "",
             bundle.get("leakage_limit", "数据泄漏审计尚未完成。"),
             "无同协议完整 baseline，只报告 TECT-Diff 自身结果，不声称超过 DcDsDiff。", "",
             "## Reproduce / recover", "", "```bash",
             f"{config['environment']}/bin/python -s {run}/source/scripts/tect_diff/controller.py status --run-dir {run}",
             f"{config['environment']}/bin/python -s {run}/source/scripts/tect_diff/controller.py resume --run-dir {run}",
             "```", "", f"Source snapshot: `{run / 'source'}`; logs/status/checkpoints: `{run}`.",
             "模型权重、训练原图和诊断可视化不上传 GitHub。", ""]
    if report["failure_reason"]:
        lines += ["## Failure", "", str(report["failure_reason"]), ""]
    with acquire_file(root / "runtime/locks/tect-reports.lock"):
        destination = target / "analysis_reports/runs" / run.name
        atomic_json(destination / "report.json", report)
        atomic_text(destination / "report.md", "\n".join(lines))
        records = _merge_ledger(target, row)
        group_records = [entry for entry in records if entry.get("group_id") == config["group_id"]]
        group_lines = [f"# {config['group_id']}", "", SELECTION_NOTICE, "",
                       "各 run 的数值只读取机器可读结果。未完成 run 不提供最终成绩。", ""]
        group_lines += [f"- [{entry['run_id']}](../../runs/{entry['run_id']}/report.md): {entry['status']}"
                        for entry in group_records]
        atomic_text(target / "analysis_reports/groups" / config["group_id"] / "report.md", "\n".join(group_lines) + "\n")
        root_ledger = target / "experiment_ledger.md"
        ledger_text = root_ledger.read_text(encoding="utf-8") if root_ledger.exists() else "# Experiment ledger\n"
        link = f"[TECT-Diff {run.name}](analysis_reports/runs/{run.name}/report.md)"
        if link not in ledger_text:
            atomic_text(root_ledger, ledger_text.rstrip() + "\n\n- " + link + "; test_selected All8 F1; fixed final reported separately.\n")
        if report["failure_reason"]:
            failure_path = target / "analysis_reports/failure_registry.jsonl"
            failures = [json.loads(line) for line in failure_path.read_text().splitlines() if line] if failure_path.exists() else []
            event = {"run_id": run.name, "commit": report["commit"], "stage": report["stage"],
                     "reason": report["failure_reason"], "at": state.get("updated_at")}
            key = (event["run_id"], event["commit"], event["reason"])
            if key not in {(item["run_id"], item["commit"], item["reason"]) for item in failures}:
                failures.append(event)
            atomic_text(failure_path, "".join(json.dumps(item, sort_keys=True) + "\n" for item in failures))
    return report


def publish(run_dir):
    run = Path(run_dir).resolve()
    config = read_json(run / "resolved_config.json")
    root = Path(config["project_root"]).resolve()
    settings = config["runtime"]
    remote, branch = settings["publish_remote"], settings["publish_branch"]
    # Exact authorization binding; never fall back to upstream/another project.
    actual_url = git(root, "remote", "get-url", "--push", remote)
    if actual_url != settings["publish_url"] or "sleepDinner/DcDsDiff" not in actual_url:
        raise RuntimeError("Publication remote differs from authorized project repository")
    env = runtime_environment(root, run.name)
    env.update(GIT_TERMINAL_PROMPT="0")
    directory = inside(root, root / "runtime/tect-publish" / run.name)
    directory.parent.mkdir(parents=True, exist_ok=True)
    def command(*args, cwd=root):
        return subprocess.check_output(["git", "-C", str(cwd), *args], text=True, env=env,
                                       stderr=subprocess.STDOUT).strip()
    with acquire_file(root / "runtime/locks/tect-publish.lock"):
        command("fetch", remote, branch)
        remote_ref = f"refs/remotes/{remote}/{branch}"
        remote_commit = command("rev-parse", remote_ref)
        if directory.exists():
            expected = optional_json(directory.parent / f"{run.name}.ownership.json")
            if expected.get("run_id") != run.name or expected.get("path") != str(directory):
                raise RuntimeError("Publication worktree ownership mismatch")
            if command("status", "--porcelain", cwd=directory):
                # Keep previous attempt safely. A fresh detached worktree is
                # based on the latest branch; no reset/clean destroys content.
                directory = inside(root, directory.with_name(directory.name + f"-{time.time_ns()}"))
            elif command("rev-parse", "HEAD", cwd=directory) != remote_commit:
                command("checkout", "--detach", remote_commit, cwd=directory)
        if not directory.exists():
            command("worktree", "add", "--detach", str(directory), remote_commit)
            atomic_json(directory.parent / f"{directory.name}.ownership.json",
                        {"run_id": run.name, "path": str(directory), "created_by": "tect-diff"})
        generate(run, target_root=directory)
        allowed = [f"analysis_reports/runs/{run.name}/report.md", f"analysis_reports/runs/{run.name}/report.json",
                   f"analysis_reports/groups/{config['group_id']}/report.md",
                   "analysis_reports/experiment_ledger.csv", "analysis_reports/experiment_ledger.md", "experiment_ledger.md"]
        if (directory / "analysis_reports/failure_registry.jsonl").exists():
            allowed.append("analysis_reports/failure_registry.jsonl")
        command("add", "--", *allowed, cwd=directory)
        staged = command("diff", "--cached", "--name-only", cwd=directory).splitlines()
        if set(staged) - set(allowed):
            raise RuntimeError("Non-whitelisted publication file")
        for relative in staged:
            if inside(directory, directory / relative).stat().st_size > 4 * 1024 * 1024:
                raise RuntimeError("Report exceeds small-artifact publication limit")
        if staged:
            command("commit", "-m", f"docs: summarize TECT-Diff {run.name}", cwd=directory)
        commit = command("rev-parse", "HEAD", cwd=directory)
        command("push", remote, f"HEAD:refs/heads/{branch}", cwd=directory)
        verified = command("ls-remote", remote, f"refs/heads/{branch}").split()[0]
        if verified != commit:
            raise RuntimeError("Publication push could not be verified")
        receipt = {"status": "SYNCED", "run_id": run.name, "commit": commit,
                   "branch": branch, "repository": actual_url, "at": timestamp()}
        atomic_json(run / "publish_receipt.json", receipt)
        atomic_json(run / "pending_sync.json", {"pending_sync": False, **receipt})
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = publish(args.run_dir) if args.publish else generate(args.run_dir)
    print(json.dumps({key: result.get(key) for key in ("run_id", "status", "commit")}, sort_keys=True))


if __name__ == "__main__":
    main()
