"""Small two-test-set pilot reports and isolated, whitelisted publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts.tect_diff.common import atomic_json, inside, read_json, runtime_environment, timestamp
from scripts.tect_diff.controller import read_run
from scripts.tect_diff.pilot_controller import effective_outcome
from scripts.tect_diff.report import atomic_text, optional_json
from tools.resource_locks import acquire_file

NOTICE = ("selection_protocol=test_selected：Casiav1/Columbia 用于开发、checkpoint 选择和推进判断，"
          "不是独立泛化评估。READY_FOR_FULL 只表示小规模验证通过，完整训练尚未执行。")


def _reject_nonfinite(value):
    raise ValueError(f"Nonfinite pilot epoch value: {value}")


def read_epochs(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line, parse_constant=_reject_nonfinite)
        if (row.get("epoch_complete") is not True or row.get("evaluation_complete") is not True
                or set(row.get("pixel_f1", {})) != {"Casiav1", "Columbia"}):
            raise ValueError("Pilot epoch display contains an incomplete or wrong-population result")
        rows.append(row)
    indices = [row["epoch_index"] for row in rows]
    if indices != sorted(set(indices)):
        raise ValueError("Pilot epoch display has duplicate or unordered epochs")
    return rows


def publication_paths(run_id):
    prefix = f"analysis_reports/runs/{run_id}"
    return [f"{prefix}/report.md", f"{prefix}/report.json", f"{prefix}/metrics_per_epoch.jsonl",
            "analysis_reports/pilot_ledger.json", "analysis_reports/pilot_ledger.md", "experiment_ledger.md"]


def generate(run_dir, target_root=None):
    run, root, config, provenance = read_run(run_dir)
    if not config.get("pilot"):
        raise ValueError("Pilot reports require a registered pilot configuration")
    target = root if target_root is None else inside(root, target_root)
    state = optional_json(run / "controller_status.json")
    receipt = optional_json(run / "pilot_receipt.json")
    confirmation = optional_json(run / "confirmation_result.json")
    bundle = optional_json(run / "data_bundle.json")
    epochs = read_epochs(run / "metrics_per_epoch.jsonl")
    report = {
        "run_id": run.name, "group_id": config["group_id"], "protocol_id": config["protocol_id"],
        "status": state.get("status", "REGISTERED"), "stage": state.get("stage", "PREPARING"),
        "outcome": effective_outcome(state, receipt),
        "commit": provenance["commit"], "config_hash": provenance["config_hash"],
        "source_snapshot": str(run / "source"), "selection_protocol": "test_selected",
        "selection_notice": NOTICE, "resolution": config["resolution"], "seed": config["seed"],
        "model": config["model"], "pilot": config["pilot"],
        "data_counts": {key: bundle.get(key) for key in ("train_count", "test_counts", "test_full_counts", "authentic_probe_count")},
        "manifest_hashes": bundle.get("manifest_hashes", {}),
        "fitting_reuse": optional_json(run / "fitting_reuse.json"),
        "pilot_receipt": receipt, "confirmation_result": confirmation,
        "last_complete_epoch": epochs[-1] if epochs else None,
        "complete_epoch_count": len(epochs), "elapsed_seconds": state.get("elapsed_seconds"),
        "failure_reason": state.get("failure_reason"), "progress": optional_json(run / "progress.json"),
        "updated_at": timestamp(), "full_training_started": False,
    }
    lines = [f"# {run.name}", "", f"状态：**{report['status']}**；结果：**{report['outcome']}**。", "", NOTICE, "",
             "## 数据与训练边界", "", "训练：固定 CASIA2 子集；逐轮测试：固定 Casiav1/Columbia 子集。",
             "连续健康门槛通过后，才对同一 checkpoint 执行一次完整两集确认；不在此任务启动完整训练。",
             f"源码：`{provenance['commit']}`；配置：`{provenance['config_hash']}`。", "",
             "## 逐轮 Pixel-F1", "", "| Epoch | Casiav1 | Columbia | Average Test2 |", "| ---: | ---: | ---: | ---: |"]
    for row in epochs:
        lines.append(f"| {row['epoch_number']} | {row['pixel_f1']['Casiav1']:.6f} | "
                     f"{row['pixel_f1']['Columbia']:.6f} | {row['average_test2']:.6f} |")
    if not epochs:
        lines += ["", "尚无完成的训练及测试轮次。"]
    if confirmation:
        metrics = confirmation["metrics"]
        lines += ["", "## 完整两集确认", "",
                  f"同一 epoch {confirmation['epoch']} checkpoint；确认门槛通过：{confirmation['gate']['passed']}。", "",
                  "| Dataset | Images | Pixel-F1 |", "| --- | ---: | ---: |"]
        for name in ("Casiav1", "Columbia"):
            lines.append(f"| {name} | {metrics['image_counts'][name]} | {metrics['pixel_f1'][name]:.6f} |")
        lines += ["", f"Average Test2：{metrics['average_test2']:.6f}。这些结果继续属于开发反馈。"]
    lines += ["", "详细记录：[metrics_per_epoch.jsonl](metrics_per_epoch.jsonl)。", "",
              f"完整结果与确认收据：[report.json](report.json)。服务器 run：`{run}`。",
              "权重留在服务器；已有 BN 运行与本 GN 架构不能作为相同协议的数值对照。", ""]
    if report["failure_reason"]:
        lines += ["## 停止原因", "", str(report["failure_reason"]), ""]
    with acquire_file(root / "runtime/locks/tect-reports.lock"):
        destination = target / "analysis_reports/runs" / run.name
        atomic_json(destination / "report.json", report)
        atomic_text(destination / "report.md", "\n".join(lines))
        atomic_text(destination / "metrics_per_epoch.jsonl", "".join(json.dumps(row, allow_nan=False) + "\n" for row in epochs))
        ledger_file = target / "analysis_reports/pilot_ledger.json"
        ledger = optional_json(ledger_file)
        ledger[run.name] = {key: report[key] for key in ("run_id", "group_id", "status", "stage", "outcome", "commit", "config_hash", "complete_epoch_count", "failure_reason")}
        atomic_json(ledger_file, ledger)
        ledger_lines = ["# TECT small-data pilots", "", NOTICE, "", "| Run | Status | Outcome | Complete epochs |", "| --- | --- | --- | ---: |"]
        for name, row in sorted(ledger.items()):
            ledger_lines.append(f"| [{name}](runs/{name}/report.md) | {row['status']} | {row['outcome']} | {row['complete_epoch_count']} |")
        atomic_text(target / "analysis_reports/pilot_ledger.md", "\n".join(ledger_lines) + "\n")
        root_ledger = target / "experiment_ledger.md"
        text = root_ledger.read_text(encoding="utf-8") if root_ledger.exists() else "# Experiment ledger\n"
        begin, end = f"<!-- pilot:{run.name}:begin -->", f"<!-- pilot:{run.name}:end -->"
        entry = f"{begin}\n- [TECT pilot {run.name}](analysis_reports/runs/{run.name}/report.md): {report['status']} / {report['outcome']}; selection_protocol=test_selected; complete epochs={len(epochs)}; full training not started.\n{end}"
        if begin in text:
            if text.count(begin) != 1 or text.count(end) != 1 or text.index(end) < text.index(begin):
                raise ValueError("Pilot root-ledger marker mismatch")
            start, stop = text.index(begin), text.index(end) + len(end)
            text = text[:start] + entry + text[stop:]
        else:
            text = text.rstrip() + "\n\n" + entry + "\n"
        atomic_text(root_ledger, text)
    return report


def publish(run_dir):
    run, root, config, _ = read_run(run_dir)
    settings = config["runtime"]
    remote, branch = settings["publish_remote"], settings["publish_branch"]
    if settings["publish_url"] != "git@github.com:sleepDinner/DcDsDiff.git" or branch != "feature/tect-diff":
        raise ValueError("Pilot publication repository/branch is not registered")
    env = runtime_environment(root, run.name)
    env["GIT_TERMINAL_PROMPT"] = "0"

    def git(*args, cwd=root):
        return subprocess.check_output(["git", "-C", str(cwd), *args], text=True, env=env,
                                       stderr=subprocess.STDOUT, timeout=180).strip()

    if git("remote", "get-url", "--push", remote) != settings["publish_url"]:
        raise ValueError("Pilot publication remote changed")
    with acquire_file(root / "runtime/locks/tect-publish.lock"):
        git("fetch", remote, branch)
        commit = git("rev-parse", f"refs/remotes/{remote}/{branch}")
        # One fresh owned worktree per attempt preserves failed publications.
        directory = inside(root, root / "runtime/tect-pilot-publish" / f"{run.name}-{time.time_ns()}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(directory), commit)
        atomic_json(directory.parent / f"{directory.name}.ownership.json",
                    {"run_id": run.name, "path": str(directory), "created_by": "tect-diff-pilot"})
        generate(run, target_root=directory)
        allowed = publication_paths(run.name)
        git("add", "--", *allowed, cwd=directory)
        staged = git("diff", "--cached", "--name-only", cwd=directory).splitlines()
        if set(staged) - set(allowed):
            raise RuntimeError("Non-whitelisted pilot publication file")
        if any(inside(directory, directory / item).stat().st_size > 4 * 1024 * 1024 for item in staged):
            raise RuntimeError("Pilot report exceeds small-artifact publication limit")
        if staged:
            git("commit", "-m", f"docs: summarize TECT pilot {run.name}", cwd=directory)
        commit = git("rev-parse", "HEAD", cwd=directory)
        git("push", remote, f"HEAD:refs/heads/{branch}", cwd=directory)
        if git("ls-remote", remote, f"refs/heads/{branch}").split()[0] != commit:
            raise RuntimeError("Pilot publication push could not be verified")
        result = {"status": "SYNCED", "run_id": run.name, "commit": commit, "branch": branch,
                  "repository": settings["publish_url"], "at": timestamp()}
        atomic_json(run / "publish_receipt.json", result)
        atomic_json(run / "pending_sync.json", {"pending_sync": False, **result})
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--publish", action="store_true")
    arguments = parser.parse_args()
    result = publish(arguments.run_dir) if arguments.publish else generate(arguments.run_dir)
    print(json.dumps({key: result.get(key) for key in ("run_id", "status", "outcome", "commit")}))
