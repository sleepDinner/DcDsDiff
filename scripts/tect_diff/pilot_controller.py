"""Bounded, detached GN8/CASIA2 pilot with both GPU locks and fixed source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback

SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts.tect_diff.common import atomic_json, git, inside, json_hash, read_json, runtime_environment, sha256, timestamp
from scripts.tect_diff.controller import (
    acquire_all, process_identity, same_process, read_run, source_snapshot,
    verify_source, status as base_status, owned_group_members, stop_owned_members, validate_receipt,
)
from tools.resource_locks import ResourceBusy, acquire_file

RUN_ID = "TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A"
GROUP_ID = "TECT-PILOT-CASIA2-GN8"
FITTING_PARENT = "TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E"
REFERENCE_HASH = "800f50c392a275fde3ec249c6f52275c6db79813e6f8b7e8b534f9ddb52a523f"
CALIBRATION_HASH = "bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c"


def validate_config(config, run_id):
    if run_id != RUN_ID or config.get("group_id") != GROUP_ID:
        raise ValueError("Only the registered new CASIA2/GN8 pilot is accepted")
    if config.get("project_root") != "/data1/hl/DcDsDiff-and-GIT10K" or config.get("environment") != "/data0/hl/conda_envs/dcdsdiff":
        raise ValueError("Pilot project/environment differs from the authorized server")
    if config.get("world_size") != 2 or config.get("gpus") != [0, 1] or config.get("seed") != 42 or config.get("resolution") != 512:
        raise ValueError("Pilot requires registered dual-GPU 512/seed42 settings")
    train = config["training"]
    if (train.get("micro_batch"), train.get("accumulation_steps"), train.get("global_batch")) != (6, 1, 12):
        raise ValueError("Pilot batch must remain micro6/accumulation1/global12")
    if type(train.get("epochs")) is not int or not 1 <= train["epochs"] <= 10:
        raise ValueError("Pilot training is bounded to at most ten epochs")
    model = config["model"]
    if model.get("normalization") != "groupnorm8" or model.get("architecture_version") != "tect-diff-full-v2-main-gn8":
        raise ValueError("Pilot requires the registered GN8 task architecture")
    pilot = config["pilot"]
    if Path(pilot.get("fitting_parent", "")).name != FITTING_PARENT:
        raise ValueError("Pilot must reuse E's completed frozen fitting dependencies")
    deadline = pilot.get("total_deadline_seconds", 14400)
    if type(deadline) not in (int, float) or not 0 < deadline <= 14400:
        raise ValueError("Pilot deadline must be positive and no more than four hours")
    for key, expected in {"train_authentic": 1024, "train_tampered": 1024,
                          "quick_test_per_dataset": 128, "authentic_probe": 64,
                          "preflight_train": 64, "engineering_updates": 8}.items():
        if type(pilot.get(key)) is not int or pilot[key] != expected:
            raise ValueError("Pilot registered sample/probe budget differs: " + key)
    if pilot.get("full_training_automatic_launch") is not False or pilot.get("full_confirmation_before_expansion") is not True:
        raise ValueError("Pilot must require confirmation and stop before full training")
    if config["runtime"].get("heartbeat_seconds") != 60 or config["runtime"].get("automatic_training_retries") != 0:
        raise ValueError("Pilot requires 60-second status and zero automatic training retries")
    settings = config["runtime"]
    if settings.get("publish_branch") != "feature/tect-diff" or settings.get("publish_url") != "git@github.com:sleepDinner/DcDsDiff.git":
        raise ValueError("Pilot publication destination is not registered")


def _read_pilot(run_dir):
    values = read_run(run_dir)
    validate_config(values[2], values[0].name)
    return values


def status(run_dir):
    run, _, _, _ = _read_pilot(run_dir)
    result = base_status(run)
    if (run / "pilot_receipt.json").exists():
        receipt = read_json(run / "pilot_receipt.json")
        result["pilot_outcome"] = effective_outcome(result, receipt)
    return result


def effective_outcome(state, receipt):
    """Only a completed controller has accepted the worker's outcome receipt."""
    if state.get("status") == "COMPLETED":
        return state.get("outcome", receipt.get("outcome", "PENDING"))
    if state.get("status") in {"FAILED", "INTERRUPTED"}:
        return "HOLD"
    return "PENDING"


def import_fitting(run):
    """Rebind small receipts only; preserve original artifacts and fitting hashes."""
    run, root, config, provenance = _read_pilot(run)
    parent_spec = Path(config["pilot"]["fitting_parent"])
    parent_path = parent_spec if parent_spec.is_absolute() else root / "runs" / parent_spec
    parent, parent_root, old_config, old_provenance = read_run(parent_path)
    if parent.name != FITTING_PARENT or parent_root != root or parent == run:
        raise ValueError("Fitting source must be the held same-project E run")
    with acquire_file(root / "runtime/locks" / f"tect-registration-{parent.name}.lock"):
        parent_state = base_status(parent)
        if parent_state["controller_alive"] or parent_state["worker_alive"] or not read_json(parent / "operational_hold.json").get("active"):
            raise ValueError("E must remain held with no live controller or workers")
        verify_source(parent, old_provenance)
        verify_source(run, provenance)
        for section in ("reference", "evidence", "image_diffusion", "sampling"):
            if config[section] != old_config[section]:
                raise ValueError(f"Frozen fitting mechanism changed: {section}")
        for relative in ("model/tect_diff/reference.py", "model/tect_diff/evidence.py", "model/tect_diff/diffusion.py"):
            if old_provenance["source_hashes"].get(relative) != provenance["source_hashes"].get(relative):
                raise ValueError("Fitting reuse cannot change the protected source: " + relative)
        source_bundle = read_json(parent / "data_bundle.json")
        expected = {"reference": REFERENCE_HASH, "calibration": CALIBRATION_HASH}
        originals = {}
        for stage, digest in expected.items():
            if not validate_receipt(parent, stage, old_provenance):
                raise ValueError("Missing completed fitting dependency: " + stage)
            original = read_json(parent / f"{stage}_receipt.json")
            original_path = Path(original["artifact_path"])
            path = inside(root, original_path)
            if (original_path.is_symlink() or original_path.absolute() != path or not path.is_file()
                    or original["sha256"] != digest or sha256(path) != digest):
                raise ValueError("Registered fitting artifact bytes differ: " + stage)
            originals[stage] = original
        import torch
        artifact = torch.load(originals["calibration"]["artifact_path"], map_location="cpu", weights_only=True)
        from model.tect_diff.evidence import FixedTrajectoryEvidence
        FixedTrajectoryEvidence(artifact).assert_frozen()
        if (artifact["metadata"]["reference_sha256"] != REFERENCE_HASH
                or artifact["metadata"]["training_manifest_sha256"] != source_bundle["manifest_hashes"]["train"]
                or artifact["metadata"]["fit_manifest_sha256"] != source_bundle["manifest_hashes"]["calibration"]
                or artifact["artifact_hash"] != originals["calibration"]["artifact_hash"]
                or artifact["fit_receipt"] != originals["calibration"]["fit_receipt"]):
            raise ValueError("Calibration metadata no longer binds original training/reference")
        receipt_hashes = {stage: sha256(parent / f"{stage}_receipt.json") for stage in originals}
        for stage, original in originals.items():
            source_receipt = run / "fitting_source" / f"{stage}_receipt.json"
            rebound = {**original, "protocol_id": config["protocol_id"], "config_hash": provenance["config_hash"],
                       "fitting_origin_run": parent.name, "fitting_origin_config_hash": old_provenance["config_hash"],
                       "fitting_origin_receipt_sha256": receipt_hashes[stage]}
            target = run / f"{stage}_receipt.json"
            if source_receipt.exists() and sha256(source_receipt) != receipt_hashes[stage]:
                raise ValueError("Existing fitting-origin receipt differs")
            if target.exists() and read_json(target) != rebound:
                raise ValueError("Existing rebound fitting receipt differs")
            if not source_receipt.exists():
                source_receipt.parent.mkdir(exist_ok=True)
                temporary = source_receipt.with_suffix(".importing")
                shutil.copyfile(parent / f"{stage}_receipt.json", temporary)
                if sha256(temporary) != receipt_hashes[stage]:
                    raise ValueError("Fitting-origin receipt changed during copy")
                os.replace(temporary, source_receipt)
            atomic_json(target, rebound)
        result = {"status": "COMPLETED", "parent_run": parent.name, "parent_source_commit": old_provenance["commit"],
                  "source_commit": provenance["commit"], "config_hash": provenance["config_hash"],
                  "reference_sha256": REFERENCE_HASH, "calibration_sha256": CALIBRATION_HASH,
                  "original_manifest_hashes": source_bundle["manifest_hashes"],
                  "original_manifest_paths": source_bundle["manifest_paths"], "original_receipt_sha256": receipt_hashes,
                  "reference_epochs_retrained": 0, "calibration_refitted": False,
                  "parent_main_checkpoint_reused": False, "main_start_epoch": 0, "main_start_optimizer_step": 0,
                  "selection_protocol": "test_selected", "at": timestamp()}
        atomic_json(run / "fitting_reuse.json", result)
        return source_bundle


def validate_pilot_receipt(run, provenance):
    receipt = read_json(run / "pilot_receipt.json")
    config = read_json(run / "resolved_config.json")
    if (receipt.get("status") != "COMPLETED" or receipt.get("outcome") not in {"READY_FOR_FULL", "NO_GO"}
            or receipt.get("protocol_id") != config["protocol_id"] or receipt.get("config_hash") != provenance["config_hash"]
            or receipt.get("source_commit") != provenance["commit"]):
        raise ValueError("Pilot final receipt is incomplete or bound to different source/config")
    for key in ("checkpoint", "best"):
        checkpoint = receipt.get(key, {})
        original = Path(checkpoint.get("path", ""))
        path = inside(run, original)
        expected_path = run / ("last.pth" if key == "checkpoint" else "best.pth")
        if (original.is_symlink() or path != expected_path or not path.is_file()
                or sha256(path) != checkpoint.get("sha256") or type(checkpoint.get("epoch")) is not int
                or not 0 <= checkpoint["epoch"] < config["training"]["epochs"]):
            raise ValueError("Pilot final checkpoint is missing or corrupt: " + key)
    from scripts.tect_diff.pilot_report import read_epochs
    epochs = read_epochs(run / "metrics_per_epoch.jsonl")
    if not epochs or epochs[-1]["epoch_index"] != receipt["checkpoint"]["epoch"]:
        raise ValueError("Pilot receipt is missing the corresponding complete epoch record")
    if receipt["outcome"] == "READY_FOR_FULL":
        confirmation = read_json(run / "confirmation_result.json")
        if (receipt.get("gate", {}).get("passed") is not True or confirmation.get("gate", {}).get("passed") is not True
                or confirmation.get("epoch") != receipt["checkpoint"]["epoch"]
                or confirmation.get("config_hash") != provenance["config_hash"]
                or confirmation.get("source_commit") != provenance["commit"]):
            raise ValueError("READY_FOR_FULL requires a complete same-checkpoint two-set confirmation")
    return receipt


def _spawn(run, root, config):
    env = runtime_environment(root, run.name)
    with (run / "controller.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([str(Path(config["environment"]) / "bin/python"), "-B", "-s",
                                  str(run / "source/scripts/tect_diff/pilot_controller.py"), "supervise", "--run-dir", str(run)],
                                 cwd=run / "source", env=env, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    identity = process_identity(child.pid)
    atomic_json(run / "launch_receipt.json", {"run_id": run.name, "controller_pid": child.pid, "identity": identity, "at": timestamp()})
    return {"status": "DISPATCHED", "run_id": run.name, "controller_pid": child.pid, "run_dir": str(run)}


def launch(config_file, run_id):
    config = read_json(config_file)
    validate_config(config, run_id)
    root = Path(config["project_root"]).resolve(strict=True)
    settings = config["runtime"]
    branch, commit = git(root, "branch", "--show-current"), git(root, "rev-parse", "HEAD")
    if (branch != settings["publish_branch"] or git(root, "status", "--porcelain", "--untracked-files=no")
            or git(root, "remote", "get-url", "--push", settings["publish_remote"]) != settings["publish_url"]):
        raise ValueError("Pilot launch requires the clean registered branch and remote")
    if git(root, "ls-remote", settings["publish_remote"], f"refs/heads/{branch}").split()[0] != commit:
        raise ValueError("Publish the exact server HEAD before pilot launch")
    run = inside(root, root / "runs" / run_id)
    with acquire_file(root / "runtime/locks" / f"tect-registration-{run_id}.lock"):
        if run.exists():
            return status(run)
        handles = acquire_all(root, run_id, config["gpus"])
        for handle in handles:
            handle.close()
        run.mkdir(parents=True)
        atomic_json(run / ".tect-created.json", {"run_id": run_id, "created_by": "tect-diff"})
        atomic_json(run / "resolved_config.json", config)
        hashes = source_snapshot(root, commit, run / "source")
        atomic_json(run / "provenance.json", {"run_id": run_id, "commit": commit, "branch": branch,
                    "repository": settings["publish_url"], "config_hash": json_hash(config), "source_hashes": hashes,
                    "created_at": timestamp(), "environment": config["environment"], "gpus": config["gpus"],
                    "config_source": str(Path(config_file).resolve()), "run_kind": "bounded_casia2_gn8_pilot"})
        packages = subprocess.check_output([str(Path(config["environment"]) / "bin/python"), "-B", "-s", "-m", "pip", "list", "--format=json"],
                                           text=True, env=runtime_environment(root, run_id), timeout=120)
        atomic_json(run / "environment.packages.json", json.loads(packages))
        atomic_json(run / "controller_status.json", {"status": "REGISTERED", "stage": "PREPARING", "run_id": run_id,
                    "updated_at": timestamp(), "commit": commit, "elapsed_seconds": 0})
        return _spawn(run, root, config)


def stop(run_dir):
    run, _, _, _ = _read_pilot(run_dir)
    state = status(run)
    identity = state.get("controller_identity")
    if not same_process(identity) or str(run) not in identity["cmdline"] or "pilot_controller.py" not in identity["cmdline"]:
        raise RuntimeError("No verified live pilot controller; refusing to signal another process")
    atomic_json(run / "operational_hold.json", {"active": True, "at": timestamp(), "reason": "Explicit pilot stop request"})
    os.kill(identity["pid"], signal.SIGTERM)
    return {"status": "STOP_REQUESTED", "run_id": run.name, "controller_pid": identity["pid"]}


def resume(run_dir):
    run, root, config, provenance = _read_pilot(run_dir)
    with acquire_file(root / "runtime/locks" / f"tect-registration-{run.name}.lock"):
        state = status(run)
        if state["controller_alive"] or state["worker_alive"] or state["status"] == "COMPLETED":
            return state
        # This explicit CLI action is the only resume path; never rerun a failed
        # numerical/gradient gate under the same source or clear its hold.
        hold = read_json(run / "operational_hold.json") if (run / "operational_hold.json").exists() else {}
        if state["status"] == "FAILED" or hold.get("requires_repair"):
            raise RuntimeError("Failed pilot requires a versioned repair, not automatic retry")
        if state.get("elapsed_seconds", 0) >= config["pilot"].get("total_deadline_seconds", 14400):
            raise RuntimeError("Pilot's cumulative four-hour budget is exhausted")
        verify_source(run, provenance)
        handles = acquire_all(root, run.name, config["gpus"])
        for handle in handles:
            handle.close()
        atomic_json(run / "operational_hold.json", {"active": False, "at": timestamp(), "reason": "Explicit same-source resume command"})
        return _spawn(run, root, config)


def _shutdown(child, identity, run):
    if child is not None and child.poll() is None and same_process(identity):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if same_process(identity):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=15)
    return stop_owned_members(run, child.pid) if child is not None else []


def supervise(run_dir):
    run, root, config, provenance = _read_pilot(run_dir)
    lock = acquire_file(root / "runtime/locks" / f"tect-controller-{run.name}.lock")
    state = read_json(run / "controller_status.json")
    hold = read_json(run / "operational_hold.json") if (run / "operational_hold.json").exists() else {}
    if state.get("status") in {"COMPLETED", "FAILED"} or hold.get("requires_repair"):
        lock.close()
        return status(run)
    previous_elapsed = float(state.get("elapsed_seconds", 0))
    start, deadline = time.monotonic(), config["pilot"].get("total_deadline_seconds", 14400)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    state.update(controller_pid=os.getpid(), controller_identity=process_identity(os.getpid()),
                 controller_pgid=os.getpgrp(), worker_identity=None, worker_pid=None, run_id=run.name, commit=provenance["commit"])
    resources, child, identity = [], None, None
    state_lock = threading.RLock()

    def update(**values):
        # The temporary JSON filename includes the PID, so the preparation
        # heartbeat and main control flow must serialize the complete write.
        with state_lock:
            state.update(values, elapsed_seconds=previous_elapsed + time.monotonic() - start, updated_at=timestamp())
            atomic_json(run / "controller_status.json", state)

    def check_stop():
        if stopped.is_set():
            raise InterruptedError("Pilot controller received a stop request")
        if previous_elapsed + time.monotonic() - start >= deadline:
            raise TimeoutError("Bounded pilot exceeded its cumulative four-hour deadline")

    try:
        hold = read_json(run / "operational_hold.json") if (run / "operational_hold.json").exists() else {}
        if hold.get("active"):
            raise InterruptedError("Pilot is held: " + hold.get("reason", "see operational_hold.json"))
        verify_source(run, provenance)
        update(status="STARTING", stage="PREPARING", failure_reason=None)
        env = runtime_environment(root, run.name)
        os.environ.update(env)
        while not resources:
            check_stop()
            try:
                resources = acquire_all(root, run.name, config["gpus"])
            except ResourceBusy as error:
                update(status="WAITING_FOR_RESOURCES", resource_block=str(error))
                stopped.wait(min(60, max(.1, deadline - previous_elapsed - (time.monotonic() - start))))
        check_stop()
        preparation_done = threading.Event()
        heartbeat_errors = []

        def preparation_heartbeat():
            while not preparation_done.wait(60):
                try:
                    expired = previous_elapsed + time.monotonic() - start >= deadline
                    update(preparation_stop_requested=stopped.is_set(), preparation_deadline_exceeded=expired)
                    with state_lock:
                        value = {key: state.get(key) for key in ("stage", "updated_at", "elapsed_seconds",
                                 "preparation_stop_requested", "preparation_deadline_exceeded")}
                    print(json.dumps({"event": "PREPARATION_HEARTBEAT", **value}), flush=True)
                except Exception as error:
                    heartbeat_errors.append(str(error))
                    return

        def preparation_progress(stage, message):
            update(status="RUNNING", stage=stage, resource_block=None)
            value = {"stage": stage, "message": message, "at": timestamp()}
            atomic_json(run / "progress.json", value)
            print(json.dumps(value, allow_nan=False), flush=True)
            check_stop()
            if heartbeat_errors:
                raise RuntimeError("Preparation heartbeat failed: " + heartbeat_errors[0])

        heartbeat = threading.Thread(target=preparation_heartbeat, name="pilot-preparation-heartbeat", daemon=True)
        heartbeat.start()
        try:
            preparation_progress("FITTING_REUSE", "Verifying E's frozen reference and calibration")
            inherited = import_fitting(run)
            preparation_progress("PREPARING", "Fitting dependencies verified; preparing fixed CASIA2/Test2 manifests")
            from scripts.tect_diff.pilot_data import prepare_pilot_bundle, load_pilot_bundle
            if not (run / "data_bundle.json").exists():
                bundle = prepare_pilot_bundle(root, config, inherited,
                    progress=lambda message: preparation_progress("PREPARING", message))
                atomic_json(run / "data_bundle.json", {**bundle["summary"], "summary_path": bundle["summary_path"]})
            preparation_progress("VERIFYING_MANIFESTS", "Loading and checking frozen pilot manifests")
            bundle = load_pilot_bundle(run / "data_bundle.json")
            for key in ("reference", "calibration"):
                if bundle["manifest_hashes"][key] != inherited["manifest_hashes"][key]:
                    raise ValueError("Pilot fitting manifest differs: " + key)
            preparation_progress("PREPARATION_COMPLETE", {
                "train_images": len(bundle["train"]),
                "test_images": {name: len(rows) for name, rows in bundle["tests"].items()}})
        finally:
            preparation_done.set()
            heartbeat.join(timeout=5)
        if heartbeat.is_alive():
            raise RuntimeError("Preparation heartbeat did not stop before GPU launch")
        check_stop()
        worker_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(map(str, config["gpus"])))
        command = [str(Path(config["environment"]) / "bin/python"), "-B", "-s", "-m", "torch.distributed.run",
                   "--standalone", "--nnodes=1", "--nproc-per-node=2", str(run / "source/scripts/tect_diff/pilot_worker.py"), "--run-dir", str(run)]
        with (run / "pilot.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=run / "source", env=worker_env, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(handle.fileno() for handle in resources))
        identity = process_identity(child.pid)
        update(status="RUNNING", stage="PILOT", worker_pid=child.pid, worker_identity=identity, worker_pgid=child.pid,
               worker_command=command)
        while child.poll() is None:
            check_stop()
            stopped.wait(min(60, max(.1, deadline - previous_elapsed - (time.monotonic() - start))))
            update()
        returncode = child.returncode
        remaining = _shutdown(child, identity, run)
        child = None
        update(worker_pid=None, worker_identity=None, worker_pgid=None, last_returncode=returncode)
        if returncode or remaining:
            raise RuntimeError(f"Pilot worker exit={returncode}, remaining owned workers={remaining}; see pilot.log")
        receipt = validate_pilot_receipt(run, provenance)
        update(status="COMPLETED", stage="COMPLETE", outcome=receipt["outcome"], completed_at=timestamp())
    except InterruptedError as error:
        update(status="INTERRUPTED", failure_reason=str(error))
        atomic_json(run / "operational_hold.json", {"active": True, "requires_repair": False, "reason": str(error), "at": timestamp()})
    except BaseException as error:
        traceback.print_exc()
        update(status="FAILED", outcome="HOLD", failure_reason=f"{type(error).__name__}: {error}")
        atomic_json(run / "operational_hold.json", {"active": True, "requires_repair": True, "reason": state["failure_reason"], "at": timestamp()})
    finally:
        cleanup_failure, remaining = None, []
        try:
            _shutdown(child, identity, run)
        except Exception as error:
            cleanup_failure = f"{type(error).__name__}: {error}"
        finally:
            try:
                if child is not None:
                    remaining = owned_group_members(run, child.pid)
                    if same_process(identity):
                        remaining.append(identity)
            except Exception as error:
                cleanup_failure = (cleanup_failure or "") + f"; process verification: {error}"
            for handle in resources:
                try:
                    handle.close()
                except Exception as error:
                    cleanup_failure = (cleanup_failure or "") + f"; resource close: {error}"
            if cleanup_failure or remaining:
                update(status="FAILED", outcome="HOLD", cleanup_failure=cleanup_failure,
                       remaining_owned_worker_pids=[item["pid"] for item in remaining],
                       failure_reason=state.get("failure_reason") or "Pilot worker cleanup incomplete")
                atomic_json(run / "operational_hold.json", {"active": True, "requires_repair": True,
                            "reason": state["failure_reason"], "at": timestamp()})
            update(resources_released=not remaining and cleanup_failure is None,
                   worker_pid=child.pid if remaining and child is not None else None,
                   worker_identity=identity if remaining else None,
                   worker_pgid=child.pid if remaining and child is not None else None)
        try:
            from scripts.tect_diff.pilot_report import generate, publish
            try:
                generate(run)
                # A pending_sync receipt is retried by supervision; never rerun
                # GPU work or hold this bounded task in hours of publish retries.
                published = publish(run)
                update(publication_status="SYNCED", publication_commit=published["commit"])
            except Exception as error:
                atomic_json(run / "pending_sync.json", {"pending_sync": True, "error": str(error), "at": timestamp()})
                update(publication_status="PENDING_SYNC")
        finally:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "supervise", "status", "stop", "resume"))
    parser.add_argument("--config")
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    if args.command == "launch":
        if not args.config or not args.run_id:
            parser.error("launch requires --config and --run-id")
        result = launch(args.config, args.run_id)
    else:
        directory = args.run_dir or (str(SOURCE / "runs" / args.run_id) if args.run_id else None)
        if directory is None:
            parser.error("--run-dir or --run-id is required")
        result = {"supervise": supervise, "status": status, "stop": stop, "resume": resume}[args.command](directory)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
