"""Detached, locked TECT-Diff dependency pipeline in a fixed Git source snapshot."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback

SOURCE = Path(__file__).resolve().parents[2]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from scripts.tect_diff.common import (atomic_json, git, inside, json_hash, read_json,
                                      runtime_environment, sha256, timestamp)
from tools.resource_locks import ResourceBusy, acquire_file, legacy_gpu_reservations

RUN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}")
FINAL_STATUSES = {"COMPLETED", "FAILED", "INTERRUPTED"}


def process_identity(pid):
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        suffix = text[text.rfind(")") + 2:].split()
        if suffix[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": suffix[19],
                "cmdline": Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")}
    except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
        return None


def same_process(identity):
    if not identity:
        return False
    now = process_identity(identity.get("pid"))
    return bool(now and now["start_ticks"] == identity.get("start_ticks"))


def owned_group_members(run, pgid):
    """Find attributable workers even if their torchrun group leader exited."""
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        identity = process_identity(int(directory.name))
        if not identity or str(run) not in identity["cmdline"] or "worker.py" not in identity["cmdline"]:
            continue
        try:
            if os.getpgid(identity["pid"]) == pgid:
                found.append(identity)
        except ProcessLookupError:
            continue
    return found


def stop_owned_members(run, pgid):
    members = owned_group_members(run, pgid)
    for identity in members:
        if same_process(identity):
            os.kill(identity["pid"], signal.SIGTERM)
    if members:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and any(same_process(identity) for identity in members):
            time.sleep(0.5)
        for identity in members:
            if same_process(identity):
                os.kill(identity["pid"], signal.SIGKILL)
    return [identity["pid"] for identity in members]


def read_run(run_dir):
    run = Path(run_dir).resolve(strict=True)
    config = read_json(run / "resolved_config.json")
    root = Path(config["project_root"]).resolve(strict=True)
    if run != inside(root, root / "runs" / run.name) or not RUN_PATTERN.fullmatch(run.name):
        raise ValueError("Invalid TECT run path")
    marker = read_json(run / ".tect-created.json")
    if marker != {"run_id": run.name, "created_by": "tect-diff"}:
        raise ValueError("Run creation marker mismatch")
    provenance = read_json(run / "provenance.json")
    if json_hash(config) != provenance["config_hash"]:
        raise ValueError("Frozen configuration hash mismatch")
    return run, root, config, provenance


def gpu_processes(gpus):
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    uuid_to_index = {line.split(",")[1].strip(): int(line.split(",")[0]) for line in query.splitlines() if line.strip()}
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                      "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",", 3)]
        if len(fields) == 4 and uuid_to_index.get(fields[0]) in gpus:
            rows.append({"gpu": uuid_to_index[fields[0]], "pid": int(fields[1]),
                         "name": fields[2], "memory_mib": fields[3]})
    if not set(gpus).issubset(uuid_to_index.values()):
        raise ResourceBusy("Both registered GPU IDs must exist")
    return rows


def acquire_all(root, run_id, gpus):
    handles = []
    try:
        handles.append(acquire_file(root / "runtime/locks" / f"run-{run_id}.lock"))
        for gpu in sorted(gpus):
            handles.append(acquire_file(root / "runtime/locks" / f"gpu-{gpu}.lock"))
        if set(gpus) & legacy_gpu_reservations(root):
            raise ResourceBusy("A registered GPU belongs to the legacy controller")
        occupied = gpu_processes(gpus)
        if occupied:
            raise ResourceBusy("GPU occupied; no unrelated process was stopped: " + json.dumps(occupied))
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def verify_source(run, provenance):
    for relative, expected in provenance["source_hashes"].items():
        source = inside(run / "source", run / "source" / relative)
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise RuntimeError(f"Fixed source snapshot changed: {relative}")


def source_snapshot(root, commit, destination):
    destination.mkdir()
    archive = subprocess.check_output(["git", "-C", str(root), "archive", "--format=tar", commit])
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        for member in source.getmembers():
            path = inside(destination, destination / member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as reader, path.open("wb") as writer:
                    writer.write(reader.read())
            else:
                raise RuntimeError("Source archive contains unsupported link/device entry")
    hashes = {path.relative_to(destination).as_posix(): sha256(path)
              for path in destination.rglob("*") if path.is_file()}
    for path in destination.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return hashes


def _spawn(run, root, config):
    source = run / "source/scripts/tect_diff/controller.py"
    python = Path(config["environment"]) / "bin/python"
    env = runtime_environment(root, run.name)
    log = (run / "controller.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen([str(python), "-s", str(source), "supervise", "--run-dir", str(run)],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=run / "source", env=env, start_new_session=True, close_fds=True)
    finally:
        log.close()
    identity = process_identity(process.pid)
    atomic_json(run / "launch_receipt.json", {"run_id": run.name, "controller_pid": process.pid,
                                              "identity": identity, "at": timestamp()})
    return {"status": "DISPATCHED", "run_id": run.name, "controller_pid": process.pid,
            "log": str(run / "controller.log"), "state": str(run / "controller_status.json")}


def launch(config_file, run_id, reference_parent=None, artifact_parent=None):
    if reference_parent and artifact_parent:
        raise ValueError('Choose one registered continuation boundary')
    if not RUN_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    config = read_json(config_file)
    root = Path(config["project_root"]).resolve(strict=True)
    if str(root) != "/data1/hl/DcDsDiff-and-GIT10K":
        raise ValueError("Launch is restricted to the authorized server project")
    if config["world_size"] != 2 or len(set(config["gpus"])) != 2:
        raise ValueError("TECT requires exactly two registered GPUs")
    if config["resolution"] not in (512, 352):
        raise ValueError("Unregistered input size")
    for section in ("training", "reference"):
        settings = config[section]
        if settings["micro_batch"] * settings["accumulation_steps"] * 2 != settings["global_batch"]:
            raise ValueError(f"Invalid effective batch: {section}")
    settings = config["runtime"]
    if settings["publish_branch"] != "feature/tect-diff" or settings["publish_url"] != "git@github.com:sleepDinner/DcDsDiff.git":
        raise ValueError("TECT publication must use the verified user project and feature/tect-diff")
    if not RUN_PATTERN.fullmatch(config["group_id"]):
        raise ValueError("Invalid group ID")
    if git(root, "remote", "get-url", "--push", settings["publish_remote"]) != settings["publish_url"]:
        raise ValueError("Remote is not the authorized user repository")
    commit = git(root, "rev-parse", "HEAD")
    branch = git(root, "branch", "--show-current")
    if branch != settings["publish_branch"]:
        raise ValueError("Server checkout is not the registered development branch")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Commit tracked modifications before launching")
    published = git(root, "ls-remote", settings["publish_remote"], f"refs/heads/{branch}").split()[0]
    if commit != published:
        raise ValueError("Server HEAD and GitHub branch must agree before launch")
    run = inside(root, root / "runs" / run_id)
    with acquire_file(root / "runtime/locks" / f"tect-registration-{run_id}.lock"):
        if run.exists():
            existing, _, _, _ = read_run(run)
            return status(existing)
        # Fail before registration when GPUs cannot be reserved now; the
        # supervisor repeats this race-safe gate before creating GPU workers.
        handles = acquire_all(root, run_id, config["gpus"])
        for handle in handles:
            handle.close()
        run.mkdir(parents=True)
        atomic_json(run / ".tect-created.json", {"run_id": run_id, "created_by": "tect-diff"})
        atomic_json(run / "resolved_config.json", config)
        hashes = source_snapshot(root, commit, run / "source")
        provenance = {"run_id": run_id, "commit": commit, "branch": branch,
                      "repository": settings["publish_url"], "config_hash": json_hash(config),
                      "source_hashes": hashes, "created_at": timestamp(), "environment": config["environment"],
                      "gpus": config["gpus"], "config_source": str(Path(config_file).resolve())}
        if reference_parent or artifact_parent:
            key = 'reference_parent' if reference_parent else 'artifact_parent'
            provenance[key] = str(Path(reference_parent or artifact_parent).resolve(strict=True))
            atomic_json(run / 'operational_hold.json', {
                'active': True, 'reason': 'Registered dependency import has not completed'})
        atomic_json(run / "provenance.json", provenance)
        env = runtime_environment(root, run_id)
        package_list = subprocess.check_output([str(Path(config["environment"]) / "bin/python"), "-s", "-m", "pip", "list", "--format=json"],
                                               text=True, env=env)
        atomic_json(run / "environment.packages.json", json.loads(package_list))
        atomic_json(run / "controller_status.json", {"status": "REGISTERED", "stage": "PREPARING", "run_id": run_id,
                                                       "updated_at": timestamp(), "commit": commit})
        if reference_parent:
            from scripts.tect_diff.reference_continuation import import_reference_state
            import_reference_state(run, reference_parent)
        if artifact_parent:
            from scripts.tect_diff.artifact_continuation import import_completed_artifacts
            import_completed_artifacts(run, artifact_parent)
        return _spawn(run, root, config)


def status(run_dir):
    run, root, config, provenance = read_run(run_dir)
    state = read_json(run / "controller_status.json")
    result = {**state, "controller_alive": same_process(state.get("controller_identity")),
              "worker_alive": same_process(state.get("worker_identity")),
              "run_dir": str(run), "config_hash": provenance["config_hash"]}
    if not result["worker_alive"] and state.get("worker_pgid"):
        members = owned_group_members(run, state["worker_pgid"])
        result["worker_alive"] = bool(members)
        result["remaining_owned_worker_pids"] = [item["pid"] for item in members]
    if (run / "progress.json").exists():
        result["progress"] = read_json(run / "progress.json")
    if state.get("status") not in FINAL_STATUSES and not result["controller_alive"]:
        launch_info = read_json(run / "launch_receipt.json") if (run / "launch_receipt.json").exists() else {}
        result["controller_alive"] = same_process(launch_info.get("identity"))
        if not result["controller_alive"]:
            result["status"] = "ORPHAN_RUNNING" if result["worker_alive"] else "STALE_STOPPED"
    return result


def check_operational_hold(run):
    hold = run / 'operational_hold.json'
    state = read_json(hold) if hold.exists() else {}
    if state.get('active'):
        raise RuntimeError('Run is held after user-requested stop: ' + state.get('reason', 'see operational_hold.json'))


def resume(run_dir):
    run, root, config, provenance = read_run(run_dir)
    check_operational_hold(run)
    with acquire_file(root / "runtime/locks" / f"tect-registration-{run.name}.lock"):
        check_operational_hold(run)
        current = status(run)
        if current["controller_alive"] or current["worker_alive"]:
            return current
        if current["status"] == "COMPLETED":
            return current
        verify_source(run, provenance)
        handles = acquire_all(root, run.name, config["gpus"])
        for handle in handles:
            handle.close()
        return _spawn(run, root, config)


def stop(run_dir):
    run, _, _, _ = read_run(run_dir)
    state = status(run)
    identity = state.get("controller_identity")
    if not same_process(identity):
        raise RuntimeError("No verified live controller; refusing to signal a possibly unrelated PID")
    if str(run) not in identity["cmdline"] or "controller.py" not in identity["cmdline"]:
        raise RuntimeError("Controller command line does not belong to this run")
    os.kill(identity["pid"], signal.SIGTERM)
    return {"run_id": run.name, "status": "STOP_REQUESTED", "controller_pid": identity["pid"]}


def validate_receipt(run, stage, provenance):
    path = run / f"{stage}_receipt.json"
    if not path.exists():
        return False
    receipt = read_json(path)
    config = read_json(run / "resolved_config.json")
    if receipt.get("status") != "COMPLETED" or receipt.get("config_hash") != provenance["config_hash"]:
        raise RuntimeError(f"Incompatible or incomplete {stage} receipt")
    if receipt.get("protocol_id") != config["protocol_id"]:
        raise RuntimeError(f"Protocol mismatch in {stage} receipt")
    if stage in ("reference", "calibration"):
        artifact = inside(config["project_root"], receipt["artifact_path"])
        if not artifact.is_file() or sha256(artifact) != receipt.get("sha256"):
            raise RuntimeError(f"Missing/corrupted {stage} artifact")
    if stage == "reference":
        if receipt.get("optimizer_step", 0) < 1 or receipt.get("initial_parameter_hash") == receipt.get("final_parameter_hash"):
            raise RuntimeError("Reference receipt does not demonstrate actual weight updates")
        if receipt.get("epoch") != config["reference"]["epochs"] - 1:
            raise RuntimeError("Reference did not reach its fixed training endpoint")
        if config['reference'].get('health_policy'):
            from scripts.tect_diff.reference_health import (
                POLICY_ID, FINAL_RATIO_THRESHOLD, require_healthy_reference, require_final_reference_probe,
            )
            import math
            metrics = receipt.get('health_metrics') or {}
            health = metrics.get('reference_health') or {}
            ratios = metrics.get('mse_ratio_by_scale', [])
            if (config['reference']['health_policy'] != POLICY_ID or health.get('policy_id') != POLICY_ID
                    or not health.get('final_endpoint_checked') or not health.get('all_scales_pass_final_margin')
                    or metrics.get('epoch') != config['reference']['epochs']-1
                    or len(ratios) != 4 or not all(math.isfinite(x) and 0 <= x < FINAL_RATIO_THRESHOLD for x in ratios)
                    or receipt.get('architecture_version') != config['reference']['architecture_version']):
                raise RuntimeError('Reference receipt is missing valid per-scale final learning evidence')
            require_healthy_reference(metrics)
            bundle = read_json(run / 'data_bundle.json')
            probe = receipt.get('final_health_probe')
            require_final_reference_probe(probe, config['reference'], receipt['final_parameter_hash'],
                                          bundle['manifest_hashes']['reference'])
            from scripts.tect_diff.data import canonical_hash
            records = read_json(bundle['manifest_paths']['reference'])
            if (canonical_hash(records) != bundle['manifest_hashes']['reference']
                    or probe['selected_reference_ids'] != [row['id'] for row in records[:16]]):
                raise RuntimeError('Final reference probe IDs differ from the frozen training manifest')
    if stage == "main":
        for endpoint in ("best", "final"):
            item = receipt[endpoint]
            artifact = inside(config["project_root"], item["path"])
            if not artifact.is_file() or sha256(artifact) != item["sha256"]:
                raise RuntimeError(f"Missing/corrupted {endpoint} checkpoint")
        if receipt["final"]["epoch"] != config["training"]["epochs"] - 1:
            raise RuntimeError("Main training did not reach its fixed endpoint")
    return True


def clean_empty_startup_files(run, state):
    """Remove only owned empty logs/atomic remnants before any valid training."""
    marker = read_json(run / ".tect-created.json")
    if marker != {"run_id": run.name, "created_by": "tect-diff"}:
        raise RuntimeError("Refusing cleanup without matching run creation marker")
    if state.get("status") != "FAILED" or state.get("stage") not in ("PREPARING", "PREFLIGHT", "STARTING"):
        return
    progress = read_json(run / "progress.json") if (run / "progress.json").exists() else {}
    if progress.get("optimizer_step", 0) > 0 or any(run.glob("*.pth")) or (run / "reference_receipt.json").exists():
        return
    removed = []
    for candidate in list(run.glob("*.log")) + list(run.glob("*.tmp")):
        if candidate.is_symlink():
            raise RuntimeError("Refusing cleanup through a symbolic link")
        path = inside(run, candidate)
        if path.is_file() and (path.suffix == ".tmp" or path.stat().st_size == 0):
            path.unlink()
            removed.append(path.name)
    atomic_json(run / "startup_cleanup_receipt.json", {"run_id": run.name, "removed": removed,
                                                       "retained_nonempty_failure_logs": True, "at": timestamp()})


def supervise(run_dir):
    run, root, config, provenance = read_run(run_dir)
    check_operational_hold(run)
    # A dedicated controller lock prevents duplicate supervisors even while
    # resource locks are unavailable and while final publication is retrying.
    controller_lock = acquire_file(root / "runtime/locks" / f"tect-controller-{run.name}.lock")
    state = read_json(run / "controller_status.json")
    stop_event = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop_event.set())
    start = time.monotonic()
    previous_elapsed = float(state.get("elapsed_seconds", 0))
    state.update(status="STARTING", controller_pid=os.getpid(), controller_identity=process_identity(os.getpid()),
                 controller_pgid=os.getpgrp(), commit=provenance["commit"], run_id=run.name,
                 worker_identity=None, worker_pid=None, failure_reason=None)
    resources, child = [], None
    stage_costs = state.get("stage_costs", {})
    def update(**values):
        state.update(values, updated_at=timestamp(), elapsed_seconds=previous_elapsed + time.monotonic() - start,
                     stage_costs=stage_costs)
        atomic_json(run / "controller_status.json", state)
    def summarize():
        from scripts.tect_diff.report import generate
        try:
            generate(run)
        except Exception as error:
            atomic_json(run / "report_failure.json", {"at": timestamp(), "error": str(error)})
    try:
        verify_source(run, provenance)
        env = runtime_environment(root, run.name)
        os.environ.update(env)
        while not stop_event.is_set():
            try:
                resources = acquire_all(root, run.name, config["gpus"])
                break
            except ResourceBusy as error:
                update(status="WAITING_FOR_RESOURCES", resource_block=str(error))
                stop_event.wait(config["runtime"]["heartbeat_seconds"])
        if stop_event.is_set():
            raise InterruptedError("Controller stopped before acquiring both GPUs")
        if not (run / "data_bundle.json").exists():
            update(status="RUNNING", stage="PREPARING", resource_block=None)
            from scripts.tect_diff.data import prepare_data
            def data_progress(value):
                atomic_json(run / "progress.json", {"stage": "PREPARING", "message": str(value), "at": timestamp()})
                update()
                if stop_event.is_set():
                    raise InterruptedError("Stopped during data preparation")
            data_start = time.monotonic()
            bundle = prepare_data(root, config, progress=data_progress)
            atomic_json(run / "data_bundle.json", {**bundle["summary"], "summary_path": bundle["summary_path"]})
            stage_costs["preparation_seconds"] = time.monotonic() - data_start
        bundle = read_json(run / "data_bundle.json")
        if bundle["reference_source_mode"] == "mask_clean_proxy" and "MASKCLEAN" not in run.name:
            raise RuntimeError("Reference requires mask_clean_proxy; register the run with MASKCLEAN label before training")
        update(reference_source_mode=bundle["reference_source_mode"])
        summarize()
        for stage in ("preflight", "reference", "calibration", "main"):
            if stop_event.is_set():
                raise InterruptedError("Controller received stop request")
            if validate_receipt(run, stage, provenance):
                continue
            update(status="RUNNING", stage=stage.upper())
            stage_start = time.monotonic()
            worker_env = dict(env, CUDA_VISIBLE_DEVICES=",".join(map(str, config["gpus"])))
            command = [str(Path(config["environment"]) / "bin/python"), "-s", "-m", "torch.distributed.run",
                       "--standalone", "--nnodes=1", "--nproc-per-node=2",
                       str(run / "source/scripts/tect_diff/worker.py"), "--run-dir", str(run), "--stage", stage]
            with (run / f"{stage}.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, cwd=run / "source", env=worker_env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=tuple(handle.fileno() for handle in resources))
            identity = process_identity(child.pid)
            update(worker_pid=child.pid, worker_identity=identity, worker_pgid=child.pid, worker_command=command)
            while child.poll() is None:
                if stop_event.wait(config["runtime"]["heartbeat_seconds"]):
                    if same_process(identity):
                        os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        if same_process(identity):
                            os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=30)
                    raise InterruptedError("Controller terminated only its registered worker process group")
                update()
            stage_costs[stage + "_seconds"] = stage_costs.get(stage + "_seconds", 0) + time.monotonic() - stage_start
            returncode = child.returncode
            remaining = stop_owned_members(run, child.pid)
            child = None
            update(worker_pid=None, worker_identity=None, worker_pgid=None, last_returncode=returncode)
            if remaining:
                raise RuntimeError(f"{stage} torchrun exited with remaining owned workers {remaining}; stopped only verified run members")
            if returncode:
                raise RuntimeError(f"{stage} worker failed with exit code {returncode}; see {run / (stage + '.log')}")
            if not validate_receipt(run, stage, provenance):
                raise RuntimeError(f"{stage} exited without a complete verified receipt")
            summarize()
        update(status="COMPLETED", stage="COMPLETE", completed_at=timestamp())
    except InterruptedError as error:
        update(status="INTERRUPTED", failure_reason=str(error))
    except BaseException as error:
        traceback.print_exc()
        update(status="FAILED", failure_reason=f"{type(error).__name__}: {error}")
    finally:
        if child is not None and child.poll() is None:
            identity = state.get("worker_identity")
            if same_process(identity):
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    if same_process(identity):
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=30)
        if child is not None:
            stop_owned_members(run, child.pid)
        for handle in resources:
            handle.close()
        update(resources_released=True, worker_pid=None, worker_identity=None, worker_pgid=None)
        try:
            clean_empty_startup_files(run, state)
        except Exception as error:
            update(cleanup_failure=str(error))
        summarize()
        # Training outcome is preserved even if GitHub is unreachable. Retry
        # publication on this server with bounded attempts; never rerun training.
        from scripts.tect_diff.report import publish
        attempts = int(config["runtime"]["publish_attempts"])
        for attempt in range(attempts):
            try:
                receipt = publish(run)
                update(publication_status="SYNCED", publication_commit=receipt["commit"])
                break
            except Exception as error:
                atomic_json(run / "pending_sync.json", {"pending_sync": True, "attempt": attempt + 1,
                                                         "error": str(error), "at": timestamp()})
                update(publication_status="PENDING_SYNC")
                if attempt + 1 == attempts or stop_event.is_set():
                    break
                stop_event.wait(config["runtime"]["publish_retry_seconds"])
        controller_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "resume", "status", "stop", "supervise"))
    parser.add_argument("--config")
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir")
    parser.add_argument("--reference-parent", help="Stopped same-config reference run; state-preserving source-version continuation")
    parser.add_argument("--artifact-parent", help="Registered held MAIN repair parent; reuse completed reference/calibration and initialize MAIN fresh")
    args = parser.parse_args()
    if args.command == "launch":
        if not args.config or not args.run_id:
            parser.error("launch requires --config and --run-id")
        result = launch(args.config, args.run_id, args.reference_parent, args.artifact_parent)
    else:
        run_dir = args.run_dir or (str(SOURCE / "runs" / args.run_id) if args.run_id else None)
        if run_dir is None:
            parser.error("--run-dir or --run-id is required")
        result = {"resume": resume, "status": status, "stop": stop, "supervise": supervise}[args.command](run_dir)
    if result is not None:
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
