"""Attach one durable All8 best-checkpoint evaluation to an existing experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import site
import subprocess
import sys
import tempfile
import time

if __name__ == '__main__':
    os.environ['PYTHONNOUSERSITE'] = '1'
    if site.ENABLE_USER_SITE:
        os.execv(sys.executable, [sys.executable, '-s', *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_protocol import load_benchmark
from tools.manage_experiment import (freeze_source, now, output, proc_identity, read_json, run_path,
                                    sha256, terminate_stage, validate_environment, validate_gpu, write_json)
from tools.resource_locks import ResourceBusy, acquire_file, acquire_resources, inherit_file


def controller_alive(run):
    identity = read_json(run / 'controller_identity.json')
    return identity['start_ticks'] is not None and proc_identity(identity['pid']) == identity['start_ticks']


def dependency_ready(run):
    state = read_json(run / 'controller_status.json')
    alive = controller_alive(run)
    if state['status'] in ('FAILED', 'INTERRUPTED', 'FAILED_CLEANUP'):
        raise RuntimeError(f"Training dependency needs attention: {state['status']}")
    if state['status'] == 'COMPLETED':
        return not alive
    if not alive:
        raise RuntimeError('Training controller disappeared before completion; no checkpoint was evaluated.')
    return False


def queue(args, root, run, job):
    lock = acquire_file(root / 'runtime/locks' / f'followup-{args.run_id}.lock')
    if args.command == 'queue':
        if job.exists():
            raise SystemExit('Follow-up already exists; use status or resume rather than duplicating it.')
        validate_environment(root, root)
        spec, _, receipt, _ = load_benchmark(root)
        if output(['git', 'status', '--porcelain', '--untracked-files=no'], root):
            raise SystemExit('Commit changes before freezing the follow-up.')
        original = read_json(run / 'provenance.json')
        job.parent.mkdir(parents=True, exist_ok=True)
        commit = output(['git', 'rev-parse', 'HEAD'], root)
        # A failed archive/extraction must not leave a half-created permanent job.
        with tempfile.TemporaryDirectory(prefix='all8-best.preparing-', dir=job.parent) as draft:
            staging = Path(draft)
            hashes = freeze_source(root, staging / 'source', commit)
            write_json(staging / 'provenance.json', {
                'created_at': now(), 'training_run_id': args.run_id, 'training_commit': original['git_commit'],
                'evaluation_commit': commit, 'gpu': original['gpu'], 'suite_id': spec['suite_id'],
                'dataset_manifest_sha256': receipt['manifest_sha256'], 'source_files_sha256': hashes,
                'checkpoint': 'model-best.pt', 'policy': 'wait for existing train/final-eval controller completion; evaluate saved best once',
            })
            write_json(staging / 'status.json', {'status': 'QUEUED', 'updated_at': now()})
            staging.rename(job)
    else:
        if read_json(job / 'status.json')['status'] == 'COMPLETED':
            raise SystemExit('Follow-up already completed; no repeat evaluation.')
        provenance = read_json(job / 'provenance.json')
        for name, digest in provenance['source_files_sha256'].items():
            if sha256(job / 'source' / name) != digest:
                raise SystemExit(f'Frozen follow-up source changed: {name}')
        validate_environment(root, job / 'source')
    environment = os.environ.copy()
    environment.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled')
    write_json(job / 'status.json', {'status': 'QUEUED', 'updated_at': now(), 'training_run_id': args.run_id})
    with (job / 'controller.log').open('a') as log:
        proc = subprocess.Popen([sys.executable, str(job / 'source/tools/queue_benchmark.py'), '_worker',
                                 '--project', str(root), '--run-id', args.run_id, '--lock-fd', str(lock.fileno())],
                                cwd=job / 'source', env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                start_new_session=True, pass_fds=(lock.fileno(),))
    write_json(job / 'controller_identity.json', {'pid': proc.pid, 'start_ticks': proc_identity(proc.pid)})
    lock.close()
    print({'status': 'QUEUED', 'pid': proc.pid, 'job': str(job)})


def worker(args, root, run, job):
    lock = inherit_file(args.lock_fd, root / 'runtime/locks' / f'followup-{args.run_id}.lock')
    resources, proc = [], None
    provenance = read_json(job / 'provenance.json')
    state = {'training_run_id': args.run_id, 'controller_pid': os.getpid(), 'started_at': now()}

    def update(status, **fields):
        state.update(status=status, updated_at=now(), **fields)
        write_json(job / 'status.json', state)

    def stop(signum, frame):
        raise InterruptedError(f'Follow-up received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not dependency_ready(run):
            update('WAITING_FOR_TRAINING', dependency=read_json(run / 'controller_status.json')['status'])
            time.sleep(60)
        if read_json(run / 'provenance.json')['git_commit'] != provenance['training_commit']:
            raise RuntimeError('Training run identity changed.')
        validate_environment(root, job / 'source')
        _, _, receipt, _ = load_benchmark(job / 'source')
        if receipt['manifest_sha256'] != provenance['dataset_manifest_sha256']:
            raise RuntimeError('Follow-up dataset changed since registration.')
        result_path = job / 'evaluation/results.json'
        if result_path.exists():
            result = read_json(result_path)
            if result['status'] != 'COMPLETED' or result['checkpoint_sha256'] != sha256(run / 'model-best.pt'):
                raise RuntimeError('Existing follow-up result does not match the saved best.')
        else:
            while True:
                try:
                    resources = acquire_resources(root, args.run_id, provenance['gpu'])
                    validate_gpu(provenance['gpu'])
                    break
                except (ResourceBusy, SystemExit):
                    for handle in resources:
                        handle.close()
                    resources = []
                    update('WAITING_FOR_GPU', gpu=provenance['gpu'])
                    time.sleep(60)
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = str(provenance['gpu'])
            command = [sys.executable, 'tools/evaluate_benchmarks.py', '--run', str(run),
                       '--output', str(job / 'evaluation')]
            with (job / 'evaluation.log').open('a') as log:
                proc = subprocess.Popen(command, cwd=job / 'source', env=environment, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=log, start_new_session=True,
                                        pass_fds=tuple(x.fileno() for x in [lock, *resources]))
            update('EVALUATING', gpu=provenance['gpu'], child_pid=proc.pid)
            while True:
                try:
                    code = proc.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    update('EVALUATING')
            if code:
                raise RuntimeError(f'Benchmark evaluator failed with exit code {code}.')
            terminate_stage(proc, grace_seconds=5)
            proc = None
            result = read_json(result_path)
        update('COMPLETED', finished_at=now(), child_pid=None, metrics=result['metrics'],
               checkpoint_sha256=result['checkpoint_sha256'])
    except BaseException as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cleanup = None
        if proc is not None:
            try:
                terminate_stage(proc)
            except Exception as failure:
                cleanup = str(failure)
        update('FAILED_CLEANUP' if cleanup else ('INTERRUPTED' if isinstance(error, InterruptedError) else 'FAILED'),
               error=str(error), cleanup_error=cleanup, child_pid=proc.pid if cleanup and proc else None)
        raise
    finally:
        for handle in [lock, *resources]:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('queue', 'resume', 'status', 'stop', '_worker'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    root = args.project.resolve()
    run = run_path(root, args.run_id)
    job = run / 'followups/all8-best'
    if args.command in ('queue', 'resume'):
        queue(args, root, run, job)
    elif args.command == '_worker':
        worker(args, root, run, job)
    else:
        alive = controller_alive(job)
        if args.command == 'stop':
            if not alive:
                raise SystemExit('No matching live follow-up controller.')
            os.kill(read_json(job / 'controller_identity.json')['pid'], signal.SIGTERM)
        else:
            print({**read_json(job / 'status.json'), 'controller_alive': alive})


if __name__ == '__main__':
    main()
