"""Detached single-GPU train/evaluate controller with immutable source snapshots."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import signal
import site
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.resource_locks import acquire_resources, inherit_file


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def output(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def proc_identity(pid):
    try:
        # Fields after the final ')' start at /proc stat field 3.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def live_stage_processes(group_id):
    """Find live members of the session created for one owned stage."""
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == group_id and int(fields[3]) == group_id:
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def terminate_stage(proc, grace_seconds=30):
    # A dead group leader can leave DataLoader descendants holding the project lock.
    # Therefore signalling must not depend on proc.poll() being None.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        proc.poll()
        if not live_stage_processes(proc.pid):
            return
        time.sleep(.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        proc.poll()
        if not live_stage_processes(proc.pid):
            return
        time.sleep(.1)
    raise RuntimeError(f'Owned stage processes survived SIGKILL: {live_stage_processes(proc.pid)}')


def run_path(root, run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', run_id):
        raise ValueError('Invalid run ID')
    path = root / 'runs' / run_id
    if path.resolve().parent != (root / 'runs').resolve():
        raise ValueError('Run path escapes the project')
    return path


def validate_gpu(gpu):
    info = output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used,utilization.gpu',
                   '--format=csv,noheader,nounits'])
    uuid, memory, utilization = [part.strip() for part in info.split(',')]
    processes = output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'])
    if int(memory) > 512 or int(utilization) > 10 or uuid in processes:
        raise SystemExit(f'GPU {gpu} is occupied; leave its work untouched and choose another GPU.')
    return info


def validate_data_config(config, root):
    if config.batch_size != 6 or config.gradient_accumulate_every != 1:
        raise SystemExit('This registered controller requires global batch 6 without accumulation.')
    for section, split, size_key in ((config.train_dataset, 'train', 'trainsize'),
                                      (config.test_dataset.Mix, 'test', 'testsize')):
        if section.params[size_key] != 352:
            raise SystemExit('This registered controller requires 352px inputs.')
        for key, kind in (('image_root', 'f'), ('gt_root', 'm'), ('de_root', 'd'), ('trace_root', 't')):
            configured = (root / section.params[key]).resolve()
            expected = (root / 'data/git10k-recon-v1' / split / kind).resolve()
            if configured != expected:
                raise SystemExit(f'{key} does not refer to the registered {split} manifest data.')


def validate_environment(root, environment_source):
    if Path(sys.prefix).resolve() != Path('/data0/hl/conda_envs/dcdsdiff').resolve():
        raise SystemExit('Run this controller using /data0/hl/conda_envs/dcdsdiff/bin/python.')
    environment = read_json(root / 'runtime/bootstrap/environment.ready')
    installed = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    if (environment['status'] != 'READY'
            or sha256(environment_source / 'environment/requirements.txt') != environment['requirements_sha256']
            or sha256(environment_source / 'environment/requirements.lock.txt') != environment['dependency_lock_sha256']
            or hashlib.sha256(installed).hexdigest() != environment['freeze_sha256']):
        raise SystemExit('Environment differs from its verified requirements/lock/freeze receipt.')


def freeze_source(root, source, commit):
    source.mkdir()
    archive = subprocess.check_output(['git', 'archive', commit], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            dest = (source / member.name).resolve()
            if not dest.is_relative_to(source) or member.issym() or member.islnk():
                raise RuntimeError('Unexpected archive member')
        tar.extractall(source)
    (source / 'data').symlink_to(root / 'data', target_is_directory=True)
    (source / 'pretrained_weights').mkdir(exist_ok=True)
    (source / 'pretrained_weights/pvt_v2_b2.pth').symlink_to(root / 'pretrained_weights/pvt_v2_b2.pth')
    return {p.relative_to(source).as_posix(): sha256(p)
            for p in source.rglob('*') if p.is_file() and not p.is_symlink()
            and 'data' not in p.relative_to(source).parts}


def start(args):
    root = args.project.resolve()
    run = run_path(root, args.run_id)
    if args.command == 'resume' and args.gpu != read_json(run / 'provenance.json')['gpu']:
        raise SystemExit('Resume must use the GPU registered in this run\'s provenance.')
    frozen_controller = run / 'source/tools/manage_experiment.py'
    if args.command == 'resume' and frozen_controller.is_file() and '--run-lock-fd' not in frozen_controller.read_text():
        # Keep legacy recovery on its tested controller and inherited global lock.
        # New launchers attribute that lock to its GPU, so the second arm can coexist.
        os.execv(sys.executable, [sys.executable, '-s', str(frozen_controller), 'resume',
                                 '--project', str(root), '--run-id', args.run_id, '--gpu', str(args.gpu)])
    locks = acquire_resources(root, args.run_id, args.gpu)
    validate_environment(root, root if args.command == 'launch' else run / 'source')
    gpu_info = validate_gpu(args.gpu)
    if args.command == 'launch':
        if run.exists():
            raise SystemExit('Run exists. Use resume for an interrupted run or choose a new ID.')
        dirty = output(['git', 'status', '--porcelain', '--untracked-files=no'], root)
        if dirty:
            raise SystemExit('Commit all tracked changes before launch.')
        commit = output(['git', 'rev-parse', 'HEAD'], root)
        config_relative = Path(args.config)
        config_file = (root / config_relative).resolve()
        if config_relative.is_absolute() or not config_file.is_relative_to(root):
            raise SystemExit('Config must be a committed path inside this project.')
        output(['git', 'ls-files', '--error-unmatch', config_relative.as_posix()], root)
        sys.path.insert(0, str(root))
        from utils.init_utils import _load_config
        config = _load_config(config_file)
        benchmark = config.get('data_protocol') == 'casia2-all8-v1'
        if benchmark:
            from tools.benchmark_protocol import validate_casia_config
            _, snapshot, receipt = validate_casia_config(config, root)
            manifest_relative = snapshot.relative_to(root) / 'manifest.csv'
        else:
            if config.get('data_protocol') is not None:
                raise SystemExit('Unregistered data protocol.')
            validate_data_config(config, root)
            receipt = read_json(root / 'data/git10k-recon-v1/dataset_receipt.json')
            if receipt['status'] != 'READY' or receipt['train_count'] != 9000 or receipt['test_count'] != 1000:
                raise SystemExit('Dataset receipt is not READY for this protocol.')
            manifest_relative = Path('data/git10k-recon-v1/manifest.csv')
        if config.num_epoch != 100 or config.diffusion_model.params.num_sample_steps != 10:
            raise SystemExit('This controller requires the registered 100-epoch / 10-step protocol.')
        manifest = root / manifest_relative
        if sha256(manifest) != receipt['manifest_sha256']:
            raise SystemExit('Dataset manifest hash mismatch.')
        weights = root / 'pretrained_weights/pvt_v2_b2.pth'
        if sha256(weights) != '80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d':
            raise SystemExit('Pretrained weight checksum differs from audited PVT-B2.')
        run.mkdir(parents=True)
        source = run / 'source'
        source_hashes = freeze_source(root, source, commit)
        (run / 'environment.freeze.txt').write_text(output([sys.executable, '-m', 'pip', 'freeze', '--all']) + '\n')
        provenance = {
            'run_id': args.run_id, 'created_at': now(), 'git_commit': commit,
            'protocol_id': config.protocol_id, 'source': str(source),
            'python': sys.executable, 'gpu': args.gpu, 'gpu_snapshot': gpu_info,
            'config_path': config_relative.as_posix(),
            'config_sha256': sha256(source / config_relative),
            'source_files_sha256': source_hashes,
            'dataset_manifest_sha256': receipt['manifest_sha256'],
            'dataset_manifest_path': manifest_relative.as_posix(),
            'evaluation_kind': 'all8_best' if benchmark else 'git10k_final',
            'benchmark_spec': config.get('benchmark_spec'),
            'pretrained_sha256': sha256(weights), 'dataset': receipt,
            'checkpoint_policy': ('retain epoch99; requested All8 report uses saved best pooled test MAE'
                                  if benchmark else 'epoch99 final primary; best MAE is test-selected diagnostic only'),
        }
        write_json(run / 'provenance.json', provenance)
    else:
        provenance = read_json(run / 'provenance.json')
        source = run / 'source'
        status = read_json(run / 'controller_status.json')
        if status['status'] == 'COMPLETED':
            raise SystemExit('Experiment already completed; refusing to rerun it.')
        if not (run / 'model-last.pt').is_file():
            raise SystemExit('No complete epoch checkpoint available to resume.')
        if sha256(source / provenance['config_path']) != provenance['config_sha256']:
            raise SystemExit('Frozen configuration changed.')
        for name, digest in provenance['source_files_sha256'].items():
            if sha256(source / name) != digest:
                raise SystemExit(f'Frozen source changed: {name}')
        if sha256(root / provenance.get('dataset_manifest_path', 'data/git10k-recon-v1/manifest.csv')) != provenance['dataset_manifest_sha256']:
            raise SystemExit('Dataset changed since launch.')
        if provenance.get('evaluation_kind') == 'all8_best':
            from tools.benchmark_protocol import load_benchmark, verify_benchmark_files
            _, snapshot, _, rows = load_benchmark(source, provenance['benchmark_spec'], require_disjoint_train=True)
            verify_benchmark_files(snapshot, rows)
    command = [sys.executable, str(source / 'tools/manage_experiment.py'), '_worker',
               '--project', str(root), '--run-id', args.run_id, '--gpu', str(args.gpu),
               '--lock-fd', str(locks[0].fileno()), '--run-lock-fd', str(locks[1].fileno())]
    if args.command == 'resume':
        command.append('--resume-training')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
                       WANDB_MODE='disabled', TOKENIZERS_PARALLELISM='false')
    with (run / 'controller.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=source, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                start_new_session=True, pass_fds=tuple(lock.fileno() for lock in locks), env=environment)
    write_json(run / 'controller_identity.json', {'pid': proc.pid, 'start_ticks': proc_identity(proc.pid)})
    # The worker and active child inherit the lock; it survives this command and SSH.
    for lock in locks:
        lock.close()
    print(json.dumps({'run_id': args.run_id, 'controller_pid': proc.pid, 'run_path': str(run)}))


def worker(args):
    root = args.project.resolve()
    run = run_path(root, args.run_id)
    locks = [inherit_file(args.lock_fd, root / 'runtime/locks' / f'gpu-{args.gpu}.lock'),
             inherit_file(args.run_lock_fd, root / 'runtime/locks' / f'run-{args.run_id}.lock')]
    source = run / 'source'
    state = {'run_id': args.run_id, 'controller_pid': os.getpid(), 'started_at': now(),
             'gpu': args.gpu, 'status': 'STARTING'}
    def update(**changes):
        state.update(changes, updated_at=now())
        write_json(run / 'controller_status.json', state)
    def stopped(signum, frame):
        raise InterruptedError(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    proc = None
    try:
        update()
        provenance = read_json(run / 'provenance.json')
        command = [sys.executable, 'train.py', '--config', provenance['config_path'],
                   '--results_folder', str(run), '--log_with', 'none']
        if args.resume_training:
            command.extend(['--resume', str(run / 'model-last.pt')])
        stages = [('TRAINING', command, 'train.process.log'),
                  ('EVALUATING', [sys.executable, 'tools/evaluate_reproduction.py',
                    '--config', str(run / 'resolved_config.yaml'), '--checkpoint', str(run / 'model-final.pt'),
                    '--output', str(run / 'evaluation')], 'evaluation.log')]
        checkpoint = run / 'model-final.pt'
        if provenance.get('evaluation_kind') == 'all8_best':
            checkpoint = run / 'model-best.pt'
            stages[1] = ('EVALUATING', [sys.executable, 'tools/evaluate_benchmarks.py',
                         '--run', str(run), '--spec', provenance['benchmark_spec'],
                         '--output', str(run / 'evaluation')], 'evaluation.log')
        for stage, command, logfile in stages:
            if stage == 'TRAINING' and (run / 'training_status.json').exists():
                training = read_json(run / 'training_status.json')
                if (training['state'] == 'COMPLETED' and training['next_epoch'] == 100
                        and (run / 'model-final.pt').is_file()):
                    continue
            if stage == 'EVALUATING' and (run / 'evaluation/results.json').exists():
                result = read_json(run / 'evaluation/results.json')
                if result.get('status') == 'COMPLETED' and result['checkpoint_sha256'] == sha256(checkpoint):
                    continue
                raise RuntimeError('Existing evaluation does not match the registered checkpoint.')
            with (run / logfile).open('a') as log:
                print(f'{now()} {stage}: {command}', flush=True)
                proc = subprocess.Popen(command, cwd=source, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=log, pass_fds=tuple(lock.fileno() for lock in locks),
                                        start_new_session=True)
                update(status=stage, child_pid=proc.pid, command=command)
                while True:
                    try:
                        code = proc.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        update()
                if code != 0:
                    raise RuntimeError(f'{stage} exited with code {code}; see {logfile}')
                # Completed stages must also release all inherited lock descriptors.
                terminate_stage(proc, grace_seconds=5)
                proc = None
        result = read_json(run / 'evaluation/results.json')
        checkpoint_fields = {'evaluated_checkpoint_sha256': result['checkpoint_sha256']}
        if provenance.get('evaluation_kind', 'git10k_final') == 'git10k_final':
            checkpoint_fields['final_checkpoint_sha256'] = result['checkpoint_sha256']
        update(status='COMPLETED', child_pid=None, finished_at=now(), metrics=result['metrics'],
               **checkpoint_fields)
    except BaseException as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cleanup_error = None
        if proc is not None:
            try:
                terminate_stage(proc)
            except Exception as cleanup:
                cleanup_error = str(cleanup)
        update(status='FAILED_CLEANUP' if cleanup_error else (
                   'INTERRUPTED' if isinstance(error, InterruptedError) else 'FAILED'),
               child_pid=proc.pid if cleanup_error and proc is not None else None,
               error=str(error), cleanup_error=cleanup_error, finished_at=now())
        raise
    finally:
        for lock in locks:
            lock.close()


def status_or_stop(args):
    run = run_path(args.project.resolve(), args.run_id)
    state = read_json(run / 'controller_status.json')
    identity = read_json(run / 'controller_identity.json')
    alive = identity['start_ticks'] is not None and proc_identity(identity['pid']) == identity['start_ticks']
    if args.command == 'stop':
        if not alive:
            raise SystemExit('No matching live controller; nothing was signalled.')
        os.kill(identity['pid'], signal.SIGTERM)
        print('Stop requested for this experiment controller only; resume starts at last complete epoch.')
        return
    state['controller_alive'] = alive
    progress = run / 'training_status.json'
    if progress.exists():
        state['training'] = read_json(progress)
    print(json.dumps(state, indent=2))


def main():
    os.environ['PYTHONNOUSERSITE'] = '1'
    if site.ENABLE_USER_SITE:
        os.execv(sys.executable, [sys.executable, '-s', *sys.argv])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('launch', 'resume', 'status', 'stop', '_worker'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config', default='config/reproduction.yaml')
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--run-lock-fd', type=int)
    parser.add_argument('--resume-training', action='store_true')
    args = parser.parse_args()
    if args.command in ('launch', 'resume'):
        start(args)
    elif args.command == '_worker':
        worker(args)
    else:
        status_or_stop(args)


if __name__ == '__main__':
    main()
