"""Register the authorized CASIA2 selection change at a complete epoch boundary."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
import site
import sys
import tempfile

if __name__ == '__main__':
    os.environ['PYTHONNOUSERSITE'] = '1'
    if site.ENABLE_USER_SITE:
        os.execv(sys.executable, [sys.executable, '-s', *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omegaconf import OmegaConf
import torch

from tools.benchmark_protocol import validate_casia_config
from tools.manage_experiment import (freeze_source, now, output, proc_identity, read_json, run_path,
                                    sha256, start, validate_environment, validate_gpu, write_json)
from tools.resource_locks import acquire_file, acquire_resources
from utils.init_utils import _load_config
from utils.trainer import atomic_checkpoint


def migrate_checkpoint(checkpoint, parent_config, config, previous_datasets):
    """Change selection metadata only; all optimization and random state is retained."""
    required = {'checkpoint_format', 'selection', 'epoch', 'next_epoch', 'global_step', 'contract',
                'model', 'opt', 'scheduler', 'scaler', 'rng', 'epoch_records', 'best_mae', 'best_epoch',
                'architecture_version', 'pretrained_load_report'}
    if not required <= checkpoint.keys() or checkpoint['checkpoint_format'] != 2:
        raise ValueError('A complete format-2 epoch checkpoint is required.')
    if (checkpoint['selection'] != 'EPOCH_BOUNDARY_RECOVERY'
            or checkpoint['next_epoch'] != checkpoint['epoch'] + 1
            or not 0 < checkpoint['next_epoch'] < 100
            or len(checkpoint['epoch_records']) != checkpoint['next_epoch']
            or checkpoint['opt'] is None or checkpoint['scheduler'] is None
            or not math.isfinite(checkpoint['best_mae'])):
        raise ValueError('Parent is not an unfinished complete epoch-boundary training checkpoint.')
    old = OmegaConf.to_container(parent_config, resolve=True)
    new = OmegaConf.to_container(config, resolve=True)
    if old['protocol_id'] != 'CASIA2-ALL8-V1' or new['protocol_id'] != 'CASIA2-SEL3-ALL8-V1':
        raise ValueError('Only the registered All8-to-three-set continuation is supported.')
    contract = checkpoint['contract']
    for key, value in contract.items():
        if key in ('train_num_epoch', 'mixed_precision'):
            if value != (100 if key == 'train_num_epoch' else 'no'):
                raise ValueError('Epoch horizon or precision changed.')
            continue
        if old.get(key) != value:
            raise ValueError(f'Parent checkpoint/config mismatch: {key}')
        if key not in ('protocol_id', 'test_dataset') and new.get(key) != value:
            raise ValueError(f'Continuation may not change training contract: {key}')
    if [row['epoch'] for row in checkpoint['epoch_records']] != list(range(checkpoint['next_epoch'])):
        raise ValueError('Parent epoch history is incomplete.')
    migrated = dict(checkpoint)
    migrated['contract'] = {**contract, 'protocol_id': new['protocol_id'], 'test_dataset': new['test_dataset']}
    migrated['best_mae'], migrated['best_epoch'] = float('inf'), None
    migrated['epoch_records'] = [dict(row, diagnostic_datasets=list(previous_datasets))
                                 for row in checkpoint['epoch_records']]
    return migrated


def publish_continuation(draft, run, parent, checkpoint_hash):
    marker = {'run_id': run.name, 'registered_at': now(), 'phase': 'PREPARING',
              'parent_checkpoint_sha256': checkpoint_hash,
              'reason': 'User changed per-epoch checkpoint selection to three datasets.'}
    # Block parent resume before the child can become visible. A failed publish
    # may be retried with the same child ID; a published child can be resumed.
    write_json(parent / 'superseded_by.json', marker)
    draft.rename(run)
    write_json(parent / 'superseded_by.json', {**marker, 'phase': 'REGISTERED'})


def recover_registration(parent, run):
    marker_path = parent / 'superseded_by.json'
    marker = read_json(marker_path) if marker_path.exists() else None
    if marker and (marker['run_id'] != run.name or marker['phase'] != 'PREPARING'):
        raise ValueError('Parent already has a registered continuation; use its ordinary resume command.')
    if not run.exists():
        return False
    if not marker or read_json(run / 'controller_status.json')['status'] != 'PREPARED':
        raise ValueError('Continuation already exists; inspect status and use ordinary resume.')
    imported = read_json(run / 'provenance.json')['continuation']
    if (imported['parent_run_id'] != parent.name
            or imported['parent_checkpoint_sha256'] != marker['parent_checkpoint_sha256']
            or sha256(run / 'model-last.pt') != imported['imported_checkpoint_sha256']):
        raise ValueError('Incomplete registration has inconsistent provenance.')
    write_json(marker_path, {**marker, 'phase': 'REGISTERED'})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--parent-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    root = args.project.resolve()
    os.chdir(root)
    parent, run = run_path(root, args.parent_run), run_path(root, args.run_id)
    config_path = Path('config/experiments/casia2_sel3_all8.yaml')
    with ExitStack() as leases:
        for handle in acquire_resources(root, args.run_id, args.gpu):
            leases.enter_context(handle)
        leases.enter_context(acquire_file(root / 'runtime/locks' / f'run-{args.parent_run}.lock'))
        if recover_registration(parent, run):
            leases.close()
            args.command = 'resume'
            start(args)
            return
        state, identity, original = [read_json(parent / name) for name in (
            'controller_status.json', 'controller_identity.json', 'provenance.json')]
        training = read_json(parent / 'training_status.json')
        if (state['status'] not in ('INTERRUPTED', 'FAILED') or state.get('child_pid') is not None
                or proc_identity(identity['pid']) == identity['start_ticks']
                or proc_identity(training['pid']) is not None or original['gpu'] != args.gpu):
            raise SystemExit('Stop the matching parent controller and all its children before continuation.')
        validate_environment(root, root)
        gpu = validate_gpu(args.gpu)
        if output(['git', 'status', '--porcelain', '--untracked-files=no'], root):
            raise SystemExit('Commit changes before registering continuation.')
        for name, expected in original['source_files_sha256'].items():
            if sha256(parent / 'source' / name) != expected:
                raise ValueError(f'Parent frozen source changed: {name}')
        config = _load_config(root / config_path)
        spec, snapshot, receipt = validate_casia_config(config, root)
        if receipt['manifest_sha256'] != original['dataset_manifest_sha256']:
            raise ValueError('The full train/All8 data snapshot changed.')
        weights_hash = sha256(root / 'pretrained_weights/pvt_v2_b2.pth')
        if weights_hash != original['pretrained_sha256']:
            raise ValueError('Initialization resource changed.')
        checkpoint_path = parent / 'model-last.pt'
        parent_hash = sha256(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        migrated = migrate_checkpoint(payload, OmegaConf.load(parent / 'resolved_config.yaml'), config,
                                      [entry['name'] for entry in spec['tests']])
        if sha256(checkpoint_path) != parent_hash:
            raise ValueError('Parent checkpoint changed while importing.')
        commit = output(['git', 'rev-parse', 'HEAD'], root)
        continuation = {'kind': 'USER_AUTHORIZED_SELECTION_CHANGE', 'registered_at': now(),
                        'parent_run_id': args.parent_run, 'parent_commit': original['git_commit'],
                        'parent_checkpoint': str(checkpoint_path), 'parent_checkpoint_sha256': parent_hash,
                        'parent_controller_stop': state, 'parent_epoch': payload['epoch'],
                        'next_epoch': payload['next_epoch'], 'global_step': payload['global_step'],
                        'preserved': ['model', 'opt', 'scheduler', 'scaler', 'rng', 'epoch', 'next_epoch', 'global_step'],
                        'best_reset': True, 'previous_selection_datasets': [x['name'] for x in spec['tests']],
                        'selection_datasets': list(config.test_dataset.Mix.params.datasets),
                        'selection_count': 1664, 'total_training_epochs': 100}
        with tempfile.TemporaryDirectory(prefix=args.run_id + '.preparing-', dir=run.parent) as temporary:
            draft = Path(temporary)
            hashes = freeze_source(root, draft / 'source', commit)
            atomic_checkpoint(draft / 'model-last.pt', migrated)
            continuation['imported_checkpoint_sha256'] = sha256(draft / 'model-last.pt')
            provenance = {**original, 'run_id': args.run_id, 'created_at': now(), 'git_commit': commit,
                          'protocol_id': config.protocol_id, 'source': str(run / 'source'),
                          'python': sys.executable, 'gpu_snapshot': gpu, 'config_path': config_path.as_posix(),
                          'config_sha256': sha256(draft / 'source' / config_path), 'source_files_sha256': hashes,
                          'dataset': receipt, 'dataset_manifest_path': (snapshot.relative_to(root) / 'manifest.csv').as_posix(),
                          'checkpoint_policy': 'best pooled three-set MAE after continuation; final report on All8',
                          'continuation': continuation}
            write_json(draft / 'provenance.json', provenance)
            write_json(draft / 'continuation.json', continuation)
            write_json(draft / 'controller_status.json', {'status': 'PREPARED', 'updated_at': now()})
            (draft / 'environment.freeze.txt').write_text(output([sys.executable, '-m', 'pip', 'freeze', '--all']) + '\n')
            publish_continuation(draft, run, parent, parent_hash)
    # Ordinary resume now accepts the imported contract. If startup fails, this
    # fully registered run remains recoverable without repeating the import.
    args.command = 'resume'
    start(args)


if __name__ == '__main__':
    main()
