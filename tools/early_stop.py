"""Bind an explicitly ended CASIA2 run to its existing best checkpoint."""
from __future__ import annotations

import json

from tools.manage_experiment import now, proc_identity, read_json, sha256

PROTOCOL = 'CASIA2-SEL3-ALL8-EARLYSTOP-V1'
BOUND_FILES = ('provenance.json', 'resolved_config.yaml', 'controller_identity.json',
               'controller_status.json', 'training_status.json', 'metrics.jsonl',
               'model-last.pt', 'model-best.pt')


def require_stopped(run):
    state = read_json(run / 'controller_status.json')
    identity = read_json(run / 'controller_identity.json')
    training = read_json(run / 'training_status.json')
    if (state['status'] != 'INTERRUPTED' or state.get('cleanup_error') is not None
            or state.get('child_pid') is not None
            or proc_identity(identity['pid']) == identity['start_ticks']
            or proc_identity(training['pid']) is not None):
        raise ValueError('Early-stop evaluation requires the training controller and children to have exited cleanly.')
    return training


def make_receipt(run):
    # Called while holding this run's GPU/run locks, before any All8 inference.
    import torch

    training = require_stopped(run)
    provenance = read_json(run / 'provenance.json')
    if provenance['protocol_id'] != 'CASIA2-SEL3-ALL8-V1' or provenance['gpu'] != 1:
        raise ValueError('This early endpoint is authorized only for the GPU 1 three-set continuation.')
    hashes = {name: sha256(run / name) for name in BOUND_FILES}
    checkpoint = torch.load(run / 'model-last.pt', map_location='cpu', weights_only=False)
    if (checkpoint.get('checkpoint_format') != 2
            or checkpoint.get('selection') != 'EPOCH_BOUNDARY_RECOVERY'
            or checkpoint['next_epoch'] != checkpoint['epoch'] + 1
            or not 3 < checkpoint['next_epoch'] < 100
            or checkpoint['best_epoch'] != training['best_epoch']
            or checkpoint['best_mae'] != training['best_mae']):
        raise ValueError('Last complete checkpoint does not match the stopped training history.')
    receipt = {
        'protocol_id': PROTOCOL, 'status': 'EARLY_STOPPED_BY_USER', 'registered_at': now(),
        'authorization': 'User request on 2026-09-15: end GPU 1 training early and test all eight datasets.',
        'run_id': run.name, 'training_commit': provenance['git_commit'], 'gpu': 1,
        'planned_epochs': 100, 'completed_epochs': checkpoint['next_epoch'],
        'last_complete_epoch': checkpoint['epoch'], 'last_complete_global_step': checkpoint['global_step'],
        'interrupted_epoch': training['epoch'], 'observed_global_step': training['global_step'],
        'best_epoch': checkpoint['best_epoch'], 'best_mae': checkpoint['best_mae'],
        'checkpoint': str(run / 'model-best.pt'), 'checkpoint_sha256': hashes['model-best.pt'],
        'files_sha256': hashes,
        'note': 'The incomplete epoch is excluded. Training was not completed to epoch 99; no final99 result exists for this continuation.',
    }
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if ([row['epoch'] for row in records] != list(range(receipt['completed_epochs']))
            or records[-1]['test_selected_best_epoch'] != receipt['best_epoch']
            or records[-1]['test_selected_best_mae'] != receipt['best_mae']):
        raise ValueError('Completed metrics history differs from the retained epoch boundary.')
    del checkpoint
    best = torch.load(run / 'model-best.pt', map_location='cpu', weights_only=False)
    if (best['epoch'] != receipt['best_epoch'] or best['best_epoch'] != receipt['best_epoch']
            or best['best_mae'] != receipt['best_mae']):
        raise ValueError('Existing best checkpoint differs from the last complete checkpoint selection.')
    return receipt


def validate_receipt(run, expected_hash):
    path = run / 'early_stop.json'
    if sha256(path) != expected_hash:
        raise ValueError('Early-stop receipt changed after registration.')
    receipt = read_json(path)
    if (receipt['protocol_id'] != PROTOCOL or receipt['status'] != 'EARLY_STOPPED_BY_USER'
            or receipt['run_id'] != run.name or receipt['gpu'] != 1
            or receipt['training_commit'] != read_json(run / 'provenance.json')['git_commit']
            or receipt['completed_epochs'] != receipt['last_complete_epoch'] + 1
            or not 3 <= receipt['best_epoch'] < receipt['completed_epochs'] < 100
            or set(receipt['files_sha256']) != set(BOUND_FILES)
            or receipt['checkpoint'] != str(run / 'model-best.pt')
            or receipt['checkpoint_sha256'] != receipt['files_sha256']['model-best.pt']):
        raise ValueError('Invalid early-stop endpoint receipt.')
    require_stopped(run)
    for name, digest in receipt['files_sha256'].items():
        if sha256(run / name) != digest:
            raise ValueError(f'Stopped training evidence changed: {name}')
    return receipt
