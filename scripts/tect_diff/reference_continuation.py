"""Explicit reference-only state transfer for the audited scheduling optimization.

This records a source-version boundary. It never claims identical source resume,
changes scientific configuration, reselects a checkpoint, or modifies parent weights.
"""
from __future__ import annotations

from scripts.tect_diff.common import (atomic_json, atomic_torch_save, append_json,
                                      json_hash, read_json, sha256, timestamp)
from tools.resource_locks import acquire_file


def same_payload(left, right):
    """Exact comparison including optimizer scalar tensors and all-rank RNG."""
    import numpy as np
    import torch
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_payload(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_payload(a, b) for a, b in zip(left, right))
    return left == right


def import_reference_state(run_dir, parent_dir):
    import torch
    from scripts.tect_diff.controller import read_run, status, verify_source, check_operational_hold
    from scripts.tect_diff.reference_health import require_healthy_reference
    run, root, config, provenance = read_run(run_dir)
    parent, parent_root, parent_config, parent_provenance = read_run(parent_dir)
    if parent == run or parent_root != root or json_hash(parent_config) != json_hash(config):
        raise ValueError('Continuation requires distinct same-project runs and identical configuration')
    if config['reference'].get('architecture_version') != 'tect-reference-pixel-unet-v2-stable-paths':
        raise ValueError('Only the repaired reference protocol is eligible')
    if (run / 'reference_last.pth').exists() or (run / 'reference_continuation.json').exists():
        raise ValueError('Never overwrite an existing reference import')
    with acquire_file(root / 'runtime/locks' / f'tect-registration-{parent.name}.lock'):
        check_operational_hold(parent)
        state = status(parent)
        if state['controller_alive'] or state['worker_alive'] or state['status'] not in ('INTERRUPTED', 'FAILED'):
            raise ValueError('Parent controller and all owned workers must have stopped')
        if any((parent / name).exists() for name in ('reference_receipt.json', 'calibration_receipt.json', 'last.pth')):
            raise ValueError('This importer is restricted to unfinished reference training')
        verify_source(parent, parent_provenance)
        allowed = {'scripts/tect_diff/worker.py', 'scripts/tect_diff/controller.py',
                   'scripts/tect_diff/reference_continuation.py', 'scripts/tect_diff/report.py'}
        before, after = parent_provenance['source_hashes'], provenance['source_hashes']
        changed_code = []
        for path in sorted(set(before) | set(after)):
            if path.endswith('.py') and before.get(path) != after.get(path):
                if path not in allowed:
                    raise ValueError(f'Unaudited training/model source change: {path}')
                changed_code.append(path)
        checkpoint_path = parent / 'reference_last.pth'
        parent_hash = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        bundle = read_json(parent / 'data_bundle.json')
        if (checkpoint['source_commit'] != parent_provenance['commit']
                or checkpoint['config_hash'] != provenance['config_hash']
                or json_hash(checkpoint['config']) != provenance['config_hash']
                or checkpoint['manifest_hashes'] != bundle['manifest_hashes']
                or checkpoint['next_epoch'] != checkpoint['epoch'] + 1
                or not checkpoint['epoch_complete'] or not checkpoint['evaluation_complete']
                or not 0 < checkpoint['next_epoch'] < config['reference']['epochs']
                or len(checkpoint['rng_by_rank']) != config['world_size']
                or checkpoint['sampler']['epoch'] != checkpoint['epoch']
                or checkpoint['sampler']['world_size'] != config['world_size']
                or checkpoint['sampler']['position'] != 'epoch_boundary'):
            raise ValueError('Incomplete or incompatible reference checkpoint')
        history = checkpoint['history']
        if ([row['epoch'] for row in history] != list(range(checkpoint['next_epoch']))
                or history[-1]['optimizer_step'] != checkpoint['optimizer_step']):
            raise ValueError('Reference checkpoint history is not a complete prefix')
        require_healthy_reference(history[-1])
        for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng_by_rank', 'sampler', 'initial_parameter_hash'):
            if key not in checkpoint:
                raise ValueError(f'Missing full training state: {key}')
        transferred = dict(checkpoint, source_commit=provenance['commit'])
        destination = run / 'reference_last.pth'
        atomic_torch_save(destination, transferred)
        restored = torch.load(destination, map_location='cpu')
        if not same_payload(transferred, restored):
            raise ValueError('Checkpoint serialization did not preserve the full transferred state')
        if not same_payload({k:v for k,v in checkpoint.items() if k != 'source_commit'},
                            {k:v for k,v in restored.items() if k != 'source_commit'}):
            raise ValueError('State outside source provenance changed')
        if sha256(checkpoint_path) != parent_hash:
            raise ValueError('Parent checkpoint changed during import')
        atomic_json(run / 'data_bundle.json', bundle)
        for row in history:
            append_json(run / 'reference_metrics.jsonl', row)
        atomic_json(run / 'reference_health.json', history[-1])
        receipt = {'at': timestamp(), 'status': 'COMPLETED', 'parent_run': parent.name,
                   'parent_checkpoint_sha256': parent_hash, 'child_checkpoint_sha256': sha256(destination),
                   'parent_source_commit': parent_provenance['commit'], 'source_commit': provenance['commit'],
                   'config_hash': provenance['config_hash'], 'manifest_hashes': bundle['manifest_hashes'],
                   'epoch': checkpoint['epoch'], 'next_epoch': checkpoint['next_epoch'],
                   'optimizer_step': checkpoint['optimizer_step'], 'changed_python_files': changed_code,
                   'changed_checkpoint_fields': ['source_commit'], 'all_other_state_exactly_equal': True,
                   'selection_protocol': 'test_selected', 'mode': 'state_preserving_source_version_continuation'}
        atomic_json(run / 'reference_continuation.json', receipt)
        atomic_json(parent / 'operational_hold.json', {
            'active': True, 'run_id': parent.name, 'continued_by': run.name, 'at': timestamp(),
            'reason': 'Reference checkpoint state transferred to the audited performance continuation; never resume both runs',
            'retained_checkpoint_sha256': parent_hash})
        atomic_json(run / 'operational_hold.json', {'active': False, 'at': timestamp(),
                    'reason': 'Reference state import verified; see reference_continuation.json'})
        return receipt
