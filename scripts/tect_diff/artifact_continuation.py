"""Reuse completed healthy dependencies after the registered MAIN-only repair.

The invalid partial main phase is retained in the parent and never resumed.
Reference/calibration files stay at their original, immutable artifact paths.
"""
import shutil

from scripts.tect_diff.common import atomic_json, json_hash, read_json, sha256, timestamp
from tools.resource_locks import acquire_file


def import_completed_artifacts(run_dir, parent_dir):
    import torch
    from scripts.tect_diff.controller import read_run, status, verify_source, validate_receipt
    from scripts.tect_diff.data import load_bundle
    from model.tect_diff.evidence import FixedTrajectoryEvidence

    run, root, config, provenance = read_run(run_dir)
    parent, parent_root, parent_config, parent_provenance = read_run(parent_dir)
    if parent == run or parent_root != root or json_hash(config) != json_hash(parent_config):
        raise ValueError('Artifact reuse requires distinct same-project runs with identical configuration')
    if parent.name != 'TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C':
        raise ValueError('Only the registered AMP/DDP repair parent is authorized')
    names = ('data_bundle.json', 'reference_receipt.json', 'calibration_receipt.json',
             'reference_health.json', 'reference_final_health.json', 'reference_metrics.jsonl')
    if any((run / name).exists() for name in (*names, 'artifact_continuation.json', 'last.pth')):
        raise ValueError('Never overwrite a dependency import or a main checkpoint')
    with acquire_file(root / 'runtime/locks' / f'tect-registration-{parent.name}.lock'):
        state = status(parent)
        if (state['controller_alive'] or state['worker_alive']
                or state['status'] not in ('INTERRUPTED', 'FAILED') or state['stage'] != 'MAIN'):
            raise ValueError('Parent must be stopped in its incomplete MAIN phase')
        if not read_json(parent / 'operational_hold.json').get('active'):
            raise ValueError('Invalid parent must remain held')
        if any((parent / name).exists() for name in ('last.pth', 'best.pth', 'final.pth', 'epoch_summary.json', 'main_receipt.json')):
            raise ValueError('This repair only restarts a main phase with no completed epoch/checkpoint')
        verify_source(parent, parent_provenance)
        verify_source(run, provenance)
        allowed = {'model/tect_diff/diffusion.py', 'model/tect_diff/amp_context.py',
                   'scripts/tect_diff/worker.py', 'scripts/tect_diff/diagnostics.py',
                   'scripts/tect_diff/controller.py', 'scripts/tect_diff/report.py',
                   'scripts/tect_diff/gradient_safety.py', 'scripts/tect_diff/artifact_continuation.py',
                   'scripts/tect_diff/test_main_gradient_safety.py',
                   'scripts/tect_diff/test_artifact_continuation.py'}
        before, after = parent_provenance['source_hashes'], provenance['source_hashes']
        changed = [p for p in sorted(set(before) | set(after)) if p.endswith('.py') and before.get(p) != after.get(p)]
        if set(changed) - allowed:
            raise ValueError('Unaudited source change: ' + str(set(changed) - allowed))
        for stage in ('reference', 'calibration'):
            if not validate_receipt(parent, stage, parent_provenance):
                raise ValueError('Missing completed ' + stage)
        bundle = load_bundle(parent / 'data_bundle.json')
        ref = read_json(parent / 'reference_receipt.json')
        cal = read_json(parent / 'calibration_receipt.json')
        artifact = torch.load(cal['artifact_path'], map_location='cpu', weights_only=True)
        FixedTrajectoryEvidence(artifact).assert_frozen()
        metadata = artifact['metadata']
        if (metadata['reference_sha256'] != ref['sha256']
                or metadata['training_manifest_sha256'] != bundle['manifest_hashes']['train']
                or metadata['fit_manifest_sha256'] != bundle['manifest_hashes']['calibration']
                or artifact['artifact_hash'] != cal['artifact_hash']
                or artifact['fit_receipt'] != cal['fit_receipt']):
            raise ValueError('Calibration/reference/training manifest binding differs')
        hashes = {}
        for name in names:
            source, destination = parent / name, run / name
            if source.is_symlink() or not source.is_file():
                raise ValueError('Missing or linked parent receipt: ' + name)
            hashes[name] = sha256(source)
            temporary = destination.with_suffix(destination.suffix + '.importing')
            shutil.copyfile(source, temporary)
            if sha256(temporary) != hashes[name] or sha256(source) != hashes[name]:
                raise ValueError('Dependency receipt copy did not preserve bytes')
            temporary.replace(destination)
        for stage in ('reference', 'calibration'):
            validate_receipt(run, stage, provenance)
        receipt = {'at': timestamp(), 'status': 'COMPLETED', 'parent_run': parent.name,
                   'mode': 'reuse_completed_reference_calibration_restart_invalid_main',
                   'parent_source_commit': parent_provenance['commit'], 'source_commit': provenance['commit'],
                   'config_hash': provenance['config_hash'], 'manifest_hashes': bundle['manifest_hashes'],
                   'reference_sha256': ref['sha256'], 'calibration_sha256': cal['sha256'],
                   'copied_receipt_sha256': hashes, 'changed_python_files': changed,
                   'parent_partial_main_valid': False, 'parent_main_checkpoint_reused': False,
                   'main_start_epoch': 0, 'main_start_optimizer_step': 0, 'main_seed': config['seed'],
                   'reference_epochs_retrained': 0, 'calibration_refitted': False,
                   'selection_protocol': 'test_selected'}
        atomic_json(run / 'artifact_continuation.json', receipt)
        hold = read_json(parent / 'operational_hold.json')
        atomic_json(parent / 'operational_hold.json', {**hold, 'continued_by': run.name})
        atomic_json(run / 'operational_hold.json', {'active': False, 'at': timestamp(),
                    'reason': 'Completed dependencies verified; corrected MAIN starts fresh; see artifact_continuation.json'})
        return receipt
