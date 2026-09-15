"""Registered microbatch6/global12 fresh MAIN restart with fitting reuse.

Only dependency receipt protocol/config identities are rebound. Frozen artifact
bytes and their original fitted metadata remain unchanged and auditable.
"""
import ast
import copy
import math
from pathlib import Path
import shutil

from scripts.tect_diff.common import atomic_json, inside, json_hash, read_json, sha256, timestamp
from tools.resource_locks import acquire_file

PARENT_RUN = 'TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D'
CHILD_RUN = 'TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E'
PARENT_COMMIT = '84fcf47e5721317d99fe186feec08d365fc3f72c'
PARENT_CONFIG_HASH = '0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88'
REFERENCE_HASH = '800f50c392a275fde3ec249c6f52275c6db79813e6f8b7e8b534f9ddb52a523f'
CALIBRATION_HASH = 'bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c'
PROTOCOL = 'TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6'
EVALUATION_NUMERICAL_POLICY = 'BF16_BATCH_EXECUTION_V1'
BF16_PROBABILITY_BUDGET = 0.0078125
FP32_CONTROL_BUDGET = 1e-5
MODEL_FILE = 'model/tect_diff/evidence.py'
WORKER_FILE = 'scripts/tect_diff/worker.py'
EVALUATION_FILE = 'scripts/tect_diff/evaluation_pipeline.py'
COPY_FILES = ('data_bundle.json', 'reference_receipt.json', 'calibration_receipt.json',
              'reference_health.json', 'reference_final_health.json', 'reference_metrics.jsonl')
REBOUND_FILES = {'reference_receipt.json', 'calibration_receipt.json'}
ALLOWED_CODE = {MODEL_FILE, WORKER_FILE, EVALUATION_FILE,
                'scripts/tect_diff/controller.py', 'scripts/tect_diff/report.py',
                'scripts/tect_diff/batch_restart.py', 'scripts/tect_diff/test_batch_restart.py',
                'scripts/tect_diff/test_evidence_geometry_cache.py', 'scripts/tect_diff/test_evaluation_pipeline.py'}


def _regular_file(root, path):
    path = Path(path).absolute()
    resolved = inside(root, path)
    if path != resolved or path.is_symlink() or not path.is_file():
        raise ValueError('Missing, linked or escaping batch-restart file: ' + str(path))
    return path


def _validate_config(config, parent_config):
    evaluation = config['evaluation']
    selected_evaluation = {'micro_batch': 8, 'metric_workers': 4, 'metric_queue_limit': 8}
    if any(type(evaluation.get(key)) is not int or evaluation[key] != value for key, value in selected_evaluation.items()):
        raise ValueError('Unregistered evaluation configuration: selected batch8, metric_workers4, queue_limit8 only')
    expected = copy.deepcopy(parent_config)
    expected['training'].update(micro_batch=6, accumulation_steps=1, global_batch=12)
    expected['protocol_id'] = PROTOCOL
    expected['evaluation'].update({key: evaluation[key] for key in ('micro_batch', 'metric_workers', 'metric_queue_limit')})
    if json_hash(parent_config) != PARENT_CONFIG_HASH or json_hash(config) != json_hash(expected):
        raise ValueError('Unregistered configuration change outside microbatch6/accumulation1/global12, selected evaluation scheduling and MEMB6 identity')
    delta = {'training.micro_batch': {'old': 2, 'new': 6},
             'training.accumulation_steps': {'old': 2, 'new': 1},
             'training.global_batch': {'old': 8, 'new': 12},
             'protocol_id': {'old': parent_config['protocol_id'], 'new': PROTOCOL}}
    for key in ('micro_batch', 'metric_workers', 'metric_queue_limit'):
        old, new = parent_config['evaluation'].get(key), evaluation[key]
        if old != new:
            delta['evaluation.' + key] = {'old': old, 'new': new}
    return delta


def _validate_benchmark(path, config, provenance, root):
    validation = read_json(path)
    speedup = validation.get('measured_speedup')
    speedup_vs_micro4 = validation.get('measured_speedup_vs_micro4')
    if (validation.get('passed') is not True or validation.get('parent_run') != PARENT_RUN
            or validation.get('parent_source_commit') != PARENT_COMMIT
            or validation.get('parent_config_hash') != PARENT_CONFIG_HASH
            or validation.get('config_hash') != json_hash(config)
            or validation.get('candidate_model_sha256') != provenance['source_hashes'].get(MODEL_FILE)
            # Original/geometry-only arms monkeypatch this same file in diagnostics;
            # its deployed implementation is specifically geometry_full_queries.
            or validation.get('accepted_geometry_variant') != 'geometry_full_queries'
            or validation.get('reference_sha256') != REFERENCE_HASH
            or validation.get('calibration_sha256') != CALIBRATION_HASH
            or validation.get('full_gradient_coverage') is not True
            or validation.get('all_rank_gradients_synchronized') is not True
            or validation.get('micro6_no_oom') is not True or validation.get('probe_finite') is not True
            or validation.get('micro_batch') != 6 or validation.get('accumulation_steps') != 1
            or validation.get('global_batch') != 12
            or type(speedup_vs_micro4) not in (int, float) or not math.isfinite(speedup_vs_micro4) or speedup_vs_micro4 < 1.05
            or type(speedup) not in (int, float) or not math.isfinite(speedup) or speedup < 1.05):
        raise ValueError('Batch-restart validation failed or is not bound to the registered measured candidate/configuration')
    evaluation = validation.get('evaluation_validation') or {}
    difference = evaluation.get('max_probability_abs_error')
    fp32_difference = evaluation.get('fp32_batch_control_max_abs')
    metric_differences = [evaluation.get(key) for key in ('max_pixel_f1_abs_difference', 'max_iou_abs_difference',
                                                        'max_boundary_f1_abs_difference', 'max_mae_abs_difference')]
    if (validation.get('candidate_worker_sha256') != provenance['source_hashes'].get(WORKER_FILE)
            or validation.get('candidate_evaluation_pipeline_sha256') != provenance['source_hashes'].get(EVALUATION_FILE)
            or evaluation.get('passed') is not True or evaluation.get('finite') is not True
            or evaluation.get('full_id_coverage') is not True
            or evaluation.get('numerical_policy') != EVALUATION_NUMERICAL_POLICY
            or evaluation.get('fp32_batch_control_passed') is not True
            or type(fp32_difference) not in (int, float) or not math.isfinite(fp32_difference)
            or not 0 <= fp32_difference <= FP32_CONTROL_BUDGET
            or evaluation.get('model_rng_unchanged') is not True
            or evaluation.get('identical_array_cpu_metrics_exact') is not True
            # This is a predeclared empirical execution budget, not an analytical
            # end-to-end error bound or a claim of cross-batch prediction equality.
            or type(difference) not in (int, float) or not math.isfinite(difference)
            or not 0 <= difference <= BF16_PROBABILITY_BUDGET
            or type(evaluation.get('binary_disagreement_count')) is not int or evaluation['binary_disagreement_count'] < 0
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in metric_differences)
            or any(type(evaluation.get(key)) is not int or evaluation[key] != config['evaluation'][key]
                   for key in ('micro_batch', 'metric_workers', 'metric_queue_limit'))):
        raise ValueError('Batch-restart evaluation validation failed: source, selected settings, coverage or numerical compatibility differs')
    try:
        proof_path = _regular_file(Path(root)/'runtime/tect-resource-recheck-20260915',
                                   validation.get('evidence_forward_validation_path', ''))
        proof = read_json(proof_path)
        proof_hash = sha256(proof_path)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError('Evidence forward validation requires a real, unlinked task-runtime proof file') from error
    if (not isinstance(proof, dict) or proof != validation.get('evidence_forward_validation')
            or proof_hash != validation.get('evidence_forward_validation_sha256')
            or proof.get('status') != 'COMPLETED' or proof.get('scientific_result') is not False
            or proof.get('candidate_evidence_sha256') != provenance['source_hashes'].get(MODEL_FILE)
            or proof.get('forward_original_repeat_bitwise') is not True
            or proof.get('reference_calibration_rng_unchanged') is not True
            or proof.get('all_global_ids_exact') is not True
            or type(proof.get('global_count')) is not int or proof['global_count'] != 16
            or not isinstance(proof.get('variants'), dict)
            or not isinstance(proof['variants'].get('geometry_full_queries'), dict)
            or proof['variants']['geometry_full_queries'].get('all_forward_outputs_bitwise_equal') is not True):
        raise ValueError('Evidence forward validation does not prove the registered deployed source/state/16-image path')
    try:
        task_root = Path(root)/'runtime/tect-resource-recheck-20260915'
        training_path = _regular_file(task_root, validation.get('training_benchmark_path', ''))
        training = read_json(training_path)
        training_hash = sha256(training_path)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError('Micro6 training benchmark requires a real, unlinked task-runtime proof file') from error
    required_training = {
        'status': 'COMPLETED', 'passed': True, 'scientific_result': False, 'mode': 'micro6',
        'parent_run': PARENT_RUN, 'parent_source_commit': PARENT_COMMIT, 'parent_config_hash': PARENT_CONFIG_HASH,
        'candidate_evidence_sha256': provenance['source_hashes'].get(MODEL_FILE),
        'candidate_worker_sha256': provenance['source_hashes'].get(WORKER_FILE),
        'reference_sha256': REFERENCE_HASH, 'calibration_sha256': CALIBRATION_HASH,
        'accepted_geometry_variant': 'geometry_full_queries',
        'full_gradient_coverage': True, 'all_rank_gradients_synchronized': True,
        'final_parameter_and_optimizer_hashes_agree': True, 'no_oom': True, 'within_memory_cap': True,
        'probe_finite': True, 'no_weights_saved': True,
        'micro_batch': 6, 'accumulation_steps': 1, 'global_batch': 12,
        'learning_rate': config['training']['learning_rate'], 'learning_rate_scaling': 'none',
        'warmup_updates': 4, 'measured_updates': 32,
        'speedup_vs_micro4_global8': speedup_vs_micro4, 'speedup_vs_original_micro2_global8': speedup,
        'forward_validation': {'passed': True, 'source': str(proof_path), 'sha256': proof_hash}}
    if (training_path != task_root/'micro6/aggregate.json'
            or training_hash != validation.get('training_benchmark_sha256') or not isinstance(training, dict)
            or any(type(training.get(key)) is not type(value) or training[key] != value
                   for key, value in required_training.items())):
        raise ValueError('Micro6 training benchmark differs from the selected source/artifacts/6-1-12/LR/speedup/gradient proof')
    return validation


def _worker_outside_evaluate(path):
    """Permit only evaluate's body and its two explicitly relocated imports."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if (node.module == 'scripts.tect_diff.evaluation_pipeline'
                    and [(item.name, item.asname) for item in node.names] == [('EvaluationMetricPipeline', None)]):
                continue
            if node.module == 'scripts.tect_diff.metrics':
                node.names = [item for item in node.names if (item.name, item.asname) != ('per_image_metrics', None)]
        body.append(node)
    tree.body = body
    workers = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Worker']
    methods = ([node for node in workers[0].body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate']
               if len(workers) == 1 else [])
    if len(methods) != 1:
        raise ValueError('Worker source is missing its unique evaluate method')
    methods[0].body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def import_batch_dependencies(run_dir, parent_dir, validation_file):
    import torch
    from scripts.tect_diff.controller import read_run, status, verify_source, validate_receipt
    from scripts.tect_diff.data import load_bundle
    from model.tect_diff.evidence import FixedTrajectoryEvidence

    run, root, config, provenance = read_run(run_dir)
    parent, parent_root, parent_config, old = read_run(parent_dir)
    if parent == run or parent_root != root:
        raise ValueError('Batch restart requires distinct same-project runs')
    delta = _validate_config(config, parent_config)
    config_hash = json_hash(config)
    if (parent.name != PARENT_RUN or run.name != CHILD_RUN or old['commit'] != PARENT_COMMIT
            or old['config_hash'] != PARENT_CONFIG_HASH or provenance['config_hash'] != config_hash):
        raise ValueError('Only the registered D -> MEMB6 E restart and frozen parent source are authorized')
    if any((run/name).exists() for name in (*COPY_FILES, 'last.pth', 'best.pth', 'final.pth', 'epochs',
                                          'batch_restart.json', 'batch_validation.json', 'main_probe_receipt.json')):
        raise ValueError('Never overwrite an existing dependency import or MAIN state')
    if not read_json(run/'operational_hold.json').get('active'):
        raise ValueError('Child must remain held until dependency import completes')
    validation_path = _regular_file(root/'runtime', validation_file)
    validation_hash = sha256(validation_path)
    with acquire_file(root/'runtime/locks'/f'tect-registration-{parent.name}.lock'):
        state = status(parent)
        if (state['controller_alive'] or state['worker_alive'] or state['status'] not in ('INTERRUPTED', 'FAILED')
                or state['stage'] != 'MAIN'):
            raise ValueError('Batch parent controller and all workers must be stopped in MAIN')
        hold = read_json(parent/'operational_hold.json')
        if not hold.get('active') or hold.get('continued_by') not in (None, run.name):
            raise ValueError('Batch parent must remain held and unclaimed by another child')
        if any((parent/name).exists() for name in ('last.pth', 'best.pth', 'final.pth', 'epoch_summary.json', 'main_receipt.json')):
            raise ValueError('Registered restart parent must have no completed MAIN epoch/checkpoint')
        epochs = parent/'epochs'
        if epochs.is_symlink() or (epochs.exists() and any(path.is_symlink() or path.is_file() for path in epochs.rglob('*'))):
            raise ValueError('Existing evaluation outputs are outside the registered fresh restart')
        verify_source(parent, old)
        verify_source(run, provenance)
        before, after = old['source_hashes'], provenance['source_hashes']
        changed = [name for name in sorted(set(before) | set(after))
                   if name.endswith('.py') and before.get(name) != after.get(name)]
        if set(changed) - ALLOWED_CODE:
            raise ValueError('Unaudited source change: ' + str(set(changed) - ALLOWED_CODE))
        for name in (MODEL_FILE, WORKER_FILE, EVALUATION_FILE):
            candidate_path = _regular_file(run/'source', run/'source'/name)
            if sha256(candidate_path) != after.get(name):
                raise ValueError('Candidate source does not match the published snapshot: ' + name)
        parent_worker = _regular_file(parent/'source', parent/'source'/WORKER_FILE)
        if _worker_outside_evaluate(parent_worker) != _worker_outside_evaluate(run/'source'/WORKER_FILE):
            raise ValueError('Unaudited Worker change outside evaluate and its registered metric imports')
        validation = _validate_benchmark(validation_path, config, provenance, root)
        sources = {name: _regular_file(parent, parent/name) for name in COPY_FILES}
        parent_hashes = {name: sha256(path) for name, path in sources.items()}
        inherited_path = _regular_file(parent, parent/'artifact_continuation.json')
        inherited_hash = sha256(inherited_path)
        inherited = read_json(inherited_path)
        for stage in ('reference', 'calibration'):
            if not validate_receipt(parent, stage, old):
                raise ValueError('Missing completed fitting dependency: ' + stage)
        bundle = load_bundle(parent/'data_bundle.json')
        ref, cal = read_json(parent/'reference_receipt.json'), read_json(parent/'calibration_receipt.json')
        if ref.get('sha256') != REFERENCE_HASH or cal.get('sha256') != CALIBRATION_HASH:
            raise ValueError('Only the fixed registered artifact hashes are eligible for this restart')
        if any(item.get('config_hash') != PARENT_CONFIG_HASH or item.get('protocol_id') != parent_config['protocol_id']
               for item in (ref, cal)):
            raise ValueError('Original fitting receipt configuration binding differs')
        artifact_path = _regular_file(root, cal['artifact_path'])
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
        FixedTrajectoryEvidence(artifact).assert_frozen()
        metadata = artifact['metadata']
        if (sha256(artifact_path) != cal['sha256'] or metadata['reference_sha256'] != ref['sha256']
                or metadata['training_manifest_sha256'] != bundle['manifest_hashes']['train']
                or metadata['fit_manifest_sha256'] != bundle['manifest_hashes']['calibration']
                or artifact['artifact_hash'] != cal['artifact_hash'] or artifact['fit_receipt'] != cal['fit_receipt']
                or inherited.get('status') != 'COMPLETED' or inherited.get('config_hash') != PARENT_CONFIG_HASH
                or inherited.get('reference_sha256') != ref['sha256'] or inherited.get('calibration_sha256') != cal['sha256']):
            raise ValueError('Reference/calibration/original lineage binding differs')
        child_hashes = {}
        for name, source in sources.items():
            if name in REBOUND_FILES:
                original = read_json(source)
                value = {**original, 'config_hash': config_hash, 'protocol_id': PROTOCOL}
                atomic_json(run/name, value)
                if read_json(run/name) != value:
                    raise ValueError('Receipt identity rebinding did not preserve the dependency payload')
            else:
                temporary = run/(name + '.importing')
                shutil.copyfile(source, temporary)
                if sha256(temporary) != parent_hashes[name]:
                    raise ValueError('Dependency copy did not preserve bytes: ' + name)
                temporary.replace(run/name)
            child_hashes[name] = sha256(run/name)
        if (any(sha256(path) != parent_hashes[name] for name, path in sources.items())
                or sha256(validation_path) != validation_hash or sha256(inherited_path) != inherited_hash
                or sha256(validation['evidence_forward_validation_path']) != validation['evidence_forward_validation_sha256']
                or sha256(validation['training_benchmark_path']) != validation['training_benchmark_sha256']
                or sha256(artifact_path) != cal['sha256']):
            raise ValueError('Parent dependency, artifact or validation changed during import')
        for stage in ('reference', 'calibration'):
            if not validate_receipt(run, stage, provenance):
                raise ValueError('Rebound dependency receipt did not validate: ' + stage)
        atomic_json(run/'batch_validation.json', validation)
        receipt = {'at': timestamp(), 'status': 'COMPLETED', 'parent_run': parent.name,
                   'mode': 'authorized_microbatch_protocol_restart_fresh_main_reuse_fitting',
                   'parent_source_commit': old['commit'], 'source_commit': provenance['commit'],
                   'parent_config_hash': PARENT_CONFIG_HASH, 'config_hash': config_hash,
                   'parent_protocol_id': parent_config['protocol_id'], 'protocol_id': PROTOCOL,
                   'config_delta': delta, 'manifest_hashes': bundle['manifest_hashes'],
                   'reference_sha256': ref['sha256'], 'calibration_sha256': cal['sha256'],
                   'original_receipts': {'reference': ref, 'calibration': cal},
                   'parent_receipt_sha256': parent_hashes, 'child_receipt_sha256': child_hashes,
                   'rebound_receipt_fields': {name: ['config_hash', 'protocol_id'] for name in sorted(REBOUND_FILES)},
                   'artifact_bytes_and_metadata_unchanged': True,
                   'inherited_artifact_continuation': inherited, 'inherited_artifact_continuation_sha256': inherited_hash,
                   'validation_source_path': str(validation_path), 'validation_source_sha256': validation_hash,
                   'validation': validation, 'reference_epochs_retrained': 0, 'calibration_refitted': False,
                   'parent_main_state_reused': False, 'main_start_epoch': 0, 'main_start_optimizer_step': 0,
                   'main_seed': 42, 'main_initialization': 'fresh registered ImageNet weights',
                   'main_batch': {'micro_batch_per_rank': 6, 'accumulation_steps': 1, 'global_batch': 12},
                   'training_validation_scope': {'warmup_updates': 4, 'measured_updates': 32,
                                                 'full_gradient_coverage_all_updates': True,
                                                 'all_rank_gradients_synchronized': True,
                                                 'final_parameter_and_optimizer_hashes_agree': True,
                                                 'learning_rate_scaling': 'none'},
                   'evaluation_scheduling': {key: config['evaluation'][key]
                                             for key in ('micro_batch', 'metric_workers', 'metric_queue_limit')},
                   'evaluation_numerical_policy': {'policy_id': EVALUATION_NUMERICAL_POLICY,
                                                   'probability_budget': BF16_PROBABILITY_BUDGET,
                                                   'budget_basis': 'predeclared empirical guard, not analytical end-to-end bound',
                                                   'fp32_control_budget': FP32_CONTROL_BUDGET,
                                                   'difference_scope': 'Cross-batch binary and metric differences are disclosed, not required to be zero; CPU metrics on identical arrays remain exact'},
                   'selection_protocol': 'test_selected'}
        atomic_json(run/'batch_restart.json', receipt)
        atomic_json(parent/'operational_hold.json', {**hold, 'active': True, 'continued_by': run.name, 'at': timestamp()})
        atomic_json(run/'operational_hold.json', {'active': False, 'at': timestamp(),
                    'reason': 'Measured microbatch6/global12 restart registered; healthy fitting artifacts reused; MAIN starts fresh'})
        return receipt
