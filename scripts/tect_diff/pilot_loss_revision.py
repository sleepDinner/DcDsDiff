"""Strict, CPU-only provenance gate for the registered empty-target loss pilot."""
import hashlib
import json
from pathlib import Path

from scripts.tect_diff.common import atomic_json, json_hash, read_json, sha256, timestamp


RUN_ID = 'TECT-PILOT-CASIA2-GN8-EMPTYBCE-R512-S42-20260916-D'
PROTOCOL = 'TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-N8192-EMPTYBCE-V1'
MASK_LOSS_POLICY = 'empty_target_bce_v1'
PARENT_RUN_ID = 'TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C'
PARENT_PROTOCOL = 'TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-N8192-V1'
PARENT_CONFIG_RELATIVE = 'configs/tect_diff/pilot_casia2_gn8_data2_n8192_r512_s42.json'
PARENT_CONFIG_HASH = 'c42769f4f4fedc18f5ae21ae4c6525bb551a416e2e5ffd98ed067bc6877bd702'
PARENT_SOURCE_COMMIT = '02691917da17016f64f26ef820d5ccfa175e18ab'
PARENT_TERMINAL_HASH = '7184fcfb257601f7fdaaef9e229b75a5c9e6e3ab38dc15b2fe9813f3db74a27b'
PARENT_BUNDLE_HASH = 'ab8bec8091f387b1fc0edbd12fd918ceb6ef9f5a0681e2aa22158c3cb7bbc03e'
ROLES = ('train', 'preflight_train', 'reference', 'calibration', 'authentic_probe',
         'test_Casiav1', 'test_Columbia', 'test_full_Casiav1', 'test_full_Columbia')
DIFFUSION_RELATIVE = 'model/tect_diff/diffusion.py'
OLD_IMPORT = 'from model.loss import structure_loss'
NEW_IMPORT = 'from model.tect_diff.mask_loss import tect_mask_loss'
OLD_CALL = '            mask_loss = structure_loss(controlled.float(), gt.float())'
NEW_CALL = "            mask_loss = tect_mask_loss(controlled.float(), gt.float(), self.config['training'].get('mask_loss', 'structure_v1'))"


def _regular(path, root=None):
    path = Path(path)
    resolved = path.resolve(strict=True)
    if (not path.is_absolute() or path.is_symlink() or resolved != path or not path.is_file()
            or (root is not None and Path(root).resolve(strict=True) not in resolved.parents)):
        raise ValueError('Loss revision requires an unlinked, same-project regular file: ' + str(path))
    return path


def _manifest_hash(value):
    # This is the existing data manifest canonicalization, including Unicode.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def validate_loss_config(config, source_root):
    """Only protocol_id and training.mask_loss may differ from canonical C."""
    source_root = Path(source_root).resolve(strict=True)
    canonical = read_json(_regular(source_root / PARENT_CONFIG_RELATIVE, source_root))
    if json_hash(canonical) != PARENT_CONFIG_HASH:
        raise ValueError('The registered canonical C configuration changed')
    expected = {**canonical, 'protocol_id': PROTOCOL,
                'training': {**canonical['training'], 'mask_loss': MASK_LOSS_POLICY}}
    if json_hash(config) != json_hash(expected):
        raise ValueError('Pilot D permits only its protocol ID and registered empty-target loss change over C')


def validate_diffusion_extension(original_path, revised_path):
    """Reverse exactly the import and loss call; protect all remaining source."""
    original_path, revised_path = _regular(original_path), _regular(revised_path)
    original = original_path.read_text(encoding='utf-8')
    revised = revised_path.read_text(encoding='utf-8')
    if (original.count(OLD_IMPORT) != 1 or original.count(OLD_CALL) != 1
            or revised.count(NEW_IMPORT) != 1 or revised.count(NEW_CALL) != 1
            or revised.replace(NEW_IMPORT, OLD_IMPORT).replace(NEW_CALL, OLD_CALL) != original):
        raise ValueError('Loss revision diffusion differs beyond its exact import and forward loss call')
    return {'status': 'PASSED', 'original_sha256': sha256(original_path),
            'revised_sha256': sha256(revised_path), 'allowed_changes': ['loss_import', 'forward_mask_loss_call']}


def _read_manifests(summary, root):
    if (summary.get('audit_version') != 'TECT-CASIA2-PILOT-DATA-V2-QUARANTINE'
            or summary.get('current_audit_status') != 'COMPLETED'
            or summary.get('current_blocking_audit_error_count') != 0
            or set(summary.get('manifest_paths', {})) != set(ROLES)
            or set(summary.get('manifest_hashes', {})) != set(ROLES)):
        raise ValueError('Loss revision requires the completed nine-role pilot data audit')
    loaded = {}
    for role in ROLES:
        rows = read_json(_regular(summary['manifest_paths'][role], root))
        if type(rows) is not list or not rows or _manifest_hash(rows) != summary['manifest_hashes'][role]:
            raise ValueError('Loss revision frozen manifest changed: ' + role)
        loaded[role] = rows
    return loaded


def _validate_data(bundle, parent_summary, root):
    summary = bundle['summary']
    # Identical data preparation can store the same rows at a different regular
    # path. Every other frozen audit field, including quarantines, stays equal.
    if ({key: value for key, value in summary.items() if key != 'manifest_paths'}
            != {key: value for key, value in parent_summary.items() if key != 'manifest_paths'}):
        raise ValueError('Loss revision changed the frozen C data contract or role hashes')
    parent = _read_manifests(parent_summary, root)
    child = _read_manifests(summary, root)
    actual = {key: bundle[key] for key in ('train', 'preflight_train', 'reference', 'calibration', 'authentic_probe')}
    actual.update({'test_' + name: bundle['tests'][name] for name in ('Casiav1', 'Columbia')})
    actual.update({'test_full_' + name: bundle['tests_full'][name] for name in ('Casiav1', 'Columbia')})
    if (parent != child or actual != child or bundle['manifest_hashes'] != summary['manifest_hashes']
            or bundle['manifest_paths'] != summary['manifest_paths']
            or bundle['reference_source_mode'] != summary['reference_source_mode']):
        raise ValueError('Loss revision requires identical loaded C rows and all nine frozen role hashes')
    rows = child['train']
    authentic = sum(row['authentic_declared'] is True for row in rows)
    sizes = summary['sample_sizes']
    if (len(rows) != 8192 or authentic != 4096 or len({row['id'] for row in rows}) != 8192
            or len({row['rgb_pixels_sha256'] for row in rows}) != 8192
            or summary['train_count'] != 8192 or summary['train_authentic_count'] != 4096
            or sizes['train_authentic'] != 4096 or sizes['train_tampered'] != 4096
            or summary['seed'] != 42 or summary['quarantined_pair_count'] != 2):
        raise ValueError('Loss revision requires the same balanced 8192-image seed42 C training set')
    held = [*child['authentic_probe'], *child['test_full_Casiav1'], *child['test_full_Columbia']]
    held_ids = {row['id'] for row in held}
    held_rgb = {row['rgb_pixels_sha256'] for row in held}
    held_sources = {source for row in held for source in row['source_ids']}
    quarantine_ids = {row['id'] for row in summary['quarantined_pairs']}
    for row in rows:
        if (type(row['authentic_declared']) is not bool or row['empty_mask'] != row['authentic_declared']
                or row['id'] in quarantine_ids or row['id'] in held_ids
                or row['rgb_pixels_sha256'] in held_rgb or held_sources.intersection(row['source_ids'])):
            raise ValueError('Loss revision inherited training semantics/isolation changed: ' + row['id'])
    return {'train_count': 8192, 'train_authentic_count': 4096, 'train_tampered_count': 4096,
            'all_parent_rows_identical': True, 'unchanged_manifest_hashes': dict(summary['manifest_hashes']),
            'test_probe_source_and_rgb_disjoint': True, 'quarantined_pair_count': 2,
            'parent_manifest_id': parent_summary['manifest_id'], 'manifest_id': summary['manifest_id']}


def validate_loss_parent(run, root, bundle):
    """Bind fresh D to stopped C and identical data; never open C main weights."""
    run, root = Path(run), Path(root).resolve(strict=True)
    if run.name != RUN_ID or run != root / 'runs' / RUN_ID or run.resolve(strict=True) != run:
        raise ValueError('Only the registered same-project pilot D may revise C loss')
    # Lazy imports keep the policy helper lightweight and avoid controller cycles.
    from scripts.tect_diff.controller import read_run, status, verify_source
    from tools.resource_locks import acquire_file

    parent_path = root / 'runs' / PARENT_RUN_ID
    with acquire_file(root / 'runtime/locks' / ('tect-registration-' + PARENT_RUN_ID + '.lock')):
        parent, parent_root, parent_config, provenance = read_run(parent_path)
        if (parent != parent_path or parent.resolve(strict=True) != parent or parent_root != root
                or provenance['commit'] != PARENT_SOURCE_COMMIT
                or provenance['config_hash'] != PARENT_CONFIG_HASH or json_hash(parent_config) != PARENT_CONFIG_HASH):
            raise ValueError('Loss revision parent source/config differs from registered C')
        state = status(parent)
        terminal_path = _regular(parent / 'pilot_receipt.json', root)
        if sha256(terminal_path) != PARENT_TERMINAL_HASH:
            raise ValueError('Loss revision parent terminal receipt changed')
        terminal = read_json(terminal_path)
        if (state.get('status') != 'COMPLETED' or state.get('outcome') != 'NO_GO'
                or state.get('controller_alive') is not False or state.get('worker_alive') is not False
                or terminal.get('status') != 'COMPLETED' or terminal.get('outcome') != 'NO_GO'
                or terminal.get('protocol_id') != PARENT_PROTOCOL
                or terminal.get('source_commit') != PARENT_SOURCE_COMMIT
                or terminal.get('config_hash') != PARENT_CONFIG_HASH):
            raise ValueError('Loss revision requires stopped, completed NO_GO parent C')
        verify_source(parent, provenance)
        child, child_root, config, child_provenance = read_run(run)
        if child != run or child_root != root or json_hash(config) != child_provenance['config_hash']:
            raise ValueError('Loss revision child provenance differs')
        verify_source(run, child_provenance)
        validate_loss_config(config, run / 'source')
        bundle_path = _regular(parent / 'data_bundle.json', root)
        if sha256(bundle_path) != PARENT_BUNDLE_HASH:
            raise ValueError('Loss revision parent data bundle differs from registered C')
        parent_summary = read_json(bundle_path)
        proof = _validate_data(bundle, parent_summary, root)
        diffusion = validate_diffusion_extension(parent / 'source' / DIFFUSION_RELATIVE,
                                                 run / 'source' / DIFFUSION_RELATIVE)
        if sha256(bundle_path) != PARENT_BUNDLE_HASH or sha256(terminal_path) != PARENT_TERMINAL_HASH:
            raise ValueError('Loss revision parent evidence changed during validation')
        receipt = {**proof, 'status': 'PASSED', 'run_id': run.name, 'protocol_id': PROTOCOL,
                   'mask_loss': MASK_LOSS_POLICY, 'source_commit': child_provenance['commit'],
                   'config_hash': child_provenance['config_hash'], 'parent_run': parent.name,
                   'parent_source_commit': PARENT_SOURCE_COMMIT, 'parent_config_hash': PARENT_CONFIG_HASH,
                   'parent_outcome': 'NO_GO', 'parent_terminal_sha256': PARENT_TERMINAL_HASH,
                   'parent_data_bundle_path': str(bundle_path), 'parent_data_bundle_sha256': PARENT_BUNDLE_HASH,
                   'parent_weights_loaded': False, 'main_initialization': 'fresh_seed42_imagenet',
                   'main_initialization_seed': 42, 'main_start_epoch': 0, 'main_start_optimizer_step': 0,
                   'diffusion_extension': diffusion, 'at': timestamp()}
        atomic_json(run / 'loss_revision.json', receipt)
        return receipt
