"""CPU provenance gates for the single registered empty-target loss revision."""
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts.tect_diff.common import atomic_json, json_hash, read_json, sha256
from scripts.tect_diff import pilot_loss_revision as revision


PROJECT = Path(__file__).resolve().parents[2]
OLD_IMPORT = 'from model.loss import structure_loss'
NEW_IMPORT = 'from model.tect_diff.mask_loss import tect_mask_loss'
OLD_CALL = '            mask_loss = structure_loss(controlled.float(), gt.float())'
NEW_CALL = "            mask_loss = tect_mask_loss(controlled.float(), gt.float(), self.config['training'].get('mask_loss', 'structure_v1'))"
ORIGINAL = OLD_IMPORT + '\n\nclass Example:\n    def forward(self):\n        with scope:\n' + OLD_CALL + '\n\n    def sample(self):\n        return 10\n'
REVISED = ORIGINAL.replace(OLD_IMPORT, NEW_IMPORT).replace(OLD_CALL, NEW_CALL)


def manifest_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


class LossRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='loss-revision-', dir=PROJECT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.parent = self.root / 'runs' / revision.PARENT_RUN_ID
        self.run = self.root / 'runs' / revision.RUN_ID
        self.parent.mkdir(parents=True)
        self.run.mkdir(parents=True)
        self.baseline = read_json(PROJECT / revision.PARENT_CONFIG_RELATIVE)
        self.config = copy.deepcopy(self.baseline)
        self.config['protocol_id'] = revision.PROTOCOL
        self.config['training']['mask_loss'] = revision.MASK_LOSS_POLICY
        for run, config, source in ((self.parent, self.baseline, revision.PARENT_SOURCE_COMMIT),
                                    (self.run, self.config, 'candidate-source')):
            canonical = run / 'source' / revision.PARENT_CONFIG_RELATIVE
            atomic_json(canonical, self.baseline)
            diffusion = run / 'source/model/tect_diff/diffusion.py'
            diffusion.parent.mkdir(parents=True)
            diffusion.write_text(ORIGINAL if run == self.parent else REVISED, encoding='utf-8')
            atomic_json(run / 'resolved_config.json', config)
            atomic_json(run / 'provenance.json', {'commit': source, 'config_hash': json_hash(config)})
        self.terminal = {'status': 'COMPLETED', 'outcome': 'NO_GO', 'protocol_id': revision.PARENT_PROTOCOL,
                         'source_commit': revision.PARENT_SOURCE_COMMIT, 'config_hash': revision.PARENT_CONFIG_HASH}
        atomic_json(self.parent / 'pilot_receipt.json', self.terminal)
        self.state = {'status': 'COMPLETED', 'outcome': 'NO_GO', 'controller_alive': False, 'worker_alive': False}
        train = [{'id': 'CASIA2:fixture-' + str(i), 'authentic_declared': i < 4096,
                  'empty_mask': i < 4096, 'rgb_pixels_sha256': 'rgb-' + str(i),
                  'source_ids': ['CASIA2:source-' + str(i)]} for i in range(8192)]
        self.rows = {role: [{'id': role + ':非ASCII', 'authentic_declared': False, 'empty_mask': False,
                            'rgb_pixels_sha256': role, 'source_ids': [role]}] for role in revision.ROLES}
        self.rows['train'] = train
        self.rows['preflight_train'] = train[:32] + train[4096:4128]
        self.parent_summary = self.make_summary(self.parent)
        self.summary = self.make_summary(self.run)
        atomic_json(self.parent / 'data_bundle.json', self.parent_summary)
        atomic_json(self.run / 'data_bundle.json', self.summary)
        self.bundle = self.make_bundle()
        self.controller = types.ModuleType('scripts.tect_diff.controller')
        self.controller.read_run = self.read_run
        self.controller.status = lambda run: dict(self.state)
        self.controller.verify_source = lambda run, provenance: None
        self.locks = types.ModuleType('tools.resource_locks')
        self.locks.acquire_file = lambda path: nullcontext()

    def make_summary(self, run):
        paths, hashes = {}, {}
        for role, rows in self.rows.items():
            path = run / 'manifests' / (role + '.json')
            atomic_json(path, rows)
            paths[role], hashes[role] = str(path), manifest_hash(rows)
        return {'audit_version': 'TECT-CASIA2-PILOT-DATA-V2-QUARANTINE', 'manifest_id': 'fixed-roles',
                'manifest_paths': paths, 'manifest_hashes': hashes, 'seed': 42,
                'sample_sizes': {key: self.baseline['pilot'][key] for key in (
                    'train_authentic', 'train_tampered', 'quick_test_per_dataset', 'authentic_probe', 'preflight_train')},
                'train_count': 8192, 'train_authentic_count': 4096,
                'quarantine_policy': 'exact-pairs', 'quarantined_pair_count': 2,
                'quarantined_pairs': self.baseline['data']['quarantine_pairs'], 'invalid_training_pairs': [],
                'reference_source_mode': 'inherited', 'inherited_manifest_hashes': {'train': 'same'},
                'inherited_manifest_paths': {'train': 'same'}, 'mask_semantics': 'binary',
                'selection_rule': 'fixed', 'leakage_rule': 'fixed',
                'current_audit_status': 'COMPLETED', 'current_blocking_audit_error_count': 0}

    def make_bundle(self):
        value = {role: copy.deepcopy(self.rows[role]) for role in (
            'train', 'preflight_train', 'reference', 'calibration', 'authentic_probe')}
        value.update(tests={name: copy.deepcopy(self.rows['test_' + name]) for name in ('Casiav1', 'Columbia')},
                     tests_full={name: copy.deepcopy(self.rows['test_full_' + name]) for name in ('Casiav1', 'Columbia')},
                     summary=self.summary, summary_path=str(self.run / 'data_bundle.json'),
                     manifest_hashes=self.summary['manifest_hashes'], manifest_paths=self.summary['manifest_paths'],
                     reference_source_mode=self.summary['reference_source_mode'])
        return value

    def read_run(self, path):
        path = Path(path)
        return path, self.root, read_json(path / 'resolved_config.json'), read_json(path / 'provenance.json')

    def validate_parent(self):
        with patch.dict('sys.modules', {'scripts.tect_diff.controller': self.controller,
                                        'tools.resource_locks': self.locks}), \
                patch.object(revision, 'PARENT_TERMINAL_HASH', sha256(self.parent / 'pilot_receipt.json')), \
                patch.object(revision, 'PARENT_BUNDLE_HASH', sha256(self.parent / 'data_bundle.json')):
            return revision.validate_loss_parent(self.run, self.root, self.bundle)

    def test_config_only_protocol_and_loss_change_without_mutating_input(self):
        before = copy.deepcopy(self.config)
        revision.validate_loss_config(self.config, self.run / 'source')
        self.assertEqual(self.config, before)
        for section, key, value in [('training', 'epochs', 9), ('training', 'learning_rate', 2e-4),
                                     ('sampling', 'steps', 9), ('pilot', 'train_tampered', 1024),
                                     ('training', 'mask_loss', 'structure_v1')]:
            changed = copy.deepcopy(self.config)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                revision.validate_loss_config(changed, self.run / 'source')
        changed = copy.deepcopy(self.config)
        changed['unregistered'] = True
        with self.assertRaises(ValueError):
            revision.validate_loss_config(changed, self.run / 'source')
        with self.assertRaises(ValueError):
            revision.validate_loss_config(self.baseline, self.run / 'source')

    def test_config_rejects_changed_canonical_c(self):
        changed = copy.deepcopy(self.baseline)
        changed['seed'] = 43
        atomic_json(self.run / 'source' / revision.PARENT_CONFIG_RELATIVE, changed)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            revision.validate_loss_config(self.config, self.run / 'source')

    def test_diffusion_accepts_exact_two_changes_and_rejects_any_other_change(self):
        original = self.parent / 'source/model/tect_diff/diffusion.py'
        revised = self.run / 'source/model/tect_diff/diffusion.py'
        revision.validate_diffusion_extension(original, revised)
        for text in (ORIGINAL, REVISED.replace('return 10', 'return 9'), REVISED + '\n# extra\n',
                     REVISED.replace(NEW_IMPORT, NEW_IMPORT + '\n' + NEW_IMPORT),
                     REVISED.replace("'structure_v1'", "'empty_target_bce_v1'")):
            revised.write_text(text, encoding='utf-8')
            with self.subTest(text=text[-60:]), self.assertRaises(ValueError):
                revision.validate_diffusion_extension(original, revised)

    def test_parent_success_records_all_nine_roles_without_parent_mutation_or_weights(self):
        forbidden = self.parent / 'last.pth'
        forbidden.write_bytes(b'not a checkpoint; this must never be loaded')
        before = {str(path): sha256(path) for path in self.parent.rglob('*') if path.is_file()}
        result = self.validate_parent()
        self.assertEqual(result['status'], 'PASSED')
        self.assertEqual(result['train_count'], 8192)
        self.assertEqual(result['train_authentic_count'], 4096)
        self.assertFalse(result['parent_weights_loaded'])
        self.assertEqual(result['main_initialization_seed'], 42)
        self.assertEqual(set(result['unchanged_manifest_hashes']), set(revision.ROLES))
        self.assertEqual(read_json(self.run / 'loss_revision.json'), result)
        self.assertEqual(before, {str(path): sha256(path) for path in self.parent.rglob('*') if path.is_file()})

    def test_parent_rejects_alive_or_noncompleted_parent_without_receipt(self):
        for key, value in [('controller_alive', True), ('worker_alive', True), ('status', 'RUNNING'), ('outcome', 'READY_FOR_FULL')]:
            state = dict(self.state)
            self.state[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate_parent()
            self.assertFalse((self.run / 'loss_revision.json').exists())
            self.state = state

    def test_parent_rejects_wrong_run_source_config_or_terminal(self):
        for target, key, value in [(self.parent / 'provenance.json', 'commit', 'other'),
                                    (self.parent / 'provenance.json', 'config_hash', 'other'),
                                    (self.parent / 'pilot_receipt.json', 'source_commit', 'other'),
                                    (self.parent / 'pilot_receipt.json', 'outcome', 'READY_FOR_FULL')]:
            original = read_json(target)
            atomic_json(target, dict(original, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate_parent()
            atomic_json(target, original)
        with self.assertRaises(ValueError):
            revision.validate_loss_parent(self.parent, self.root, self.bundle)

    def test_parent_enforces_registered_receipt_byte_hashes(self):
        with patch.dict('sys.modules', {'scripts.tect_diff.controller': self.controller, 'tools.resource_locks': self.locks}):
            with self.assertRaises(ValueError):
                revision.validate_loss_parent(self.run, self.root, self.bundle)
            with patch.object(revision, 'PARENT_TERMINAL_HASH', sha256(self.parent / 'pilot_receipt.json')):
                with self.assertRaises(ValueError):
                    revision.validate_loss_parent(self.run, self.root, self.bundle)

    def test_parent_rejects_each_changed_role_even_with_internally_valid_hash(self):
        for role in revision.ROLES:
            path = Path(self.summary['manifest_paths'][role])
            original = read_json(path)
            changed = copy.deepcopy(original)
            changed[0]['id'] += '-changed'
            atomic_json(path, changed)
            old_hash = self.summary['manifest_hashes'][role]
            self.summary['manifest_hashes'][role] = manifest_hash(changed)
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.validate_parent()
            atomic_json(path, original)
            self.summary['manifest_hashes'][role] = old_hash

    def test_parent_rejects_tampered_manifest_loaded_rows_and_semantic_contract(self):
        path = Path(self.summary['manifest_paths']['reference'])
        original = path.read_bytes()
        path.write_text('[]', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate_parent()
        path.write_bytes(original)
        self.bundle['train'][0]['source_ids'] = ['tampered']
        with self.assertRaises(ValueError):
            self.validate_parent()
        self.bundle['train'][0] = copy.deepcopy(self.rows['train'][0])
        self.summary['quarantined_pair_count'] = 1
        with self.assertRaises(ValueError):
            self.validate_parent()

    def test_parent_rejects_missing_role_external_path_and_wrong_training_count(self):
        saved = self.summary['manifest_paths'].pop('preflight_train')
        with self.assertRaises(ValueError):
            self.validate_parent()
        self.summary['manifest_paths']['preflight_train'] = saved
        self.summary['manifest_paths']['preflight_train'] = str(PROJECT / 'AGENTS.md')
        with self.assertRaises(ValueError):
            self.validate_parent()
        self.summary['manifest_paths']['preflight_train'] = saved
        self.summary['train_count'] = 8191
        with self.assertRaises(ValueError):
            self.validate_parent()

    def test_parent_rejects_source_verifier_failure_and_late_evidence_mutation(self):
        self.controller.verify_source = lambda *args: (_ for _ in ()).throw(ValueError('source changed'))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.validate_parent()
        self.controller.verify_source = lambda *args: None
        actual = revision.validate_diffusion_extension
        def mutate(*args):
            proof = actual(*args)
            with (self.parent / 'pilot_receipt.json').open('a') as handle:
                handle.write(' ')
            return proof
        with patch.object(revision, 'validate_diffusion_extension', side_effect=mutate):
            with self.assertRaises(ValueError):
                self.validate_parent()
        self.assertFalse((self.run / 'loss_revision.json').exists())


if __name__ == '__main__':
    unittest.main()
