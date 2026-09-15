"""Import-policy tests; real artifact content is validated again at launch."""
from contextlib import ExitStack
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.tect_diff.artifact_continuation import import_completed_artifacts
from scripts.tect_diff.common import atomic_json, sha256


class ArtifactContinuationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.temp = tempfile.TemporaryDirectory(prefix='tect-artifact-policy-', dir=root/'runtime')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root/'runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C'
        self.child = self.root/'runs/TEST-CHILD'
        self.parent.mkdir(parents=True)
        self.child.mkdir(parents=True)
        self.config = {'seed': 42}
        self.parent_config = copy.deepcopy(self.config)
        self.old = {'commit': 'old', 'config_hash': 'same', 'source_hashes': {'model/tect_diff/reference.py': 'fixed'}}
        self.new = {'commit': 'new', 'config_hash': 'same', 'source_hashes': dict(self.old['source_hashes'])}
        self.state = {'controller_alive': False, 'worker_alive': False, 'status': 'INTERRUPTED', 'stage': 'MAIN'}
        self.bundle = {'manifest_hashes': {'train': 'train', 'calibration': 'cal'}}
        self.ref = {'sha256': 'reference'}
        self.cal = {'sha256': 'calibration', 'artifact_path': str(self.parent/'calibration.pth'),
                    'artifact_hash': 'logical', 'fit_receipt': {'images': 2048}}
        self.artifact = {'metadata': {'reference_sha256': 'reference', 'training_manifest_sha256': 'train',
                                      'fit_manifest_sha256': 'cal'}, 'artifact_hash': 'logical', 'fit_receipt': {'images': 2048}}
        atomic_json(self.parent/'operational_hold.json', {'active': True})
        for name, value in [('data_bundle.json', self.bundle), ('reference_receipt.json', self.ref),
                            ('calibration_receipt.json', self.cal), ('reference_health.json', {}),
                            ('reference_final_health.json', {})]:
            atomic_json(self.parent/name, value)
        (self.parent/'reference_metrics.jsonl').write_text('{"epoch":19}\n')
        stack = ExitStack()
        self.addCleanup(stack.close)
        def read_run(path):
            p = Path(path)
            return ((p, self.root, self.config, self.new) if p == self.child
                    else (p, self.root, self.parent_config, self.old))
        for target, options in [
            ('scripts.tect_diff.controller.read_run', {'side_effect': read_run}),
            ('scripts.tect_diff.controller.status', {'return_value': self.state}),
            ('scripts.tect_diff.controller.verify_source', {}),
            ('scripts.tect_diff.controller.validate_receipt', {'return_value': True}),
            ('scripts.tect_diff.data.load_bundle', {'return_value': self.bundle}),
            ('model.tect_diff.evidence.FixedTrajectoryEvidence', {'autospec': True}),
            ('torch.load', {'return_value': self.artifact})]:
            stack.enter_context(patch(target, **options))

    def test_copy_receipts_and_reject_repeat(self):
        result = import_completed_artifacts(self.child, self.parent)
        self.assertEqual(result['main_start_optimizer_step'], 0)
        self.assertFalse(result['parent_main_checkpoint_reused'])
        self.assertFalse(result['calibration_refitted'])
        for name, digest in result['copied_receipt_sha256'].items():
            self.assertEqual(sha256(self.parent/name), digest)
            self.assertEqual(sha256(self.child/name), digest)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            import_completed_artifacts(self.child, self.parent)

    def test_changed_config_rejected(self):
        self.parent_config['seed'] = 43
        with self.assertRaisesRegex(ValueError, 'identical configuration'):
            import_completed_artifacts(self.child, self.parent)

    def test_live_parent_rejected(self):
        self.state['worker_alive'] = True
        with self.assertRaisesRegex(ValueError, 'stopped'):
            import_completed_artifacts(self.child, self.parent)

    def test_parent_without_hold_rejected(self):
        atomic_json(self.parent/'operational_hold.json', {'active': False})
        with self.assertRaisesRegex(ValueError, 'held'):
            import_completed_artifacts(self.child, self.parent)

    def test_main_checkpoint_rejected(self):
        (self.parent/'last.pth').write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'completed epoch/checkpoint'):
            import_completed_artifacts(self.child, self.parent)

    def test_changed_reference_code_rejected(self):
        self.new['source_hashes']['model/tect_diff/reference.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Unaudited source'):
            import_completed_artifacts(self.child, self.parent)

    def test_wrong_calibration_binding_rejected_before_copy(self):
        self.artifact['metadata']['reference_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            import_completed_artifacts(self.child, self.parent)
        self.assertFalse((self.child/'data_bundle.json').exists())


if __name__ == '__main__':
    unittest.main()
