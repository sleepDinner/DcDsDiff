"""Exercise actual worker control flow with CPU stubs, without importing CUDA models.

The class AST is compiled unchanged with explicit IO/collective/model stubs.
These tests cover recovery routing, not numerical equivalence of real DDP runs.
"""
import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from scripts.tect_diff.common import atomic_json, json_hash, read_json
from scripts.tect_diff.metrics import aggregate_dataset, per_image_metrics
from scripts.tect_diff.pilot_metrics import (
    DEFAULT_GATE_THRESHOLDS, aggregate_test2, build_epoch_record,
    pilot_health_gate, upsert_epoch_jsonl,
)
from scripts.tect_diff.pilot_state import (
    ensure_evaluation_binding, load_completed_dataset, validate_confirmation,
)
from scripts.tect_diff.test_pilot_metrics import COUNTS, probe


def worker_class():
    namespace = {
        'Worker': object, 'Path': Path, 'json': json, 'math': __import__('math'),
        'time': __import__('time'), 'atomic_json': atomic_json, 'read_json': read_json,
        'json_hash': json_hash, 'aggregate_dataset': aggregate_dataset,
        'aggregate_test2': aggregate_test2, 'build_epoch_record': build_epoch_record,
        'pilot_health_gate': pilot_health_gate, 'upsert_epoch_jsonl': upsert_epoch_jsonl,
        'ensure_evaluation_binding': ensure_evaluation_binding,
        'load_completed_dataset': load_completed_dataset, 'validate_confirmation': validate_confirmation,
        'rng_state': lambda: 'saved-rng', 'restore_rng': Mock(),
        'tensor_hash': lambda _: 'model-digest',
        'dist': SimpleNamespace(barrier=Mock(), broadcast=Mock(), broadcast_object_list=Mock()),
    }
    source = ast.parse(Path(__file__).with_name('pilot_worker.py').read_text())
    definition = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'PilotWorker')
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, 'pilot_worker.py (isolated CPU control flow)', 'exec'), namespace)
    return namespace['PilotWorker'], namespace


def dataset_payload(name, count):
    gt = np.zeros((10, 10), dtype=bool)
    gt[2:5, 2:5] = True
    rows = [{'id': f'{name}-{i}', **per_image_metrics(gt.astype(float), gt)} for i in range(count)]
    return {'per_image': rows, 'metrics': aggregate_dataset(rows, [row['id'] for row in rows]),
            'evaluation_seconds': 2.5, 'new_inference_images': count, 'reused_same_epoch_images': 0}


class PilotWorkerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        cls, self.namespace = worker_class()
        self.worker = cls.__new__(cls)
        self.worker.run = Path(self.temporary.name)/'pilot-a'
        self.worker.run.mkdir()
        self.worker.rank, self.worker.world = 0, 2
        self.worker.config_hash = 'config-a'
        self.worker.device = SimpleNamespace(index=0)
        self.worker.config = {
            'seed': 42, 'resolution': 512,
            'training': {'accumulation_steps': 1, 'global_batch': 12, 'micro_batch': 6,
                         'epochs': 3, 'scheduler_epochs': 100, 'minimum_learning_rate': 1e-6},
            'pilot': {'gate': dict(DEFAULT_GATE_THRESHOLDS), 'consecutive_degenerate_epochs': 3},
        }
        tests = {name: [{'id': f'{name}-{i}', 'positive_pixels': 9, 'mask_hw': [10, 10],
                          'ignored_pixels': 0} for i in range(count)] for name, count in COUNTS.items()}
        self.worker.bundle = {'tests': tests, 'tests_full': tests, 'train': [{'id': 'training-only'}],
                              'authentic_probe': [{'id': f'auth-{i}'} for i in range(64)]}
        self.model = SimpleNamespace(reference=object(), evidence=SimpleNamespace(assert_frozen=Mock()),
                                     buffers=lambda: [], eval=Mock(), train=Mock(),
                                     sample=Mock(side_effect=AssertionError('Unexpected inference')))
        atomic_json(self.worker.run/'provenance.json', {'commit': 'source-a'})

    def row(self, epoch):
        result = {'datasets': {name: dataset_payload(name, count)['metrics'] for name, count in COUNTS.items()},
                  'evaluation_seconds': 5.}
        result['datasets']['CASIA2_authentic_probe'] = probe()
        return self.worker.epoch_row(epoch, result, {'loss': .1})

    def checkpoint(self, epoch):
        return {'epoch': epoch, 'next_epoch': epoch+1, 'epoch_complete': True, 'evaluation_complete': True,
                'optimizer_step': 171*(epoch+1), 'trainable_parameter_sha256': 'parameters-a',
                'pilot_epoch_record': self.row(epoch), 'training_summary': {'loss': .1},
                'training_seconds': 200., 'evaluation_seconds': 5.*(epoch+1),
                'best': {'epoch': 0, 'score': 1.}}

    def test_fully_cached_evaluation_never_samples_and_preserves_cost(self):
        worker = self.worker
        binding = worker.evaluation_binding(self.checkpoint(0))
        groups = worker.bundle['tests']
        directory = worker.run/'epochs'/'epoch-000'
        ensure_evaluation_binding(directory, {**binding, 'populations': {name: json_hash(rows) for name, rows in groups.items()}})
        for name, count in COUNTS.items():
            atomic_json(directory/f'{name}.json', dataset_payload(name, count))
        answer = worker.evaluate_groups(self.model, groups, directory, binding)
        self.assertTrue(answer['evaluation_complete'])
        self.assertEqual(answer['evaluation_seconds'], 5.)
        self.assertEqual({name: result['count'] for name, result in answer['datasets'].items()}, COUNTS)
        self.model.sample.assert_not_called()
        self.namespace['restore_rng'].assert_called_once_with('saved-rng')

    def test_complete_checkpoint_replay_upserts_once_and_rejects_foreign_history(self):
        checkpoint = self.checkpoint(0)
        self.worker.write_epoch(checkpoint)
        self.worker.write_epoch(checkpoint)
        path = self.worker.run/'metrics_per_epoch.jsonl'
        self.assertEqual(len(path.read_text().splitlines()), 1)
        row = json.loads(path.read_text())
        row['source_commit'] = 'wrong-source'
        path.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(RuntimeError, 'history'):
            self.worker.gate_and_confirmation(self.model, checkpoint)

    def test_existing_confirmation_is_revalidated_without_inference(self):
        worker = self.worker
        for epoch in range(3):
            checkpoint = self.checkpoint(epoch)
            worker.write_epoch(checkpoint)
        binding = worker.evaluation_binding(checkpoint)
        row = self.row(2)
        gate = pilot_health_gate([row], worker.foreground_baselines(True),
                                 thresholds={**worker.config['pilot']['gate'], 'consecutive_epochs': 1},
                                 expected_counts=COUNTS, authentic_expected_count=64)
        path = worker.run/'confirmation_result.json'
        atomic_json(path, {**binding, 'metrics': row, 'gate': gate})
        worker.evaluate_groups = Mock(side_effect=AssertionError('Unexpected repeated confirmation'))
        outcome, result = worker.gate_and_confirmation(self.model, checkpoint)
        self.assertEqual(outcome, 'READY_FOR_FULL')
        self.assertEqual(result, gate)
        worker.evaluate_groups.assert_not_called()
        gate['passed'] = False
        atomic_json(path, {**binding, 'metrics': row, 'gate': gate})
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            worker.gate_and_confirmation(self.model, checkpoint)

    def test_pending_evaluation_resume_does_not_train_or_step_scheduler(self):
        worker = self.worker
        worker.config['training']['epochs'] = 1
        pending = self.checkpoint(0)
        pending.update(evaluation_complete=False, best=None)
        pending.pop('pilot_epoch_record')
        atomic_json(worker.run/'last.pth', pending)
        optimizer, scheduler = Mock(), Mock()
        self.namespace.update({
            'torch': SimpleNamespace(load=Mock(return_value=copy.deepcopy(pending)),
                optim=SimpleNamespace(lr_scheduler=SimpleNamespace(CosineAnnealingLR=lambda *a, **kw: scheduler)),
                cuda=SimpleNamespace(amp=SimpleNamespace(GradScaler=lambda **kw: object()), reset_peak_memory_stats=Mock())),
            'DDP': lambda model, **kwargs: model,
            'ManifestDataset': lambda rows, *a, **kw: list(rows),
            'atomic_torch_save': atomic_json,
        })
        worker.create_model = lambda: (self.model, {}, {}, {})
        worker.optimizer = lambda _: optimizer
        worker.restore = Mock()
        worker.engineering_probe = Mock(side_effect=AssertionError('Unexpected engineering restart'))
        worker.loader = Mock(side_effect=AssertionError('Unexpected repeated training'))
        result = {'datasets': {name: dataset_payload(name, count)['metrics'] for name, count in COUNTS.items()},
                  'evaluation_seconds': 5.}
        result['datasets']['CASIA2_authentic_probe'] = probe()
        worker.evaluate_groups = Mock(return_value=result)
        worker.training_checkpoint = lambda model, opt, sched, scaler, epoch, step, complete, **extra: {
            **pending, 'epoch': epoch, 'optimizer_step': step, 'evaluation_complete': complete, **extra}
        worker.status, worker.persist_outcome = Mock(), Mock()
        worker.main_training()
        worker.restore.assert_called_once()
        worker.engineering_probe.assert_not_called()
        worker.loader.assert_not_called()
        optimizer.step.assert_not_called()
        scheduler.step.assert_not_called()
        worker.evaluate_groups.assert_called_once()
        worker.persist_outcome.assert_called_once()
        self.assertEqual(read_json(worker.run/'last.pth')['optimizer_step'], pending['optimizer_step'])
        self.assertEqual(len((worker.run/'metrics_per_epoch.jsonl').read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
