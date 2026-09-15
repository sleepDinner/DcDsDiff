"""Pure CPU recovery tests: reuse only complete, model-bound evaluation state."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.tect_diff.common import atomic_json
from scripts.tect_diff.pilot_metrics import DEFAULT_GATE_THRESHOLDS, pilot_health_gate
from scripts.tect_diff.pilot_state import (
    ensure_evaluation_binding, load_completed_dataset, validate_confirmation,
)
from scripts.tect_diff.test_pilot_metrics import COUNTS, record
from scripts.tect_diff.metrics import aggregate_dataset, per_image_metrics
import numpy as np


BINDING = {'epoch': 2, 'config_hash': 'config-a', 'source_commit': 'source-a',
           'checkpoint_parameter_sha256': 'parameters-a', 'group_records_hash': 'groups-a'}


class PilotStateTests(unittest.TestCase):
    def test_binding_replay_is_idempotent_and_change_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'epoch-002'
            ensure_evaluation_binding(directory, BINDING)
            before = (directory/'evaluation_binding.json').read_bytes()
            ensure_evaluation_binding(directory, BINDING)
            self.assertEqual((directory/'evaluation_binding.json').read_bytes(), before)
            for field in BINDING:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    ensure_evaluation_binding(directory, {**BINDING, field: 'changed'})
            self.assertEqual((directory/'evaluation_binding.json').read_bytes(), before)

    def test_unbound_prior_results_fail_before_any_write(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            atomic_json(directory/'Casiav1.rank0.json', [])
            with self.assertRaises(ValueError):
                ensure_evaluation_binding(directory, BINDING)
            self.assertFalse((directory/'evaluation_binding.json').exists())

    def test_completed_dataset_reuse_checks_ids_and_metric_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.assertIsNone(load_completed_dataset(directory, 'Casiav1', ['one']))
            gt = np.zeros((10, 10), dtype=bool)
            gt[3:7, 2:8] = True
            rows = [{'id': 'one', **per_image_metrics(gt.astype(float), gt)}]
            payload = {'per_image': rows, 'metrics': aggregate_dataset(rows, ['one']), 'new_inference_images': 1}
            atomic_json(directory/'Casiav1.json', payload)
            self.assertEqual(load_completed_dataset(directory, 'Casiav1', ['one']), payload)
            with self.assertRaises(ValueError):
                load_completed_dataset(directory, 'Casiav1', ['different-id'])
            payload['metrics']['dataset_pixel_f1'] = .2
            atomic_json(directory/'Casiav1.json', payload)
            with self.assertRaises(ValueError):
                load_completed_dataset(directory, 'Casiav1', ['one'])

    def test_corrupt_complete_file_is_not_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'Columbia.json'
            path.write_text('{"per_image":')
            with self.assertRaises(ValueError):
                load_completed_dataset(Path(folder), 'Columbia', ['one'])
            self.assertEqual(path.read_text(), '{"per_image":')

    def test_confirmation_recomputes_gate_and_requires_full_binding(self):
        thresholds = {**DEFAULT_GATE_THRESHOLDS, 'consecutive_epochs': 1}
        baselines = dict.fromkeys(COUNTS, .2)
        row = record(2)
        row.update(run_id='pilot-a', config_hash=BINDING['config_hash'], source_commit=BINDING['source_commit'])
        gate = pilot_health_gate([row], baselines, thresholds=thresholds,
                                 expected_counts=COUNTS, authentic_expected_count=64)
        existing = {**BINDING, 'metrics': row, 'gate': gate}
        self.assertEqual(validate_confirmation(existing, BINDING, baselines, COUNTS, thresholds, 64), gate)
        for mutate in (
            lambda x: x.update(checkpoint_parameter_sha256='wrong'),
            lambda x: x['gate'].update(passed=False),
            lambda x: x['metrics'].update(epoch_index=1),
            lambda x: x['metrics'].update(config_hash='wrong'),
            lambda x: x['metrics'].update(average_test2=.2),
            lambda x: x['metrics']['health'].update(runtime_healthy=False),
            lambda x: x['metrics']['health']['authentic_probe'].update(authentic_pixel_false_positive_rate=.1),
        ):
            changed = copy.deepcopy(existing)
            mutate(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_confirmation(changed, BINDING, baselines, COUNTS, thresholds, 64)


if __name__ == '__main__':
    unittest.main()
