"""CPU-only tests for pilot coverage, honest reporting and fail-closed promotion."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.tect_diff.metrics import aggregate_dataset, per_image_metrics
from scripts.tect_diff.pilot_metrics import (
    aggregate_test2, build_epoch_record, pilot_health_gate, upsert_epoch_jsonl,
)


COUNTS = {'Casiav1': 2, 'Columbia': 1}


def results():
    answer = {}
    for name, count in COUNTS.items():
        gt = np.zeros((10, 10), dtype=bool)
        gt[2:5, 2:5] = True
        rows = [{'id': f'{name}-{i}', **per_image_metrics(gt.astype(float), gt)}
                for i in range(count)]
        answer[name] = aggregate_dataset(rows, [row['id'] for row in rows])
    return answer


def probe():
    gt = np.zeros((10, 10), dtype=bool)
    rows = [{'id': f'auth-{i}', **per_image_metrics(gt.astype(float), gt)} for i in range(64)]
    return aggregate_dataset(rows, [row['id'] for row in rows])


def record(epoch=0):
    return build_epoch_record(epoch, 12, results(), COUNTS,
                              selection_authority='registered pilot protocol',
                              selection_scope='Casiav1+Columbia (fixed pilot subsets)',
                              health={'runtime_healthy': True, 'authentic_probe': probe()})


class PilotMetricsTests(unittest.TestCase):
    def test_two_dataset_macro_preserves_counts_confusion_and_mae(self):
        data = results()
        data['Casiav1']['dataset_pixel_f1'] = .4
        data['Casiav1']['pixel_f1_sum'] = .8
        answer = aggregate_test2(data, COUNTS)
        self.assertAlmostEqual(answer['test2_macro_pixel_f1'], .7)
        self.assertEqual(answer['count'], 3)
        self.assertEqual(answer['tp'], 27)
        self.assertEqual(answer['test2_macro_mae'], 0)
        self.assertEqual(answer['specificity'], 1)

    def test_incomplete_nonfinite_or_inconsistent_counts_cannot_select(self):
        for mutate in (
            lambda x: x.pop('Columbia'),
            lambda x: x.update(NIST16=x['Columbia']),
            lambda x: x['Casiav1'].update(count=1),
            lambda x: x['Casiav1'].update(id_unique_count=1),
            lambda x: x['Casiav1'].update(dataset_pixel_f1=float('nan')),
            lambda x: x['Casiav1'].update(dataset_mae=float('inf')),
            lambda x: x['Casiav1'].update(tp=-1),
            lambda x: x['Casiav1'].update(valid_pixels=1),
        ):
            data = results()
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                aggregate_test2(data, COUNTS)

    def test_display_format_names_only_evaluated_populations(self):
        row = record()
        self.assertEqual((row['epoch_index'], row['epoch_number'], row['total_epochs']), (0, 1, 12))
        self.assertEqual(row['metric'], 'PixelF1(threshold=0.5, mode=origin)')
        self.assertEqual(row['image_counts'], COUNTS)
        self.assertEqual(row['pixel_f1'], {'Casiav1': 1., 'Columbia': 1.})
        self.assertEqual(row['selection_protocol'], 'test_selected')
        self.assertEqual(row['average_test2'], 1.)
        self.assertNotIn('average_all8', row)
        self.assertNotIn('average_mvss5', row)

    def test_jsonl_upsert_is_sorted_idempotent_and_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics_per_epoch.jsonl'
            upsert_epoch_jsonl(path, record(1))
            upsert_epoch_jsonl(path, record(0))
            before = path.read_bytes()
            upsert_epoch_jsonl(path, record(1))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([json.loads(line)['epoch_index'] for line in path.read_text().splitlines()], [0, 1])
            changed = record(1)
            changed['selected'] = True
            upsert_epoch_jsonl(path, changed)
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_corrupt_duplicate_or_nonfinite_jsonl_is_not_overwritten(self):
        for content in ('{"epoch_index":', json.dumps(record()) + '\n' + json.dumps(record()) + '\n',
                        json.dumps({**record(), 'loss': float('nan')}) + '\n'):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'metrics.jsonl'
                path.write_text(content)
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    upsert_epoch_jsonl(path, record(1))
                self.assertEqual(before, path.read_bytes())

    def test_atomic_replace_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.jsonl'
            upsert_epoch_jsonl(path, record())
            before = path.read_bytes()
            with patch('scripts.tect_diff.pilot_metrics.os.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    upsert_epoch_jsonl(path, record(1))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_promotion_requires_three_complete_consecutive_healthy_epochs(self):
        baseline = dict.fromkeys(COUNTS, .2)
        rows = [record(i) for i in range(3)]
        self.assertTrue(pilot_health_gate(rows, baseline)['passed'])
        self.assertFalse(pilot_health_gate(rows[:2], baseline)['passed'])
        self.assertFalse(pilot_health_gate([record(0), record(2), record(3)], baseline)['passed'])
        rows[-1]['epoch_complete'] = False
        self.assertFalse(pilot_health_gate(rows, baseline)['passed'])

    def test_bad_runtime_authentic_probe_and_all_foreground_fail_gate(self):
        baseline = dict.fromkeys(COUNTS, .2)
        for mode in ('runtime', 'missing-probe', 'false-positive-probe', 'foreground'):
            rows = [record(i) for i in range(3)]
            last = rows[-1]
            if mode == 'runtime':
                last['health']['runtime_healthy'] = False
            elif mode == 'missing-probe':
                last['health'].pop('authentic_probe')
            elif mode == 'false-positive-probe':
                last['health']['authentic_probe']['authentic_pixel_false_positive_rate'] = .1
            else:
                for ds in last['dataset_metrics'].values():
                    ds.update(tp=ds['tp'] + ds['fn'], fp=ds['tn'], fn=0, tn=0,
                              dataset_pixel_f1=.2, pixel_f1_sum=.2 * ds['count'])
            with self.subTest(mode=mode):
                self.assertFalse(pilot_health_gate(rows, baseline)['passed'])

    def test_gate_rejects_unknown_thresholds_invalid_baselines_and_duplicates(self):
        rows = [record(i) for i in range(3)]
        for baselines, thresholds in (({'Casiav1': .2}, None),
                                      (dict.fromkeys(COUNTS, float('nan')), None),
                                      (dict.fromkeys(COUNTS, .2), {'macro_f1_mni': .5})):
            with self.assertRaises(ValueError):
                pilot_health_gate(rows, baselines, thresholds=thresholds)
        with self.assertRaises(ValueError):
            pilot_health_gate([record(), record(), record(1)], dict.fromkeys(COUNTS, .2))

    def test_gate_rejects_population_change_and_checks_foreground_margin(self):
        rows = [record(i) for i in range(3)]
        answer = pilot_health_gate(rows, dict.fromkeys(COUNTS, .2),
                                   expected_counts={'Casiav1': 128, 'Columbia': 128})
        self.assertFalse(answer['passed'])
        self.assertFalse(pilot_health_gate(rows, dict.fromkeys(COUNTS, .95))['passed'])
        self.assertTrue(pilot_health_gate([record(4)], dict.fromkeys(COUNTS, .2),
                                         thresholds={'consecutive_epochs': 1})['passed'])


if __name__ == '__main__':
    unittest.main()
