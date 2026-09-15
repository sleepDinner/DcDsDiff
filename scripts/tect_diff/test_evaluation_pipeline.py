"""CPU-only parity, ordering and bounded-queue tests for epoch metrics."""
import json
import random
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.tect_diff.evaluation_pipeline import EvaluationMetricPipeline
from scripts.tect_diff.metrics import aggregate_dataset, per_image_metrics


def fixtures():
    result = []
    for index, shape in enumerate(((23, 31), (35, 28), (37, 41), (17, 49))):
        target, valid = np.zeros(shape, dtype=bool), np.ones(shape, dtype=bool)
        probability = np.full(shape, .1, dtype=np.float32)
        if index == 0:  # Authentic image, including a threshold tie and a false positive.
            probability[4, 6], probability[7, 8] = .5, .9
        elif index == 1:
            target[5:19, 8:23] = True
            probability[6:20, 8:22] = .8
        elif index == 2:  # Ignored pixels cannot create contours or affect finite checks.
            target[4:30, 7:35] = True
            probability[5:29, 6:36] = .7
            valid[:2] = False
            valid[12:18, 14:20] = False
            probability[~valid] = np.nan
        else:  # Sparse positive and a nearby one-pixel prediction.
            target[8, 21] = True
            probability[8, 22] = .95
        result.append((f'image-{3-index}', probability, target, valid))
    return result


class EvaluationPipelineTests(unittest.TestCase):
    def test_actual_metrics_and_aggregate_are_identical(self):
        examples = fixtures()
        expected = [{'id': sample_id, **per_image_metrics(probability, target, valid, .5, .005)}
                    for sample_id, probability, target, valid in examples]
        for workers, limit in ((0, 8), (1, 1), (4, 2), (4, 8)):
            with self.subTest(workers=workers, limit=limit):
                with EvaluationMetricPipeline(workers, limit) as pipeline:
                    for example in examples:
                        pipeline.submit(*example, threshold=.5, boundary_ratio=.005)
                    rows = pipeline.finish()
                    self.assertEqual(pipeline.pending_count, 0)
                self.assertEqual(json.dumps(expected, sort_keys=True), json.dumps(rows, sort_keys=True))
                self.assertEqual(aggregate_dataset(expected, [row[0] for row in examples]),
                                 aggregate_dataset(rows, [row[0] for row in examples]))

    def test_completion_out_of_order_keeps_input_order(self):
        first_release, second_finished = threading.Event(), threading.Event()
        original = per_image_metrics

        def delayed(probability, *args):
            if probability[0, 0] == 0:
                if not first_release.wait(5):
                    raise RuntimeError('test release timed out')
            else:
                second_finished.set()
            return original(probability, *args)

        gt = np.zeros((7, 9), dtype=bool)
        with patch('scripts.tect_diff.evaluation_pipeline.per_image_metrics', side_effect=delayed):
            with EvaluationMetricPipeline(2, 2) as pipeline:
                try:
                    pipeline.submit('first', np.zeros(gt.shape), gt, None)
                    pipeline.submit('second', np.ones(gt.shape), gt, None)
                    self.assertTrue(second_finished.wait(2))
                    self.assertEqual(pipeline.rows, [])
                finally:
                    first_release.set()
                self.assertEqual([row['id'] for row in pipeline.finish()], ['first', 'second'])

    def test_queue_applies_backpressure_before_another_submission(self):
        release, third_entered, third_returned = threading.Event(), threading.Event(), threading.Event()
        original = per_image_metrics
        errors = []

        def blocked(*args):
            if not release.wait(5):
                raise RuntimeError('test release timed out')
            return original(*args)

        gt = np.zeros((7, 9), dtype=bool)
        with patch('scripts.tect_diff.evaluation_pipeline.per_image_metrics', side_effect=blocked):
            with EvaluationMetricPipeline(2, 2) as pipeline:
                def submit_third():
                    third_entered.set()
                    try:
                        pipeline.submit('third', gt.astype(float), gt, None)
                        third_returned.set()
                    except BaseException as error:
                        errors.append(error)

                pipeline.submit('first', gt.astype(float), gt, None)
                pipeline.submit('second', gt.astype(float), gt, None)
                third = threading.Thread(target=submit_third)
                third.start()
                try:
                    self.assertTrue(third_entered.wait(2))
                    self.assertFalse(third_returned.wait(.05))
                    self.assertEqual(pipeline.pending_count, 2)
                    self.assertEqual(pipeline.submitted, 2)
                finally:
                    release.set()
                    third.join(5)
                self.assertFalse(third.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual([row['id'] for row in pipeline.finish()], ['first', 'second', 'third'])

    def test_metric_errors_propagate_and_pool_closes(self):
        gt = np.zeros((7, 9), dtype=bool)
        for workers in (0, 2):
            pipeline = EvaluationMetricPipeline(workers, 2)
            with self.assertRaisesRegex(ValueError, 'ALL_IGNORED_SAMPLE'):
                with pipeline:
                    pipeline.submit('invalid', gt.astype(float), gt, gt)
                    pipeline.finish()
            self.assertTrue(pipeline.closed)
            self.assertEqual(pipeline.pending_count, 0)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                pipeline.submit('late', gt.astype(float), gt, None)

    def test_rng_and_input_arrays_are_unchanged(self):
        examples = fixtures()
        before = [(p.tobytes(), gt.tobytes(), valid.tobytes()) for _, p, gt, valid in examples]
        torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        with EvaluationMetricPipeline(4, 8) as pipeline:
            for example in examples:
                pipeline.submit(*example)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        self.assertEqual(numpy_rng[0], np.random.get_state()[0])
        self.assertTrue(np.array_equal(numpy_rng[1], np.random.get_state()[1]))
        self.assertEqual(numpy_rng[2:], np.random.get_state()[2:])
        self.assertEqual(before, [(p.tobytes(), gt.tobytes(), valid.tobytes()) for _, p, gt, valid in examples])

    def test_config_and_cpu_array_contract(self):
        for workers, limit in ((-1, 8), (True, 8), (1.5, 8), (4, 0), (4, True)):
            with self.assertRaises(ValueError):
                EvaluationMetricPipeline(workers, limit)
        with EvaluationMetricPipeline(2, 2) as pipeline:
            with self.assertRaisesRegex(TypeError, 'NumPy'):
                pipeline.submit('not-an-array', [[0., 0.], [0., 0.]], np.zeros((2, 2)), None)


if __name__ == '__main__':
    unittest.main()
