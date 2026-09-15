"""Bounded CPU-only regression checks for reference collapse detection."""
import math
import copy
import unittest

from scripts.tect_diff.reference_health import (
    ReferenceHealthError,
    require_final_reference_probe,
    require_healthy_reference,
    summarize_reference_epoch,
)


def sums_for(ratios, counts=(100, 100, 100, 100)):
    """Construct valid moments of epsilon prediction with unit noise energy."""
    rows = []
    for ratio, count in zip(ratios, counts):
        # For ratio <= 1 this models signal plus residual; above 1, pure noise.
        energy = abs(1.0 - ratio)
        cross = max(0.0, 1.0 - ratio)
        rows.append([ratio * count, count, energy * count, cross * count, count])
    return rows


class ReferenceHealthTests(unittest.TestCase):
    def test_historical_collapse_is_blocked_at_epoch_seven(self):
        # Epochs 4 and 6-7 follow measured history; epoch 5 is a mixed transition.
        history = []
        ratios_by_epoch = [[0.086, 0.111, 0.150, 0.214], [0.7, 0.8, 1.0, 1.0],
                           [1.00001] * 4, [1.00001] * 4]
        for epoch, ratios in enumerate(ratios_by_epoch, start=4):
            metrics = summarize_reference_epoch(epoch, sums_for(ratios), history)
            if epoch < 7:
                require_healthy_reference(metrics)
            else:
                with self.assertRaisesRegex(ReferenceHealthError, 'collapsed'):
                    require_healthy_reference(metrics)
                self.assertEqual(metrics['reference_health']['first_learned_epoch'], 4)
                self.assertEqual(metrics['reference_health']['consecutive_collapsed_epochs'], 2)
            history.append(metrics)

    def test_fresh_unlearned_epochs_do_not_trigger_collapse(self):
        history = []
        for epoch in range(6):
            metrics = summarize_reference_epoch(epoch, sums_for([1.01] * 4), history)
            require_healthy_reference(metrics)
            self.assertIsNone(metrics['reference_health']['first_learned_epoch'])
            history.append(metrics)

    def test_single_scale_collapse_does_not_stop_early_but_fails_final(self):
        history = [summarize_reference_epoch(0, sums_for([0.3] * 4))]
        for epoch in (1, 2):
            metrics = summarize_reference_epoch(epoch, sums_for([0.2, 0.3, 0.4, 1.0]), history)
            require_healthy_reference(metrics)
            history.append(metrics)
        metrics = summarize_reference_epoch(3, sums_for([0.2, 0.3, 0.4, 1.0]), history, final=True)
        with self.assertRaisesRegex(ReferenceHealthError, 'fixed final'):
            require_healthy_reference(metrics)

    def test_threshold_equality_and_near_zero_final_are_rejected(self):
        for ratio in (0.98, 0.99, 1.0):
            with self.subTest(ratio=ratio):
                metrics = summarize_reference_epoch(19, sums_for([ratio] * 4), final=True)
                with self.assertRaises(ReferenceHealthError):
                    require_healthy_reference(metrics)
        metrics = summarize_reference_epoch(19, sums_for([0.9799] * 4), final=True)
        require_healthy_reference(metrics)

    def test_collapse_threshold_equality_counts_toward_patience(self):
        history = [summarize_reference_epoch(0, sums_for([0.2] * 4))]
        history.append(summarize_reference_epoch(1, sums_for([0.98] * 4), history))
        metrics = summarize_reference_epoch(2, sums_for([0.98] * 4), history)
        with self.assertRaises(ReferenceHealthError):
            require_healthy_reference(metrics)

    def test_recovery_or_missing_epoch_resets_consecutive_count(self):
        history = [summarize_reference_epoch(0, sums_for([0.2] * 4))]
        for epoch, ratios in ((1, [1.0] * 4), (2, [0.5] * 4),
                              (3, [1.0] * 4), (5, [1.0] * 4)):
            metrics = summarize_reference_epoch(epoch, sums_for(ratios), history)
            require_healthy_reference(metrics)
            self.assertLessEqual(metrics['reference_health']['consecutive_collapsed_epochs'], 1)
            history.append(metrics)

    def test_recorded_collapse_cannot_be_bypassed_by_resuming_later(self):
        history = [summarize_reference_epoch(0, sums_for([0.2] * 4))]
        for epoch in (1, 2):
            history.append(summarize_reference_epoch(epoch, sums_for([1.0] * 4), history))
        metrics = summarize_reference_epoch(3, sums_for([0.2] * 4), history)
        with self.assertRaises(ReferenceHealthError):
            require_healthy_reference(metrics)

    def test_original_metrics_history_is_sufficient_for_resume(self):
        history = [{'epoch': 4, 'mse_by_scale': [0.2] * 4,
                    'zero_predictor_mse_by_scale': [1.0] * 4,
                    'samples_by_scale': [100] * 4}]
        history.append(summarize_reference_epoch(5, sums_for([1.0] * 4), history))
        metrics = summarize_reference_epoch(6, sums_for([1.0] * 4), history)
        with self.assertRaises(ReferenceHealthError):
            require_healthy_reference(metrics)

    def test_nonfinite_statistics_fail_closed(self):
        for column in range(5):
            for invalid in (math.nan, math.inf, -math.inf):
                with self.subTest(column=column, invalid=invalid):
                    rows = sums_for([0.5] * 4)
                    rows[2][column] = invalid
                    with self.assertRaises((ValueError, FloatingPointError)):
                        summarize_reference_epoch(0, rows)

    def test_empty_or_fractional_scale_counts_fail_closed(self):
        for count in (0, -1, 0.5):
            rows = sums_for([0.5] * 4)
            rows[1][4] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                summarize_reference_epoch(0, rows)

    def test_missing_scales_and_malformed_rows_fail_closed(self):
        for rows in ([], sums_for([0.5] * 3), [[1, 1, 1, 1]] * 4):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_reference_epoch(0, rows)

    def test_negative_energy_or_nonpositive_noise_energy_fail_closed(self):
        for column, value in ((0, -1), (1, 0), (1, -1), (2, -1)):
            rows = sums_for([0.5] * 4)
            rows[0][column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                summarize_reference_epoch(0, rows)

    def test_statistics_use_aggregated_image_counts_and_normalized_cross_moment(self):
        rows = [[12, 24, 6, 9, 3], [8, 16, 4, 6, 2],
                [4, 8, 2, 3, 1], [20, 40, 10, 15, 5]]
        metrics = summarize_reference_epoch(0, rows)
        self.assertEqual(metrics['mse_by_scale'], [4.0] * 4)
        self.assertEqual(metrics['zero_predictor_mse_by_scale'], [8.0] * 4)
        self.assertEqual(metrics['mse_ratio_by_scale'], [0.5] * 4)
        self.assertEqual(metrics['prediction_energy_ratio_by_scale'], [0.25] * 4)
        for correlation in metrics['prediction_noise_correlation_by_scale']:
            self.assertAlmostEqual(correlation, 0.75)
        self.assertEqual(metrics['samples_by_scale'], [3, 2, 1, 5])

    def test_exact_zero_prediction_has_undefined_correlation(self):
        metrics = summarize_reference_epoch(0, sums_for([1.0] * 4))
        self.assertEqual(metrics['prediction_noise_correlation_by_scale'], [None] * 4)


class FinalReferenceProbeTests(unittest.TestCase):
    def setUp(self):
        self.config = {'epochs': 20, 'architecture_version': 'tect-reference-pixel-unet-v2-stable-paths',
                       'health_policy': 'REFERENCE-NOISE-HEALTH-v1', 'health_probe_images': 16,
                       'health_probe_seed': 42, 'micro_batch': 2}
        self.model_hash, self.manifest_hash = 'a' * 64, 'b' * 64
        metrics = summarize_reference_epoch(19, sums_for([0.2] * 4, [16] * 4), final=True)
        metrics['reference_health']['statistics_source'] = 'fixed_final_weights_training_reference_probe'
        self.probe = {
            'probe_id': 'REFERENCE-FIXED-FINAL-TRAINING-PROBE-v1',
            'model_parameter_hash': self.model_hash,
            'architecture_version': self.config['architecture_version'],
            'reference_manifest_sha256': self.manifest_hash,
            'selection': 'first_16_in_frozen_reference_manifest_order',
            'selected_reference_ids': [f'image-{i}' for i in range(16)],
            'image_count': 16, 'image_noise_observations_per_scale': 32,
            'seed': 42, 'micro_batch': 2, 'resolution': 512, 'lambdas': [2, 3, 4, 5],
            'replicas': 2,
            'noise_purposes': ['reference-final-probe-replica-0', 'reference-final-probe-replica-1'],
            'shared_noise_across_scales': True, 'training_augmentation': False,
            'ddp_padding': False, 'parameters_unchanged': True, 'rng_and_modes_restored': True,
            'checkpoint_selection': 'fixed_final_only_no_reselection',
            'scope': 'training_reference_images_only_no_validation_split_or_test_data',
            'metrics': metrics,
        }

    def verify(self, probe=None):
        require_final_reference_probe(self.probe if probe is None else probe, self.config,
                                      self.model_hash, self.manifest_hash)

    def test_valid_fixed_final_probe_passes(self):
        self.verify()

    def test_low_epoch_average_does_not_approve_final_zero_predictor(self):
        require_healthy_reference(summarize_reference_epoch(19, sums_for([0.1] * 4), final=True))
        self.probe['metrics'] = summarize_reference_epoch(19, sums_for([1.0] * 4, [16] * 4), final=True)
        self.probe['metrics']['reference_health']['statistics_source'] = 'fixed_final_weights_training_reference_probe'
        with self.assertRaises(ReferenceHealthError):
            self.verify()

    def test_wrong_or_malformed_model_hash_is_rejected(self):
        for value in ('c' * 64, 'not-a-hash'):
            with self.subTest(value=value), self.assertRaises(ReferenceHealthError):
                probe = copy.deepcopy(self.probe)
                probe['model_parameter_hash'] = value
                self.verify(probe)

    def test_forged_ratio_is_rejected_even_with_passing_health_flags(self):
        self.probe['metrics']['mse_by_scale'][-1] = 1.0
        self.probe['metrics']['mse_ratio_by_scale'][-1] = 0.2
        with self.assertRaises(ReferenceHealthError):
            self.verify()

    def test_misreported_ratio_is_rejected_even_when_true_ratio_passes(self):
        self.probe['metrics']['mse_ratio_by_scale'][0] = 0.1
        with self.assertRaises(ReferenceHealthError):
            self.verify()

    def test_coverage_and_nonfinite_statistics_fail_closed(self):
        probes = []
        for field, value in [('selected_reference_ids', ['same'] * 16), ('ddp_padding', True),
                             ('rng_and_modes_restored', False), ('reference_manifest_sha256', 'c' * 64)]:
            probe = copy.deepcopy(self.probe)
            probe[field] = value
            probes.append(probe)
        for field, value in [('samples_by_scale', [16, 16, 16, 15]),
                             ('mse_by_scale', [0.2, 0.2, math.nan, 0.2]), ('epoch', 18)]:
            probe = copy.deepcopy(self.probe)
            probe['metrics'][field] = value
            probes.append(probe)
        for probe in probes:
            with self.subTest(probe=probe), self.assertRaises(ReferenceHealthError):
                self.verify(probe)


if __name__ == '__main__':
    unittest.main()
