"""Training-only, fixed-endpoint diagnostics for reference noise prediction.

No model calls, checkpoint selection, or test data enter this helper. The caller
enables this policy only in its explicitly registered repair configuration.
"""
from __future__ import annotations

import math


POLICY_ID = 'REFERENCE-NOISE-HEALTH-v1'
SCALE_COUNT = 4
LEARNING_RATIO_THRESHOLD = 0.98
COLLAPSE_RATIO_THRESHOLD = 0.98
COLLAPSE_PATIENCE_EPOCHS = 2
FINAL_RATIO_THRESHOLD = 0.98
FINAL_PROBE_ID = 'REFERENCE-FIXED-FINAL-TRAINING-PROBE-v1'
FINAL_PROBE_NOISE_PURPOSES = ('reference-final-probe-replica-0', 'reference-final-probe-replica-1')


class ReferenceHealthError(RuntimeError):
    """The reference cannot advance under the registered health policy."""


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError(f'Non-finite reference {label}')
    return value


def _epoch(value):
    number = _finite(value, 'epoch')
    if number < 0 or not number.is_integer():
        raise ValueError('Reference epoch must be a nonnegative integer')
    return int(number)


def _ratios(metrics):
    """Also accepts original history entries that predate this diagnostic."""
    mse = metrics['mse_by_scale']
    zero = metrics['zero_predictor_mse_by_scale']
    counts = metrics['samples_by_scale']
    if not all(len(values) == SCALE_COUNT for values in (mse, zero, counts)):
        raise ValueError('Reference history must contain every registered scale')
    ratios = []
    for k in range(SCALE_COUNT):
        count = _finite(counts[k], f'scale {k} sample count')
        error = _finite(mse[k], f'scale {k} MSE')
        baseline = _finite(zero[k], f'scale {k} zero MSE')
        if count <= 0 or not count.is_integer() or error < 0 or baseline <= 0:
            raise ValueError(f'Invalid reference moments or sample count at scale {k}')
        ratios.append(_finite(error / baseline, f'scale {k} MSE ratio'))
    return ratios


def summarize_reference_epoch(epoch, sums, history=(), *, final=False):
    """Summarize all-rank image-mean sums and return JSON-safe health metadata.

    ``sums`` has four scale rows, each containing
    ``[MSE_sum, zero_MSE_sum, prediction_energy_sum, prediction_noise_cross_sum,
    sample_count]``. Each image contributes its own channel/pixel mean before
    summation; the caller performs DDP all_reduce before calling this function.
    ``history`` contains completed earlier epoch metrics, excluding this epoch.

    Correlation is the uncentered normalized cross moment, not Pearson
    correlation. It is ``None`` when prediction energy is exactly zero.
    A failed return remains recordable; call ``require_healthy_reference`` after
    recording it and before advancing training or writing a final artifact.
    Invalid/non-finite moments raise immediately instead of yielding NaN JSON.
    """
    epoch = _epoch(epoch)
    if len(sums) != SCALE_COUNT or any(len(row) != 5 for row in sums):
        raise ValueError('Reference sums must contain four scales and five columns')
    means, counts, correlations = [], [], []
    for k, raw_row in enumerate(sums):
        error, zero, energy, cross, count = [
            _finite(value, f'scale {k} column {column}')
            for column, value in enumerate(raw_row)
        ]
        if count <= 0 or not count.is_integer():
            raise ValueError(f'Reference scale {k} requires a positive integer sample count')
        if error < 0 or zero <= 0 or energy < 0:
            raise ValueError(f'Invalid reference energy at scale {k}')
        if energy == 0:
            if cross != 0:
                raise ValueError(f'Zero prediction energy has nonzero cross moment at scale {k}')
            correlation = None
        else:
            correlation = _finite(cross / math.sqrt(energy) / math.sqrt(zero),
                                  f'scale {k} prediction-noise correlation')
        correlations.append(correlation)
        means.append([value / count for value in (error, zero, energy, cross)])
        counts.append(int(count))
    metrics = {
        'epoch': epoch,
        'mse_by_scale': [row[0] for row in means],
        'zero_predictor_mse_by_scale': [row[1] for row in means],
        'prediction_energy_by_scale': [row[2] for row in means],
        'prediction_noise_cross_by_scale': [row[3] for row in means],
        'samples_by_scale': counts,
        'prediction_noise_correlation_by_scale': correlations,
    }
    metrics['mse_ratio_by_scale'] = _ratios(metrics)
    metrics['prediction_energy_ratio_by_scale'] = [
        _finite(row[2] / row[1], f'scale {k} prediction energy ratio')
        for k, row in enumerate(means)
    ]
    learned_epoch, collapsed_epoch, consecutive, previous_epoch = None, None, 0, None
    for entry in [*history, metrics]:
        entry_epoch = _epoch(entry['epoch'])
        if previous_epoch is not None and entry_epoch <= previous_epoch:
            raise ValueError('Reference history must have strictly increasing earlier epochs')
        if previous_epoch is not None and entry_epoch != previous_epoch + 1:
            consecutive = 0
        ratios = _ratios(entry)
        if learned_epoch is None and all(ratio < LEARNING_RATIO_THRESHOLD for ratio in ratios):
            learned_epoch = entry_epoch
        collapsed = learned_epoch is not None and all(ratio >= COLLAPSE_RATIO_THRESHOLD for ratio in ratios)
        consecutive = consecutive + 1 if collapsed else 0
        if collapsed_epoch is None and consecutive >= COLLAPSE_PATIENCE_EPOCHS:
            collapsed_epoch = entry_epoch
        previous_epoch = entry_epoch
    final_pass = all(ratio < FINAL_RATIO_THRESHOLD for ratio in metrics['mse_ratio_by_scale'])
    reason = None
    if collapsed_epoch is not None:
        reason = f'Reference noise prediction collapsed at epoch {collapsed_epoch} after learning'
    elif final and not final_pass:
        reason = 'Reference fixed final endpoint must beat zero prediction on every scale by the registered margin'
    state = ('FAILED' if reason else 'DEGRADATION_WARNING' if consecutive
             else 'LEARNED' if final_pass else 'LEARNING')
    metrics['reference_health'] = {
        'policy_id': POLICY_ID,
        'state': state,
        'failed': reason is not None,
        'failure_reason': reason,
        'first_learned_epoch': learned_epoch,
        'consecutive_collapsed_epochs': consecutive,
        'collapse_detected_epoch': collapsed_epoch,
        'final_endpoint_checked': bool(final),
        'all_scales_pass_final_margin': final_pass,
        'learning_ratio_threshold': LEARNING_RATIO_THRESHOLD,
        'collapse_ratio_threshold': COLLAPSE_RATIO_THRESHOLD,
        'collapse_patience_epochs': COLLAPSE_PATIENCE_EPOCHS,
        'final_ratio_threshold': FINAL_RATIO_THRESHOLD,
        'correlation_definition': 'uncentered_cross_over_sqrt_prediction_energy_times_noise_energy',
        'statistics_source': 'training_images_only_all_rank_per_image_mean_sums',
        'checkpoint_selection': 'fixed_final_only_no_reselection',
    }
    return metrics


def require_healthy_reference(metrics):
    """Enforce the recorded diagnostic before advancing dependent work."""
    health = metrics['reference_health']
    if health['failed']:
        raise ReferenceHealthError(health['failure_reason'])


def require_final_reference_probe(probe, reference_config, final_parameter_hash, reference_manifest_sha256):
    """Validate the bound, actual-final-weight probe before accepting reference.

    This verifies a receipt, not the model itself. The controller must separately
    compare the ordered IDs with the frozen reference manifest's first 16 rows.
    Reported ratios and success flags never replace recomputed raw-moment checks.
    """
    def require(condition, message):
        if not condition:
            raise ReferenceHealthError(f'Invalid final reference probe: {message}')

    def fields_match(actual, expected, label):
        for key, value in expected.items():
            require(key in actual and type(actual[key]) is type(value) and actual[key] == value,
                    f'{label}.{key} differs from the registered value')

    def hash_matches(actual, expected, label):
        for value in (actual, expected):
            require(isinstance(value, str) and len(value) == 64
                    and all(char in '0123456789abcdefABCDEF' for char in value),
                    f'{label} must be a 64-character hexadecimal hash')
        require(actual.lower() == expected.lower(), f'{label} is not bound to the final artifact')

    try:
        require(isinstance(probe, dict) and isinstance(reference_config, dict), 'missing probe or config')
        fields_match(reference_config, {
            'epochs': 20, 'health_policy': POLICY_ID, 'health_probe_images': 16,
            'health_probe_seed': 42, 'micro_batch': 2,
        }, 'config')
        architecture = reference_config['architecture_version']
        require(isinstance(architecture, str) and bool(architecture.strip()), 'missing architecture version')
        fields_match(probe, {
            'probe_id': FINAL_PROBE_ID, 'architecture_version': architecture,
            'selection': 'first_16_in_frozen_reference_manifest_order',
            'image_count': 16, 'image_noise_observations_per_scale': 32,
            'seed': 42, 'micro_batch': 2, 'resolution': 512,
            'lambdas': [2, 3, 4, 5], 'replicas': 2,
            'noise_purposes': list(FINAL_PROBE_NOISE_PURPOSES),
            'shared_noise_across_scales': True, 'training_augmentation': False,
            'ddp_padding': False, 'parameters_unchanged': True, 'rng_and_modes_restored': True,
            'checkpoint_selection': 'fixed_final_only_no_reselection',
            'scope': 'training_reference_images_only_no_validation_split_or_test_data',
        }, 'probe')
        hash_matches(probe['model_parameter_hash'], final_parameter_hash, 'model_parameter_hash')
        hash_matches(probe['reference_manifest_sha256'], reference_manifest_sha256, 'reference_manifest_sha256')
        ids = probe['selected_reference_ids']
        require(isinstance(ids, list) and len(ids) == 16
                and all(isinstance(value, str) and bool(value.strip()) for value in ids)
                and len(set(ids)) == 16, 'expected exactly 16 unique image IDs')
        metrics = probe['metrics']
        fields_match(metrics, {'epoch': reference_config['epochs'] - 1, 'samples_by_scale': [16] * 4}, 'metrics')
        health = metrics['reference_health']
        fields_match(health, {
            'policy_id': POLICY_ID, 'final_endpoint_checked': True, 'failed': False,
            'failure_reason': None, 'all_scales_pass_final_margin': True,
            'final_ratio_threshold': FINAL_RATIO_THRESHOLD,
            'statistics_source': 'fixed_final_weights_training_reference_probe',
            'checkpoint_selection': 'fixed_final_only_no_reselection',
        }, 'health')
        moment_fields = ('mse_by_scale', 'zero_predictor_mse_by_scale',
                         'prediction_energy_by_scale', 'prediction_noise_cross_by_scale')
        for key in moment_fields:
            require(isinstance(metrics[key], list) and len(metrics[key]) == SCALE_COUNT,
                    f'{key} has incomplete scale coverage')
        sums = [[_finite(metrics[key][scale], key) * 16 for key in moment_fields] + [16]
                for scale in range(SCALE_COUNT)]
        recomputed = summarize_reference_epoch(metrics['epoch'], sums, final=True)
        require_healthy_reference(recomputed)
        for key in ('mse_ratio_by_scale', 'prediction_energy_ratio_by_scale',
                    'prediction_noise_correlation_by_scale'):
            require(isinstance(metrics[key], list) and len(metrics[key]) == SCALE_COUNT,
                    f'{key} has incomplete scale coverage')
            for reported, expected in zip(metrics[key], recomputed[key]):
                if expected is None:
                    require(reported is None, f'{key} must be undefined for zero prediction energy')
                else:
                    require(math.isclose(_finite(reported, key), expected, rel_tol=1e-12, abs_tol=1e-12),
                            f'{key} differs from recomputed raw moments')
    except (KeyError, TypeError, ValueError, FloatingPointError, OverflowError) as error:
        raise ReferenceHealthError(f'Invalid final reference probe: {error}') from error
