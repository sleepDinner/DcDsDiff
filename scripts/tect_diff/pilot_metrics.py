"""Strict two-set pilot selection, readable epoch records and fixed promotion gates.

The caller must first run aggregate_dataset(rows, expected_ids) for each fixed
manifest. Counts here defend that completed reduction; they cannot prove IDs
from aggregate values alone. Only rank zero writes the small epoch JSONL, while
holding the run lock, after the complete checkpoint transaction has succeeded.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile


TEST_NAMES = ('Casiav1', 'Columbia')
DEFAULT_GATE_THRESHOLDS = {
    'consecutive_epochs': 3,
    'dataset_f1_min': .35,
    'macro_f1_min': .50,
    'foreground_margin_min': .10,
    'specificity_min': .90,
    'recall_min': .40,
    'authentic_fp_rate_max': .01,
}


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _unit(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{label} must be finite in [0,1]')
    return float(value)


def _dataset(result, expected_count, label):
    if not isinstance(result, dict):
        raise ValueError(f'{label} reduction must be an object')
    _integer(expected_count, f'{label} expected count', 1)
    for key in ('count', 'id_unique_count'):
        if _integer(result.get(key), f'{label} {key}', 1) != expected_count:
            raise ValueError(f'{label} {key} does not match the registered image count')
    for key in ('pixel_f1', 'iou', 'boundary_f1', 'mae'):
        mean = _unit(result.get('dataset_' + key), f'{label} {key}')
        total = result.get(key + '_sum')
        if not isinstance(total, (int, float)) or not math.isfinite(total) or not math.isclose(
                total, mean * expected_count, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f'{label} {key} sum is inconsistent')
    for key in ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'ignored_pixels'):
        _integer(result.get(key), f'{label} {key}')
    if result['valid_pixels'] <= 0 or sum(result[key] for key in ('tp', 'fp', 'fn', 'tn')) != result['valid_pixels']:
        raise ValueError(f'{label} confusion counts do not cover the valid pixels')


def aggregate_test2(results, expected_counts):
    """Select only from complete, finite Casiav1 and Columbia dataset reductions."""
    if set(results) != set(TEST_NAMES) or set(expected_counts) != set(TEST_NAMES):
        raise ValueError('Test2 requires exactly Casiav1 and Columbia with registered counts')
    for name in TEST_NAMES:
        _dataset(results[name], expected_counts[name], name)
    summary = {'selection_protocol': 'test_selected', 'dataset_count': 2,
               'count': sum(expected_counts.values()), 'evaluation_complete': True}
    for metric in ('pixel_f1', 'iou', 'boundary_f1', 'mae'):
        summary['test2_macro_' + metric] = math.fsum(results[name]['dataset_' + metric] for name in TEST_NAMES) / 2
    for key in ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'ignored_pixels'):
        summary[key] = sum(results[name][key] for name in TEST_NAMES)
    tp, fp, fn, tn = (summary[key] for key in ('tp', 'fp', 'fn', 'tn'))
    denominator = 2 * tp + fp + fn
    summary['test2_pooled_pixel_f1'] = 2 * tp / denominator if denominator else 1.
    summary['specificity'] = tn / (tn + fp) if tn + fp else None
    summary['recall'] = tp / (tp + fn) if tp + fn else None
    return summary


def build_epoch_record(epoch, total_epochs, results, expected_counts, *, selection_authority,
                       selection_scope, training=None, health=None, selected=False):
    """Match the requested DINOv3 display fields without inventing All8 scores."""
    _integer(epoch, 'epoch')
    _integer(total_epochs, 'total_epochs', 1)
    if epoch >= total_epochs:
        raise ValueError('epoch must be smaller than total_epochs')
    if not isinstance(selection_authority, str) or not selection_authority.strip() or not isinstance(
            selection_scope, str) or not selection_scope.strip():
        raise ValueError('Selection authority and actual population scope are required')
    if type(selected) is not bool:
        raise ValueError('selected must be boolean')
    summary = aggregate_test2(results, expected_counts)
    row = {
        'epoch_index': epoch, 'epoch_number': epoch + 1, 'total_epochs': total_epochs,
        'metric': 'PixelF1(threshold=0.5, mode=origin)',
        'pixel_f1': {name: results[name]['dataset_pixel_f1'] for name in TEST_NAMES},
        'average_test2': summary['test2_macro_pixel_f1'],
        'selection_authority': selection_authority, 'selection_scope': selection_scope,
        'selection_metric': 'test2_macro_pixel_f1', 'selection_direction': 'maximize',
        'selection_protocol': 'test_selected', 'image_counts': dict(expected_counts),
        'dataset_metrics': results, 'summary': summary, 'selected': selected,
        'epoch_complete': True, 'evaluation_complete': True,
        'training': training or {}, 'health': health or {},
    }
    # Also detach nested caller-owned dictionaries so later mutation cannot
    # silently change a queued record or its gate evidence.
    return json.loads(json.dumps(row, allow_nan=False))


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def _record_index(row):
    if not isinstance(row, dict):
        raise ValueError('Each epoch record must be an object')
    epoch = _integer(row.get('epoch_index'), 'epoch_index')
    if row.get('epoch_complete') is not True or row.get('evaluation_complete') is not True:
        raise ValueError('Only complete epochs may enter metrics_per_epoch.jsonl')
    return epoch


def upsert_epoch_jsonl(path, record):
    """Single-writer atomic rewrite: idempotent epochs, no truncated-tail guessing.

    Invalid existing content, including a truncated final line, fails before any
    write. A valid final JSON object without a newline is safe to rewrite. This
    is deliberately not an append log or a cross-process concurrency primitive.
    """
    path = Path(path)
    new_index = _record_index(record)
    encoded = json.dumps(record, allow_nan=False)
    rows = {}
    if path.exists():
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            try:
                row = json.loads(line, parse_constant=_reject_constant)
                index = _record_index(row)
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid epoch JSONL at line {number}: {error}') from error
            if index in rows:
                raise ValueError(f'Duplicate epoch_index {index} in existing JSONL')
            rows[index] = row
    rows[new_index] = json.loads(encoded)
    payload = ''.join(json.dumps(rows[index], allow_nan=False, separators=(',', ':')) + '\n' for index in sorted(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _epoch_failures(row, baselines, thresholds, authentic_expected_count, expected_counts):
    reasons = []
    if row.get('epoch_complete') is not True or row.get('evaluation_complete') is not True:
        reasons.append('incomplete_epoch')
    health = row.get('health', {})
    if not isinstance(health, dict) or health.get('runtime_healthy') is not True:
        reasons.append('runtime_not_healthy')
        health = health if isinstance(health, dict) else {}
    try:
        datasets, counts = row['dataset_metrics'], row['image_counts']
        if counts != expected_counts:
            raise ValueError('image counts differ from the registered gate population')
        summary = aggregate_test2(datasets, counts)
        if summary['test2_macro_pixel_f1'] < thresholds['macro_f1_min']:
            reasons.append('macro_f1_below_minimum')
        for name in TEST_NAMES:
            result = datasets[name]
            f1 = result['dataset_pixel_f1']
            if f1 < thresholds['dataset_f1_min']:
                reasons.append(f'{name}:f1_below_minimum')
            if f1 < baselines[name] + thresholds['foreground_margin_min']:
                reasons.append(f'{name}:foreground_margin_below_minimum')
            negative, positive = result['tn'] + result['fp'], result['tp'] + result['fn']
            if not negative or result['tn'] / negative < thresholds['specificity_min']:
                reasons.append(f'{name}:specificity_below_minimum')
            if not positive or result['tp'] / positive < thresholds['recall_min']:
                reasons.append(f'{name}:recall_below_minimum')
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f'invalid_dataset_metrics:{error}')
    try:
        probe = health['authentic_probe']
        _dataset(probe, authentic_expected_count, 'authentic probe')
        if probe.get('authentic_count') != authentic_expected_count or probe['tp'] or probe['fn']:
            raise ValueError('probe is not the registered all-authentic population')
        mean_rate = _unit(probe.get('authentic_pixel_false_positive_rate'), 'authentic mean pixel FP rate')
        pooled_rate = probe['fp'] / probe['valid_pixels']
        if max(mean_rate, pooled_rate) > thresholds['authentic_fp_rate_max']:
            reasons.append('authentic_pixel_fp_above_maximum')
    except (KeyError, TypeError, ValueError) as error:
        reasons.append(f'invalid_authentic_probe:{error}')
    return reasons


def pilot_health_gate(records, all_foreground_baselines, *, thresholds=None, authentic_expected_count=64,
                      expected_counts=None):
    """Require the registered gates on the latest consecutive complete epochs.

    Baselines are mean per-image F1 for an all-foreground prediction, computed
    from the fixed masks before training. These are developmental test-selected
    gates; passing them is not independent generalization evidence. Set only
    consecutive_epochs=1 for the separately registered full-two-set confirmation.
    """
    if set(all_foreground_baselines) != set(TEST_NAMES):
        raise ValueError('All-foreground baselines require both registered test populations')
    baselines = {name: _unit(all_foreground_baselines[name], f'{name} baseline') for name in TEST_NAMES}
    active = dict(DEFAULT_GATE_THRESHOLDS)
    if thresholds is not None:
        if set(thresholds) - set(active):
            raise ValueError('Unknown pilot gate threshold')
        active.update(thresholds)
    required = _integer(active['consecutive_epochs'], 'consecutive_epochs', 1)
    _integer(authentic_expected_count, 'authentic_expected_count', 1)
    for name, value in active.items():
        if name != 'consecutive_epochs':
            _unit(value, name)
    rows = list(records)
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Each health evidence record must be an object')
    rows.sort(key=lambda row: _integer(row.get('epoch_index'), 'epoch_index'))
    indices = [row['epoch_index'] for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate epoch_index in pilot health evidence')
    latest = rows[-required:]
    if expected_counts is None:
        expected_counts = latest[0].get('image_counts', {}) if latest else {}
    elif set(expected_counts) != set(TEST_NAMES):
        raise ValueError('Expected gate counts require both registered populations')
    if expected_counts:
        for name, count in expected_counts.items():
            _integer(count, f'{name} expected count', 1)
    latest_indices = [row['epoch_index'] for row in latest]
    reasons = []
    if len(latest) != required:
        reasons.append('insufficient_complete_epochs')
    if latest_indices and latest_indices != list(range(latest_indices[-1] - len(latest) + 1, latest_indices[-1] + 1)):
        reasons.append('nonconsecutive_epochs')
    per_epoch = {}
    for row in latest:
        failures = _epoch_failures(row, baselines, active, authentic_expected_count, expected_counts)
        per_epoch[str(row['epoch_index'])] = failures
        reasons.extend(f'epoch_{row["epoch_index"]}:{failure}' for failure in failures)
    return {'passed': not reasons, 'reasons': reasons, 'epoch_failures': per_epoch,
            'consecutive_epochs': latest_indices, 'thresholds': active,
            'all_foreground_baselines': baselines, 'authentic_expected_count': authentic_expected_count,
            'expected_counts': dict(expected_counts),
            'selection_protocol': 'test_selected'}
