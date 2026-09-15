"""CPU-only recovery checks for model-bound, complete pilot evaluations."""
from pathlib import Path

from scripts.tect_diff.common import atomic_json, json_hash, read_json
from scripts.tect_diff.metrics import aggregate_dataset
from scripts.tect_diff.pilot_metrics import aggregate_test2, pilot_health_gate


def ensure_evaluation_binding(directory, binding):
    """Rank zero calls this before evaluation, then all ranks synchronize.

    Existing results without their original model/population binding are never
    adopted. Creating a new binding is idempotent, not a cross-process lock.
    """
    directory = Path(directory)
    if not isinstance(binding, dict) or not binding:
        raise ValueError('Evaluation binding must be a nonempty object')
    digest = json_hash(binding)
    path = directory/'evaluation_binding.json'
    if path.exists():
        if json_hash(read_json(path)) != digest:
            raise ValueError('Existing evaluation belongs to a different model or population')
        return
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Existing evaluation state has no model/population binding')
    atomic_json(path, binding)


def load_completed_dataset(directory, name, expected_ids):
    """Return a verified complete dataset or None; malformed files fail closed."""
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Dataset name must be a single path component')
    path = Path(directory)/f'{name}.json'
    if not path.exists():
        return None
    try:
        saved = read_json(path)
        if not isinstance(saved, dict) or not isinstance(saved.get('per_image'), list):
            raise ValueError('Expected a per-image result list')
        # NaN in an auxiliary field must not be carried forward either.
        json_hash(saved)
        metrics = aggregate_dataset(saved['per_image'], expected_ids)
        if json_hash(metrics) != json_hash(saved.get('metrics')):
            raise ValueError('Saved aggregate differs from its complete per-image rows')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid completed dataset {name}: {error}') from error
    return saved


def validate_confirmation(existing, binding, baselines, counts, thresholds, authcount):
    """Recompute the registered confirmation gate instead of trusting its flag."""
    if not isinstance(existing, dict) or not isinstance(binding, dict) or not binding:
        raise ValueError('Confirmation and model binding must be objects')
    json_hash(existing)
    if any(existing.get(key) != value for key, value in binding.items()):
        raise ValueError('Existing confirmation belongs to a different model')
    row = existing.get('metrics')
    if not isinstance(row, dict) or row.get('epoch_complete') is not True or row.get('evaluation_complete') is not True:
        raise ValueError('Existing confirmation does not contain a complete epoch')
    if row.get('epoch_index') != binding.get('epoch') or row.get('epoch_number') != binding.get('epoch', -2) + 1:
        raise ValueError('Confirmation metrics belong to a different epoch')
    for key in ('config_hash', 'source_commit', 'run_id'):
        if key in binding and row.get(key) != binding[key]:
            raise ValueError(f'Confirmation metric {key} differs from its binding')
    if row.get('image_counts') != counts:
        raise ValueError('Confirmation metrics use a different population')
    datasets = row.get('dataset_metrics', {})
    summary = aggregate_test2(datasets, counts)
    if (row.get('selection_protocol') != 'test_selected' or row.get('summary') != summary
            or row.get('average_test2') != summary['test2_macro_pixel_f1']
            or row.get('pixel_f1') != {name: datasets[name]['dataset_pixel_f1'] for name in counts}):
        raise ValueError('Confirmation display metrics differ from their complete reductions')
    gate = pilot_health_gate([row], baselines, thresholds=thresholds,
                             expected_counts=counts, authentic_expected_count=authcount)
    if json_hash(gate) != json_hash(existing.get('gate')):
        raise ValueError('Saved confirmation gate differs from recomputed evidence')
    return gate
