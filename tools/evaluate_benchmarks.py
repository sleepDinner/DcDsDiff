"""Evaluate the saved best-MAE checkpoint once on the frozen eight-set suite."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import site
import sys
import tempfile
import time

if __name__ == '__main__':
    os.environ['PYTHONNOUSERSITE'] = '1'
    if site.ENABLE_USER_SITE:
        os.execv(sys.executable, [sys.executable, '-s', *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from dataset.data_val import test_dataset
from tools.benchmark_protocol import SELECTION_THREE, load_benchmark, verify_benchmark_files
from tools.evaluate_reproduction import binary_metrics, file_hash
from tools.early_stop import validate_receipt
from tools.manage_experiment import now, read_json, write_json
from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.train_utils import set_random_seed


def validate_best_checkpoint(checkpoint, training, cfg, early_stop=None):
    if early_stop is not None:
        if (cfg.protocol_id != 'CASIA2-SEL3-ALL8-V1'
                or checkpoint['best_epoch'] != early_stop['best_epoch']
                or checkpoint['best_mae'] != early_stop['best_mae']
                or checkpoint['epoch'] > early_stop['last_complete_epoch']):
            raise ValueError('Best checkpoint differs from the registered early endpoint.')
    elif training['state'] != 'COMPLETED' or training['next_epoch'] != 100:
        raise ValueError('Wait for all 100 training epochs before evaluating the saved best.')
    if (checkpoint.get('checkpoint_format') != 2
            or checkpoint.get('selection') != 'TEST_SELECTED_MAE_DIAGNOSTIC'
            or checkpoint['epoch'] != checkpoint['best_epoch']
            or checkpoint['next_epoch'] != checkpoint['epoch'] + 1
            or not 0 <= checkpoint['epoch'] < 100
            or checkpoint['best_epoch'] != training['best_epoch']
            or not math.isclose(checkpoint['best_mae'], training['best_mae'], rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Checkpoint does not match the completed run\'s saved best MAE.')
    config = OmegaConf.to_container(cfg, resolve=True)
    for key, value in checkpoint['contract'].items():
        if key in ('train_num_epoch', 'mixed_precision'):
            continue
        if config.get(key) != value:
            raise ValueError(f'Checkpoint and frozen training configuration differ: {key}')
    if cfg.diffusion_model.params.num_sample_steps != 10 or cfg.diffusion_model.params.image_size != 352:
        raise ValueError('Benchmark evaluation preserves the original 352px/10-step inference.')


def aggregate(rows, names):
    keys = ('F1', 'IoU', 'MAE')
    metrics = {}
    for name in names:
        group = [r for r in rows if r['dataset'] == name]
        if not group:
            raise ValueError(f'Missing dataset results: {name}')
        metrics[name] = {'count': len(group), **{k: float(np.mean([r[k] for r in group])) for k in keys}}
    metrics['All8_macro'] = {'count': len(rows), **{k: float(np.mean([metrics[n][k] for n in names])) for k in keys}}
    metrics['All8_pooled'] = {'count': len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}
    return metrics


def selection_description(cfg, provenance):
    if cfg.protocol_id == 'CASIA2-SEL3-ALL8-V1':
        if list(cfg.test_dataset.Mix.params.datasets) != SELECTION_THREE:
            raise ValueError('Unexpected three-set checkpoint selection population.')
        return ('Casiav1/Columbia/NIST16 pooled 1664 images',
                'Best uses pooled MAE on these three datasets after the registered continuation boundary. '
                'Their scores are test-selected; the other five datasets are reporting-only after that boundary. '
                'The parent previously monitored All8; this is not a fresh three-set-only selection history.')
    if cfg.protocol_id == 'CASIA2-ALL8-V1':
        return ('All8 pooled 4295 images',
                'Saved best uses All8 pooled MAE; these scores are test-selected, not held-out estimates.')
    if cfg.protocol_id == 'GIT10K-PAPER-RECON-V1':
        return (f'GIT10K reconstructed Mix {provenance["dataset"]["test_count"]} images',
                'Saved best was selected on GIT10K Mix MAE; All8 is evaluated once and is not used to reselect this checkpoint.')
    raise ValueError('Unregistered checkpoint selection protocol.')


def validate_selection_origin(checkpoint, cfg, provenance):
    description = selection_description(cfg, provenance)
    if cfg.protocol_id == 'CASIA2-SEL3-ALL8-V1':
        continuation = provenance['continuation']
        if (continuation['selection_datasets'] != SELECTION_THREE or continuation['selection_count'] != 1664
                or checkpoint['epoch'] < continuation['next_epoch']):
            raise ValueError('Best checkpoint predates or differs from the three-set selection boundary.')
    return description


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--spec', type=Path, default=Path('config/benchmark_all8.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--early-stop-sha256', help='Hash of the explicitly registered run/early_stop.json.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    if args.output.exists():
        raise SystemExit('Evaluation output already exists; preserve the completed result.')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('Select exactly one free GPU before evaluation.')
    cfg = OmegaConf.load(run / 'resolved_config.yaml')
    spec, snapshot, receipt, manifest = load_benchmark(
        root, args.spec, require_disjoint_train=cfg.protocol_id in ('CASIA2-ALL8-V1', 'CASIA2-SEL3-ALL8-V1'))
    file_verification = verify_benchmark_files(snapshot, manifest, split='test')
    names = [x['name'] for x in spec['tests']]
    expected = {x['name']: x['count'] for x in spec['tests']}
    metadata = {r['name'] + '.png': r for r in manifest if r['split'] == 'test'}
    training = read_json(run / 'training_status.json')
    early_stop = validate_receipt(run, args.early_stop_sha256) if args.early_stop_sha256 else None
    provenance = read_json(run / 'provenance.json')
    for name, digest in provenance['source_files_sha256'].items():
        if file_hash(run / 'source' / name) != digest:
            raise ValueError(f'Frozen training source changed: {name}')
        if Path(name).suffix == '.py' and Path(name).parts[0] in (
                'model', 'dataset', 'utils', 'denoisingdiffusionpytorch') and file_hash(root / name) != digest:
            raise ValueError(f'Evaluation implementation differs from the frozen training implementation: {name}')
    checkpoint_hash = file_hash(run / 'model-best.pt')
    checkpoint = torch.load(run / 'model-best.pt', map_location='cpu', weights_only=False)
    validate_best_checkpoint(checkpoint, training, cfg, early_stop)
    selection_population, selection_note = validate_selection_origin(checkpoint, cfg, provenance)
    checkpoint_path = run / 'model-best.pt'
    selected_epoch = checkpoint['epoch']
    set_random_seed(0)
    cond = instantiate_from_config(cfg.cond_uvit,
        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    net = recurse_instantiate_from_config(cfg.model, unet=cond)
    model = instantiate_from_config(cfg.diffusion_model, model=net)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model = model.cuda().eval()
    dataset = test_dataset(**{key: str(snapshot / 'test' / kind) for key, kind in (
        ('image_root', 'f'), ('gt_root', 'm'), ('de_root', 'd'), ('trace_root', 't'))}, testsize=352)
    if {Path(p).stem + '.png' for p in dataset.images} != set(metadata):
        raise ValueError('Actual benchmark loader differs from the frozen manifest.')
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        collate_fn=collate)
    forward = get_obj_from_str(cfg.train_val_forward_fn)
    rows = []
    progress = args.output.with_name(args.output.name + '.progress.json')
    progress.parent.mkdir(parents=True, exist_ok=True)
    last_status = 0.0
    set_random_seed(0)
    for batch in loader:
        targets = [np.asarray(mask, dtype=np.float32) / 255.0 for mask in batch['gt']]
        predictions = forward(model, image=batch['image'].cuda().squeeze(1), trace=batch['trace'].cuda().squeeze(1),
                              time_ensemble=True, gt_sizes=[gt.shape for gt in targets], verbose=False)['pred_gt']
        if len(predictions) != len(targets):
            raise RuntimeError('Prediction count mismatch.')
        for name, prediction, target in zip(batch['name'], predictions, targets):
            pred = prediction.detach().cpu().numpy().squeeze()
            if pred.shape != target.shape or not np.isfinite(pred).all():
                raise RuntimeError(f'Invalid prediction: {name}')
            f1, iou = binary_metrics(pred, target)
            rows.append({'name': name, 'dataset': metadata[name]['dataset'], 'F1': f1, 'IoU': iou,
                         'MAE': float(np.abs(pred - target).mean())})
        if time.monotonic() - last_status > 60:
            write_json(progress, {'status': 'EVALUATING', 'updated_at': now(), 'images': len(rows),
                                  'total': len(dataset), 'checkpoint_sha256': checkpoint_hash})
            print(f'Evaluated {len(rows)}/{len(dataset)}', flush=True)
            last_status = time.monotonic()
    if dict(Counter(r['dataset'] for r in rows)) != expected or len({r['name'] for r in rows}) != len(rows):
        raise RuntimeError('Missing or duplicated benchmark predictions.')
    if file_hash(checkpoint_path) != checkpoint_hash:
        raise RuntimeError('Best checkpoint changed during evaluation.')
    report = {
        'status': 'COMPLETED', 'completed_at': now(), 'training_run_id': provenance['run_id'],
        'training_protocol_id': cfg.protocol_id, 'training_commit': provenance['git_commit'],
        'suite_id': spec['suite_id'], 'dataset_manifest_sha256': receipt['manifest_sha256'],
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_hash, 'epoch': selected_epoch,
        'selection': 'SAVED_BEST_TEST_MAE', 'selection_mae': training['best_mae'],
        'selection_population': selection_population,
        'selection_note': selection_note,
        'continuation': provenance.get('continuation'),
        'training_endpoint': early_stop if early_stop else {'status': 'COMPLETED', 'completed_epochs': 100},
        'sampling_seed': 0, 'sampling_steps': 10, 'threshold': .5,
        'aggregation': 'per-image F1/IoU/MAE; macro gives equal dataset weight, pooled gives equal image weight; empty/empty=1',
        'input_file_verification': file_verification,
        'metrics': aggregate(rows, names),
    }
    staging = Path(tempfile.mkdtemp(prefix=args.output.name + '.partial-', dir=args.output.parent))
    with (staging / 'per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    write_json(staging / 'results.json', report)
    lines = ['# Saved best checkpoint: eight benchmark datasets', '',
             f"Run: {provenance['run_id']}; best epoch: {selected_epoch}; selection: {report['selection_population']} MAE.",
             '', report['selection_note'], '', '| Dataset | Images | F1 | IoU | MAE |', '|---|---:|---:|---:|---:|']
    if early_stop:
        lines[2:2] = [f"Training ended early by user request: {early_stop['completed_epochs']}/100 complete epochs. "
                      f"Last complete epoch: {early_stop['last_complete_epoch']}. This is not a final99 result.", '']
    for name, values in report['metrics'].items():
        lines.append(f'| {name} | {values["count"]} | {values["F1"]:.6f} | {values["IoU"]:.6f} | {values["MAE"]:.6f} |')
    (staging / 'report.md').write_text('\n'.join(lines) + '\n')
    staging.rename(args.output)
    write_json(progress, {'status': 'COMPLETED', 'images': len(rows), 'updated_at': now()})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
