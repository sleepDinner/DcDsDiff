"""Evaluate the predeclared final checkpoint, with no threshold/checkpoint search."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import site

if __name__ == '__main__':
    os.environ['PYTHONNOUSERSITE'] = '1'
    if site.ENABLE_USER_SITE:
        os.execv(sys.executable, [sys.executable, '-s', *sys.argv])

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.train_utils import set_random_seed
from tools.generate_git10k_aux import prefix_from_stem


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binary_metrics(pred, target):
    pred, target = np.asarray(pred) > 0.5, np.asarray(target) > 0.5
    tp = np.count_nonzero(pred & target)
    fp = np.count_nonzero(pred & ~target)
    fn = np.count_nonzero(~pred & target)
    f1_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn
    return (2 * tp / f1_den if f1_den else 1.0,
            tp / iou_den if iou_den else 1.0)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-epoch', type=int, default=99)
    parser.add_argument('--expected-count', type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Evaluation output exists; use a new output path to preserve results.')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the reproduction evaluation.')
    cfg = OmegaConf.load(args.config)
    if cfg.diffusion_model.params.num_sample_steps != 10:
        raise SystemExit('This protocol requires exactly 10 sampling steps.')
    checkpoint_hash = file_hash(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != args.expected_epoch:
        raise SystemExit(f'Unexpected endpoint epoch: {checkpoint["epoch"]}')
    if (checkpoint.get('checkpoint_format') != 2
            or checkpoint.get('next_epoch') != args.expected_epoch + 1
            or checkpoint.get('selection') != 'FIXED_FINAL_EPOCH'):
        raise SystemExit('Checkpoint is not a completed fixed-final endpoint.')
    set_random_seed(0)
    cond = instantiate_from_config(cfg.cond_uvit,
        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    net = recurse_instantiate_from_config(cfg.model, unet=cond)
    model = instantiate_from_config(cfg.diffusion_model, model=net)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model = model.cuda().eval()
    dataset = instantiate_from_config(cfg.test_dataset.Mix)
    if len(dataset) != args.expected_count:
        raise SystemExit(f'Unexpected test size: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate)
    forward = get_obj_from_str(cfg.train_val_forward_fn)
    rows = []
    set_random_seed(0)
    for batch in loader:
        targets = [np.asarray(mask, dtype=np.float32) / 255.0 for mask in batch['gt']]
        outputs = forward(model, image=batch['image'].cuda().squeeze(1),
                          trace=batch['trace'].cuda().squeeze(1), time_ensemble=True,
                          gt_sizes=[gt.shape for gt in targets], verbose=False)['pred_gt']
        if len(outputs) != len(targets):
            raise RuntimeError('Prediction count does not match ground truth.')
        for name, pred, gt in zip(batch['name'], outputs, targets):
            pred = pred.detach().cpu().numpy().squeeze()
            if pred.shape != gt.shape or not np.isfinite(pred).all():
                raise RuntimeError(f'Invalid prediction for {name}')
            f1, iou = binary_metrics(pred, gt)
            rows.append({'name': name, 'prefix': prefix_from_stem(Path(name).stem),
                         'F1': f1, 'IoU': iou, 'MAE': float(np.abs(pred - gt).mean())})
        if len(rows) % 60 == 0:
            print(f'evaluated {len(rows)}/{len(dataset)}', flush=True)
    if len(rows) != len(dataset) or len({r['name'] for r in rows}) != len(dataset):
        raise RuntimeError('Incomplete or duplicate evaluation records.')
    groups = defaultdict(list)
    groups['Mix'] = rows
    for row in rows:
        groups[row['prefix']].append(row)
    results = {key: {'count': len(group), **{metric: float(np.mean([r[metric] for r in group]))
                for metric in ('F1', 'IoU', 'MAE')}} for key, group in groups.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=args.output.name + '.partial-', dir=args.output.parent))
    with (staging / 'per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'status': 'COMPLETED', 'protocol_id': cfg.protocol_id,
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
        'epoch': args.expected_epoch, 'selection': 'PREDECLARED_FINAL_EPOCH',
        'sampling_seed': 0, 'sampling_steps': 10, 'threshold': 0.5,
        'aggregation': 'per-image F1/IoU then arithmetic mean; empty/empty=1',
        'postprocessing': 'upstream temporal mean, per-image minmax and positive-logit majority vote',
        'scope': 'reconstructed split; prefixes are filenames, not confirmed paper generator categories',
        'metrics': results,
    }
    (staging / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Final checkpoint evaluation', '',
             f'Protocol: {cfg.protocol_id}; epoch {args.expected_epoch}; threshold 0.5.', '',
             'This uses a reconstructed split. Prefix rows are not a verified mapping to the paper generators.', '',
             '| Split / filename prefix | Images | F1 | IoU | MAE |', '|---|---:|---:|---:|---:|']
    for key, row in results.items():
        lines.append(f'| {key} | {row["count"]} | {row["F1"]:.6f} | {row["IoU"]:.6f} | {row["MAE"]:.6f} |')
    (staging / 'report.md').write_text('\n'.join(lines) + '\n')
    staging.rename(args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
