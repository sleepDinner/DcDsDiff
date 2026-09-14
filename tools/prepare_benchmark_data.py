"""Freeze paired CASIA2 training and the requested eight benchmark datasets."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import site
import sys

if __name__ == '__main__':
    os.environ['PYTHONNOUSERSITE'] = '1'
    if site.ENABLE_USER_SITE:
        os.execv(sys.executable, [sys.executable, '-s', *sys.argv])

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_git10k_aux import IMAGE_EXTENSIONS, make_detail_map, make_high_frequency_view


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def index_files(root, suffix=''):
    result = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if suffix:
            if not stem.endswith(suffix):
                raise ValueError(f'Unexpected mask suffix: {path}')
            stem = stem[:-len(suffix)]
        if stem in result:
            raise ValueError(f'Ambiguous image/mask stem: {path}')
        result[stem] = path
    return result


def collect_jobs(spec, destination):
    jobs = []
    for split, item in [('train', spec['train']), *[('test', x) for x in spec['tests']]]:
        root = Path(item['root']).resolve()
        images = index_files(root / item.get('images', 'images'))
        masks = index_files(root / item.get('masks', 'masks'), item['mask_suffix'])
        if set(images) != set(masks) or len(images) != item['count']:
            raise ValueError(f"{item['name']}: pairing/count mismatch; images={len(images)}, masks={len(masks)}, "
                             f"unmatched={sorted(set(images)^set(masks))[:10]}")
        for stem in sorted(images):
            jobs.append((str(destination), split, item['name'], stem, str(images[stem]), str(masks[stem])))
    return jobs


def prepare_one(job):
    destination, split, dataset, stem, image_path, mask_path = job
    cv2.setNumThreads(1)
    image_path, mask_path = Path(image_path), Path(mask_path)
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert('RGB'))
    with Image.open(mask_path) as mask:
        gt = np.asarray(mask.convert('L'))
    # Preserve the baseline's /255 soft-label convention. Explicit 0/1 masks
    # require scaling to the same 8-bit label range, recorded per sample.
    binary_unit_mask = bool(gt.max() == 1)
    if binary_unit_mask:
        gt = gt * np.uint8(255)
    name = dataset + '__' + stem
    base = Path(destination) / split
    image_dest = base / 'f' / (name + image_path.suffix.lower())
    mask_dest = base / 'm' / (name + ('.png' if binary_unit_mask else mask_path.suffix.lower()))
    image_dest.symlink_to(image_path)
    if binary_unit_mask:
        Image.fromarray(gt).save(mask_dest)
    else:
        mask_dest.symlink_to(mask_path)
    detail = base / 'd' / (name + '.png')
    trace = base / 't' / (name + '.png')
    Image.fromarray(make_detail_map(gt, radius=15, edge_kernel=3)).save(detail)
    Image.fromarray(make_high_frequency_view(rgb, cutoff_ratio=.5, boost=10)).save(trace)
    return {
        'name': name, 'dataset': dataset, 'split': split,
        'image_source': str(image_path), 'mask_source': str(mask_path),
        'image_file': image_dest.name, 'mask_file': mask_dest.name,
        'image_sha256': sha256(image_path), 'mask_source_sha256': sha256(mask_path),
        'mask_sha256': sha256(mask_dest),
        'rgb_pixels_sha256': hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest(),
        'image_hw': f'{rgb.shape[0]}x{rgb.shape[1]}', 'mask_hw': f'{gt.shape[0]}x{gt.shape[1]}',
        'size_mismatch': rgb.shape[:2] != gt.shape[:2], 'binary_unit_mask_scaled': binary_unit_mask,
        'soft_mask': bool(((gt != 0) & (gt != 255)).any()), 'empty_mask': not bool((gt > 127).any()),
        'detail_sha256': sha256(detail), 'trace_sha256': sha256(trace),
    }


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['rgb_pixels_sha256']].append(row)
    overlaps = [group for group in groups.values() if {r['split'] for r in group} == {'train', 'test'}]
    return {
        'counts': dict(Counter(row['dataset'] for row in rows)),
        'train_count': sum(row['split'] == 'train' for row in rows),
        'test_count': sum(row['split'] == 'test' for row in rows),
        'anomalies_by_dataset': {name: {key: sum(row[key] for row in rows if row['dataset'] == name)
            for key in ('size_mismatch', 'binary_unit_mask_scaled', 'soft_mask', 'empty_mask')}
            for name in dict.fromkeys(row['dataset'] for row in rows)},
        'cross_train_test_identical_rgb_groups': len(overlaps),
        'cross_train_test_identical_rgb_samples': [[r['name'] for r in group] for group in overlaps],
        'duplicate_rgb_groups': sum(len(group) > 1 for group in groups.values()),
        'sample_policy': 'retain all paired tampered samples; authentic CASIA2 Au images are not added; no score-based exclusions',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=Path('config/benchmark_all8.json'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root = args.project.resolve()
    spec = json.loads(args.spec.read_text())
    destination = (root / spec['snapshot']).resolve()
    if not destination.is_relative_to(root / 'data') or destination == root / 'data':
        raise SystemExit('Snapshot must be inside project/data.')
    if destination.exists():
        raise SystemExit(f'Refusing to overwrite snapshot: {destination}')
    jobs = collect_jobs(spec, destination)
    for split in ('train', 'test'):
        for kind in ('f', 'm', 'd', 't'):
            (destination / split / kind).mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for row in pool.map(prepare_one, jobs, chunksize=1):
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f'prepared {len(rows)}/{len(jobs)} ({row["dataset"]})', flush=True)
    manifest = destination / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        'status': 'READY', 'suite_id': spec['suite_id'], 'spec_sha256': sha256(args.spec),
        'manifest_sha256': sha256(manifest), **summarize(rows),
        'auxiliary_parameters': {'detail_radius': 15, 'edge_kernel': 3, 'cutoff_ratio': .5, 'boost': 10},
        'generator_sha256': sha256(Path(__file__).with_name('generate_git10k_aux.py')),
        'preparation_script_sha256': sha256(__file__),
        'preparation_environment': {'python_prefix': sys.prefix, 'user_site_enabled': site.ENABLE_USER_SITE,
                                  'opencv': cv2.__version__, 'numpy': np.__version__, 'pillow': pillow_version},
    }
    (destination / 'dataset_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
