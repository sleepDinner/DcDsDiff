"""Create an immutable, auditable GIT10K reconstruction without editing sources."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import site

if __name__ == '__main__':
    os.environ['PYTHONNOUSERSITE'] = '1'
    if site.ENABLE_USER_SITE:
        os.execv(sys.executable, [sys.executable, '-s', *sys.argv])

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_git10k_aux import (
    make_detail_map, make_high_frequency_view, natural_key, prefix_from_stem,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_one(job):
    source, target, stem, split, reuse_root, cached_row = job
    source, target = Path(source), Path(target)
    image_path = source / 'Image' / (stem + '.png')
    mask_path = source / 'Mask' / (stem + '.png')
    with Image.open(image_path) as img:
        image = np.asarray(img.convert('RGB'))
    with Image.open(mask_path) as img:
        mask = np.asarray(img.convert('L'))
    pixel_hash = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
    image_hash, mask_hash = sha256(image_path), sha256(mask_path)
    if cached_row is not None and (pixel_hash != cached_row['rgb_pixels_sha256']
            or image_hash != cached_row['image_sha256'] or mask_hash != cached_row['mask_sha256']):
        raise RuntimeError(f'Raw data changed since cached content audit: {stem}')
    base = target / split
    (base / 'f' / image_path.name).symlink_to(image_path)
    (base / 'm' / mask_path.name).symlink_to(mask_path)
    auxiliary = {'d': make_detail_map(mask, radius=15.0, edge_kernel=3),
                 't': make_high_frequency_view(image, cutoff_ratio=0.5, boost=10.0)}
    reused = 0
    for kind, pixels in auxiliary.items():
        destination = base / kind / image_path.name
        prior = Path(reuse_root) / cached_row['split'] / kind / image_path.name if reuse_root else None
        equal = False
        if prior is not None and prior.is_file():
            with Image.open(prior) as old_image:
                equal = np.array_equal(np.asarray(old_image), pixels)
        if equal:
            os.link(prior, destination)
            reused += 1
        else:
            Image.fromarray(pixels).save(destination)
    return {
        'stem': stem, 'prefix': prefix_from_stem(stem), 'split': split,
        'image_sha256': image_hash, 'mask_sha256': mask_hash,
        'rgb_pixels_sha256': pixel_hash,
        'image_hw': f'{image.shape[0]}x{image.shape[1]}',
        'mask_hw': f'{mask.shape[0]}x{mask.shape[1]}',
        'size_mismatch': image.shape[:2] != mask.shape[:2],
        'empty_mask': not bool((mask > 127).any()),
        'soft_mask': bool(((mask != 0) & (mask != 255)).any()),
        'detail_sha256': sha256(base / 'd' / mask_path.name),
        'trace_sha256': sha256(base / 't' / image_path.name),
        'reused_auxiliary_files': reused,
    }


def rgb_hash(job):
    stem, source = job
    with Image.open(Path(source) / 'Image' / (stem + '.png')) as image:
        pixels = np.asarray(image.convert('RGB'))
    return stem, hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()


def grouped_split(stems, hashes, seed):
    groups = defaultdict(list)
    for stem in sorted(stems, key=natural_key):
        groups[hashes[stem]].append(stem)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    train_stems = set()
    remaining = 9000
    for group in groups:
        if len(group) <= remaining:
            train_stems.update(group)
            remaining -= len(group)
    if remaining:
        raise SystemExit('Cannot allocate exactly 9000 images without splitting content groups.')
    return train_stems


def verify_one(job):
    source, output, row = job
    source, output = Path(source), Path(output)
    name = row['stem'] + '.png'
    base = output / row['split']
    paths = {'image_sha256': source / 'Image' / name, 'mask_sha256': source / 'Mask' / name,
             'detail_sha256': base / 'd' / name, 'trace_sha256': base / 't' / name}
    for key, path in paths.items():
        if sha256(path) != row[key]:
            return row['stem'] + ': ' + key + ' drift'
    with Image.open(paths['image_sha256']) as image:
        rgb = np.asarray(image.convert('RGB'))
    with Image.open(paths['mask_sha256']) as mask:
        gt = np.asarray(mask.convert('L'))
    with Image.open(paths['detail_sha256']) as detail:
        if not np.array_equal(np.asarray(detail), make_detail_map(gt, radius=15, edge_kernel=3)):
            return row['stem'] + ': detail pixel mismatch'
    with Image.open(paths['trace_sha256']) as trace:
        if not np.array_equal(np.asarray(trace), make_high_frequency_view(rgb, cutoff_ratio=.5, boost=10)):
            return row['stem'] + ': trace pixel mismatch'
    for kind, path in (('f', paths['image_sha256']), ('m', paths['mask_sha256'])):
        if (base / kind / name).resolve() != path.resolve():
            return row['stem'] + ': input link mismatch'
    return None


def verify_snapshot(source, output, workers):
    receipt_file = output / 'dataset_receipt.json'
    receipt = json.loads(receipt_file.read_text())
    if sha256(output / 'manifest.csv') != receipt['manifest_sha256']:
        raise SystemExit('Manifest hash drift; refusing verification.')
    with (output / 'manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(str(source), str(output), row) for row in rows]
        for index, error in enumerate(pool.map(verify_one, jobs, chunksize=8), 1):
            if error:
                errors.append(error)
            if index % 500 == 0:
                print(f'verified {index}/{len(rows)}', flush=True)
    receipt['auxiliary_verification'] = {
        'status': 'PASS' if not errors else 'FAIL', 'count': len(rows),
        'check': 'all f/m/d/t file hashes and exact d/t pixel equality after regeneration',
        'python_prefix': sys.prefix, 'user_site_enabled': site.ENABLE_USER_SITE,
        'opencv': cv2.__version__, 'opencv_path': cv2.__file__,
        'numpy': np.__version__, 'pillow': pillow_version, 'errors': errors[:20],
    }
    if errors:
        receipt['status'] = 'FAILED_AUXILIARY_VERIFICATION'
    temporary = receipt_file.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(receipt_file)
    print(json.dumps(receipt['auxiliary_verification'], indent=2), flush=True)
    if errors:
        raise SystemExit('Auxiliary verification failed; training remains blocked.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify-output', action='store_true',
                        help='Verify every existing input/hash/auxiliary pixel in the current environment.')
    parser.add_argument('--reuse-from', type=Path,
                        help='Reuse previously audited source hashes and pixel-identical auxiliary files; never changes it.')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise SystemExit('Output must be separate from the read-only source dataset.')
    if args.verify_output:
        verify_snapshot(source, output, args.workers)
        return
    images = {p.stem for p in (source / 'Image').glob('*.png')}
    masks = {p.stem for p in (source / 'Mask').glob('*.png')}
    if images != masks or len(images) != 10000:
        raise SystemExit(f'Expected 10000 exactly paired PNGs: images={len(images)}, masks={len(masks)}')
    if output.exists():
        raise SystemExit(f'Refusing to overwrite a dataset snapshot: {output}')
    stems = sorted(images, key=natural_key)
    cached_rows = {}
    reuse = args.reuse_from.resolve() if args.reuse_from else None
    if reuse:
        if reuse == output or reuse in output.parents or output in reuse.parents:
            raise SystemExit('Reuse snapshot must be separate from output.')
        receipt = json.loads((reuse / 'dataset_receipt.json').read_text())
        if sha256(reuse / 'manifest.csv') != receipt['manifest_sha256']:
            raise SystemExit('Cached manifest hash mismatch.')
        with (reuse / 'manifest.csv').open(newline='') as stream:
            cached_rows = {row['stem']: row for row in csv.DictReader(stream)}
        if set(cached_rows) != images:
            raise SystemExit('Cached snapshot population differs from the source.')
        hashes = {stem: row['rgb_pixels_sha256'] for stem, row in cached_rows.items()}
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            hashes = dict(pool.map(rgb_hash, [(s, str(source)) for s in stems], chunksize=8))
    train_stems = grouped_split(stems, hashes, args.seed)
    for split in ('train', 'test'):
        for kind in ('f', 'm', 'd', 't'):
            (output / split / kind).mkdir(parents=True, exist_ok=True)
    jobs = [(str(source), str(output), s, 'train' if s in train_stems else 'test',
             str(reuse) if reuse else None, cached_rows.get(s))
            for s in sorted(images, key=natural_key)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(prepare_one, jobs, chunksize=8):
            rows.append(row)
            if len(rows) % 500 == 0:
                print(f'prepared {len(rows)}/10000', flush=True)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        # Keep the scientific manifest identical across fresh generation, cache
        # reuse and Git checkouts on different operating systems.
        fields = [key for key in rows[0] if key != 'reused_auxiliary_files']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    train_hashes = {r['rgb_pixels_sha256'] for r in rows if r['split'] == 'train'}
    test_hashes = {r['rgb_pixels_sha256'] for r in rows if r['split'] == 'test'}
    overlap = train_hashes & test_hashes
    summary = {
        'protocol_id': 'GIT10K-PAPER-RECON-V1',
        'source': str(source), 'seed': args.seed,
        'split_policy': 'natural-sort stems; group identical decoded RGB; Random(seed) shuffle groups; whole groups fill 9000 train',
        'official_split': False, 'official_generator_mapping_available': False,
        'count': len(rows), 'train_count': 9000, 'test_count': 1000,
        'prefix_counts': {split: dict(Counter(r['prefix'] for r in rows if r['split'] == split))
                          for split in ('train', 'test')},
        'size_mismatches': sum(r['size_mismatch'] for r in rows),
        'soft_masks': sum(r['soft_mask'] for r in rows),
        'empty_masks': sum(r['empty_mask'] for r in rows),
        'cross_split_identical_rgb_images': len(overlap),
        'distinct_rgb_images': len(train_hashes | test_hashes),
        'duplicate_content_groups': sum(count > 1 for count in Counter(r['rgb_pixels_sha256'] for r in rows).values()),
        'reused_auxiliary_files': sum(r['reused_auxiliary_files'] for r in rows),
        'auxiliary_parameters': {'detail_radius': 15, 'edge_kernel': 3, 'cutoff_ratio': 0.5, 'boost': 10},
        'source_image_and_mask_policy': 'read-only symlinks; resize both independently to 352 for training',
        'manifest_sha256': sha256(output / 'manifest.csv'),
        'generator_sha256': sha256(Path(__file__).with_name('generate_git10k_aux.py')),
        'preparation_environment': {'python_prefix': sys.prefix, 'user_site_enabled': site.ENABLE_USER_SITE,
                                    'opencv': cv2.__version__, 'opencv_path': cv2.__file__,
                                    'numpy': np.__version__, 'pillow': pillow_version},
        'auxiliary_verification': {'status': 'PASS', 'count': len(rows),
                                  'check': 'all d/t constructed in isolated pinned environment; reused files checked pixel-by-pixel'},
        'status': 'FAILED_DUPLICATE_LEAKAGE' if overlap else 'READY',
    }
    (output / 'dataset_receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if overlap:
        raise SystemExit('Identical RGB content crosses split; training is blocked. See dataset receipt.')


if __name__ == '__main__':
    main()
