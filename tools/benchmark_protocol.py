"""Validate the frozen benchmark spec and manifest without importing the model."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path


def small_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_benchmark(root, spec_path='config/benchmark_all8.json', require_disjoint_train=False):
    root = Path(root).resolve()
    spec_path = root / spec_path
    spec = json.loads(spec_path.read_text())
    snapshot = (root / spec['snapshot']).resolve()
    # A frozen run source links data back to the project-owned data directory.
    manifest = snapshot / 'manifest.csv'
    receipt = json.loads((snapshot / 'dataset_receipt.json').read_text())
    committed = root / spec['manifest']
    expected = {spec['train']['name']: spec['train']['count'],
                **{entry['name']: entry['count'] for entry in spec['tests']}}
    if (receipt['status'] != 'READY' or receipt['suite_id'] != spec['suite_id']
            or receipt['spec_sha256'] != small_hash(spec_path)
            or receipt['manifest_sha256'] != small_hash(manifest)
            or small_hash(committed) != receipt['manifest_sha256']
            or receipt['counts'] != expected):
        raise ValueError('Benchmark spec/receipt/committed manifest mismatch.')
    if require_disjoint_train and receipt.get('cross_train_test_identical_rgb_groups', 0):
        raise ValueError('Identical CASIA2 training RGB appears in All8 test data; resolve the documented overlap before launch.')
    preparation = receipt['preparation_environment']
    if (receipt['generator_sha256'] != small_hash(root / 'tools/generate_git10k_aux.py')
            or receipt['preparation_script_sha256'] != small_hash(root / 'tools/prepare_benchmark_data.py')
            or preparation != {'python_prefix': '/data0/hl/conda_envs/dcdsdiff', 'user_site_enabled': False,
                               'opencv': '4.9.0', 'numpy': '1.26.4', 'pillow': '10.2.0'}):
        raise ValueError('Benchmark generation source/environment differs from its registered preparation.')
    with manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if dict(Counter(row['dataset'] for row in rows)) != expected or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Benchmark manifest contains unexpected or duplicate records.')
    train_name = spec['train']['name']
    for row in rows:
        if row['split'] != ('train' if row['dataset'] == train_name else 'test'):
            raise ValueError('Benchmark manifest changed train/test membership.')
    return spec, snapshot, receipt, rows


def verify_benchmark_files(snapshot, rows, split=None):
    from tools.manage_experiment import sha256
    rows = [row for row in rows if split is None or row['split'] == split]
    jobs = []
    for group in dict.fromkeys(row['split'] for row in rows):
        members = [row for row in rows if row['split'] == group]
        for kind, field in (('f', 'image_file'), ('m', 'mask_file'), ('d', None), ('t', None)):
            expected = {row[field] if field else row['name'] + '.png' for row in members}
            if {path.name for path in (snapshot / group / kind).iterdir()} != expected:
                raise ValueError(f'Actual benchmark {group}/{kind} files differ from the manifest.')
    for row in rows:
        base = snapshot / row['split']
        image, mask = base / 'f' / row['image_file'], base / 'm' / row['mask_file']
        if image.resolve() != Path(row['image_source']).resolve():
            raise ValueError(f'Image source link changed: {row["name"]}')
        if row['binary_unit_mask_scaled'] == 'False' and mask.resolve() != Path(row['mask_source']).resolve():
            raise ValueError(f'Mask source link changed: {row["name"]}')
        jobs.extend([(image, row['image_sha256']), (mask, row['mask_sha256']),
                     (base / 'd' / (row['name'] + '.png'), row['detail_sha256']),
                     (base / 't' / (row['name'] + '.png'), row['trace_sha256'])])
        if row['binary_unit_mask_scaled'] == 'True':
            jobs.append((Path(row['mask_source']), row['mask_source_sha256']))
    def verify(job):
        path, digest = job
        if sha256(path) != digest:
            raise ValueError(f'Actual benchmark file content changed: {path}')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(verify, jobs))
    return {'verified_samples': len(rows), 'verified_files': len(jobs), 'status': 'PASS'}


def validate_casia_config(config, root):
    from omegaconf import OmegaConf
    from utils.init_utils import _load_config
    spec, snapshot, receipt, rows = load_benchmark(root, config.benchmark_spec, require_disjoint_train=True)
    baseline = _load_config(Path(root) / 'config/reproduction.yaml')
    # The authorized arm changes data and reporting metadata only.
    permitted = {'project_name', 'protocol_id', 'benchmark_spec', 'data_protocol',
                 'requested_evaluation', 'train_dataset', 'test_dataset'}
    original = OmegaConf.to_container(baseline, resolve=True)
    current = OmegaConf.to_container(config, resolve=True)
    if {k: v for k, v in current.items() if k not in permitted} != {
            k: v for k, v in original.items() if k not in permitted}:
        raise ValueError('CASIA2 arm may change only registered data/reporting settings.')
    for section, split, size in ((config.train_dataset, 'train', 'trainsize'),
                                  (config.test_dataset.Mix, 'test', 'testsize')):
        reference = baseline.train_dataset if split == 'train' else baseline.test_dataset.Mix
        params = OmegaConf.to_container(section.params, resolve=True)
        ref_params = OmegaConf.to_container(reference.params, resolve=True)
        for key, kind in (('image_root', 'f'), ('gt_root', 'm'), ('de_root', 'd'), ('trace_root', 't')):
            if (Path(root) / params.pop(key)).resolve() != snapshot / split / kind:
                raise ValueError(f'Unexpected CASIA2/All8 {split} {key}')
            ref_params.pop(key)
        if section.name != reference.name or params != ref_params or section.params[size] != 352:
            raise ValueError('Dataset transforms differ from the baseline.')
    if config.protocol_id != 'CASIA2-ALL8-V1' or config.requested_evaluation != 'all8_best':
        raise ValueError('Unexpected CASIA2 protocol/reporting policy.')
    verify_benchmark_files(snapshot, rows)
    return spec, snapshot, receipt
