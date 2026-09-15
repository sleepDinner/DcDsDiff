"""Small atomic IO, provenance and runtime helpers; no implicit GPU selection."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        f.flush()


def timestamp():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Expected a strict child of the project: {path}')
    return path


def runtime_environment(root, run_id):
    root = Path(root).resolve()
    # Short TMPDIR keeps Python worker Unix-domain socket paths below 108 bytes.
    cache_id = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    scratch = inside(root, root / 'cache/tect_diff' / cache_id)
    scratch.mkdir(parents=True, exist_ok=True)
    marker = scratch / '.tect-created.json'
    if marker.exists() and read_json(marker).get('run_id') != run_id:
        raise RuntimeError('Cache ownership mismatch')
    atomic_json(marker, {'run_id': run_id, 'created_by': 'tect-diff'})
    env = dict(os.environ)
    for name in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'TORCH_HOME', 'HF_HOME',
                 'PIP_CACHE_DIR', 'CONDA_PKGS_DIRS', 'TORCH_EXTENSIONS_DIR',
                 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'MPLCONFIGDIR', 'PYTHONPYCACHEPREFIX'):
        target = inside(root, scratch / ('t' if name in ('TMPDIR', 'TMP', 'TEMP') else name.lower()))
        target.mkdir(exist_ok=True)
        env[name] = str(target)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
    return env


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def tensor_hash(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
