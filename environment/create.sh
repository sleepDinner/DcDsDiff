#!/usr/bin/env bash
# Run on imdl-server. Existing environments are verified, never overwritten.
set -euo pipefail

PROJECT_ROOT=/data1/hl/DcDsDiff-and-GIT10K
PREFIX=/data0/hl/conda_envs/dcdsdiff
CONDA=/opt/anaconda3/bin/conda
REQUIREMENTS="$PROJECT_ROOT/environment/requirements.txt"
DEPENDENCY_LOCK="$PROJECT_ROOT/environment/requirements.lock.txt"
BOOT="$PROJECT_ROOT/runtime/bootstrap"
READY="$BOOT/environment.ready"
export CONDA_PKGS_DIRS="$PROJECT_ROOT/runtime/cache/conda"
export PIP_CACHE_DIR="$PROJECT_ROOT/runtime/cache/pip"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export TMPDIR="$BOOT/tmp"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

[[ -f "$REQUIREMENTS" && -x "$CONDA" ]] || {
  printf 'Missing repository requirements or Conda executable.\n' >&2
  exit 2
}
mkdir -p "$BOOT" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
exec 9>"$BOOT/environment.lock"
flock -n 9 || { printf 'An environment operation already owns the lock.\n' >&2; exit 3; }

if [[ -e "$PREFIX" ]]; then
  [[ -x "$PREFIX/bin/python" && -f "$READY" ]] || {
    printf 'Existing prefix has no verified READY receipt; leaving it unchanged.\n' >&2
    exit 4
  }
  "$PREFIX/bin/python" - "$REQUIREMENTS" "$READY" "$PREFIX" "$DEPENDENCY_LOCK" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

requirements, receipt_path, prefix, dependency_lock = map(Path, sys.argv[1:])
receipt = json.loads(receipt_path.read_text())
assert receipt['status'] == 'READY', 'Environment is not READY'
assert Path(sys.prefix).resolve() == prefix.resolve(), 'Wrong interpreter prefix'
assert receipt['prefix'] == str(prefix), 'Receipt belongs to another environment'
assert receipt['requirements_sha256'] == hashlib.sha256(requirements.read_bytes()).hexdigest(), 'Requirements changed; create a separately authorized environment migration'
lock_hash = hashlib.sha256(dependency_lock.read_bytes()).hexdigest() if dependency_lock.exists() else None
assert receipt.get('dependency_lock_sha256') == lock_hash, 'Dependency lock changed'
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
assert hashlib.sha256(freeze).hexdigest() == receipt['freeze_sha256'], 'Installed packages have drifted'
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
print(json.dumps({'status': 'VERIFIED_EXISTING', 'prefix': str(prefix), 'requirements_sha256': receipt['requirements_sha256']}))
PY
  exit 0
fi

exec > >(tee -a "$BOOT/environment.log") 2>&1
date -Is > "$BOOT/environment.started_at"
trap 'code=$?; printf "%s\n" "$code" > "$BOOT/environment.exit_code"; date -Is > "$BOOT/environment.finished_at"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
"$CONDA" create --yes --json --prefix "$PREFIX" --override-channels --channel conda-forge python=3.10.21 pip
PYTHON="$PREFIX/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check 'pip==26.1.1' 'setuptools==69.5.1' wheel

# Reusing an existing wheel is optional. This exact hash was obtained from the
# official https://download.pytorch.org/whl/cu121/torch/ index on 2026-09-14.
WHEELS="$PROJECT_ROOT/runtime/cache/wheels"
TORCH_WHEEL="$WHEELS/torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl"
if [[ -f "$TORCH_WHEEL" ]]; then
  printf '%s  %s\n' b2184b7729ef3b9b10065c074a37c1e603fd99f91e38376e25cb7ed6e1d54696 "$TORCH_WHEEL" | sha256sum --check --status
  TORCH_SOURCE="$TORCH_WHEEL"
else
  TORCH_SOURCE='torch==2.1.2+cu121'
fi
mkdir -p "$WHEELS"
"$PYTHON" -m pip install --disable-pip-version-check \
  "$TORCH_SOURCE" 'torchvision==0.16.2+cu121' 'numpy==1.26.4' \
  --index-url https://download.pytorch.org/whl/cu121
LOCK_ARGS=()
if [[ -f "$DEPENDENCY_LOCK" ]]; then
  LOCK_ARGS=(-c "$DEPENDENCY_LOCK")
fi
"$PYTHON" -m pip install --disable-pip-version-check -r "$REQUIREMENTS" "${LOCK_ARGS[@]}"
"$PYTHON" -m pip check > "$BOOT/pip-check.txt"
"$PYTHON" -m pip freeze > "$BOOT/requirements.freeze.txt"
"$CONDA" list --prefix "$PREFIX" --explicit > "$BOOT/conda-explicit.txt"
"$PYTHON" - "$REQUIREMENTS" "$BOOT" "$PREFIX" "$DEPENDENCY_LOCK" <<'PY'
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

import accelerate
import albumentations
import cv2
from mmcv.cnn import ConvModule
import numpy
import omegaconf
import torch
import torchvision

requirements, boot, prefix, dependency_lock = map(Path, sys.argv[1:])
assert Path(sys.prefix).resolve() == prefix.resolve()
assert torch.__version__ == '2.1.2+cu121'
assert torchvision.__version__ == '0.16.2+cu121'
assert numpy.__version__ == '1.26.4'
assert torch.cuda.is_available(), 'CUDA is unavailable'
for line in requirements.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    name, version = line.split('==', 1)
    assert metadata.version(name) == version, (name, metadata.version(name), version)
receipt = {
    'schema_version': 1,
    'status': 'READY',
    'prefix': str(prefix),
    'requirements_sha256': hashlib.sha256(requirements.read_bytes()).hexdigest(),
    'dependency_lock_sha256': hashlib.sha256(dependency_lock.read_bytes()).hexdigest() if dependency_lock.exists() else None,
    'freeze_sha256': hashlib.sha256((boot / 'requirements.freeze.txt').read_bytes()).hexdigest(),
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'torchvision_version': torchvision.__version__,
    'cuda_runtime': torch.version.cuda,
    'cuda_device_count': torch.cuda.device_count(),
    'pip_check': (boot / 'pip-check.txt').read_text().strip(),
    'ready_at': datetime.now(timezone.utc).isoformat(),
}
temp = boot / 'environment.ready.tmp'
temp.write_text(json.dumps(receipt, indent=2) + '\n')
temp.replace(boot / 'environment.ready')
print(json.dumps(receipt, indent=2))
PY
