"""Per-GPU/per-run locks, compatible with the already-running legacy controller."""
import fcntl
import json
from pathlib import Path
import re


class ResourceBusy(RuntimeError):
    pass


def acquire_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise ResourceBusy(f'Resource is already reserved: {path}') from None
    return handle


def legacy_gpu_reservations(root):
    path = root / 'runtime/experiment.lock'
    if not path.exists():
        return set()
    try:
        handle = acquire_file(path)
    except ResourceBusy:
        pass
    else:
        handle.close()
        return set()
    inode = path.stat()
    gpus = set()
    for provenance_file in (root / 'runs').glob('*/provenance.json'):
        run = provenance_file.parent
        state_file = run / 'controller_status.json'
        if not state_file.exists():
            continue
        state = json.loads(state_file.read_text())
        provenance = json.loads(provenance_file.read_text())
        for pid in (state.get('controller_pid'), state.get('child_pid')):
            if not isinstance(pid, int):
                continue
            try:
                # Attribute the actual inherited descriptor, not a possibly reused PID.
                for fd in Path(f'/proc/{pid}/fd').iterdir():
                    try:
                        stat = fd.stat()
                        if (stat.st_dev, stat.st_ino) == (inode.st_dev, inode.st_ino):
                            gpus.add(int(provenance['gpu']))
                    except FileNotFoundError:
                        continue
            except FileNotFoundError:
                continue
    if not gpus:
        raise ResourceBusy('Legacy project lock is held but its GPU cannot be attributed safely.')
    return gpus


def acquire_resources(root, run_id, gpu):
    root = Path(root).resolve()
    if gpu < 0 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', run_id):
        raise ValueError('Invalid GPU or run ID')
    handles = []
    try:
        handles.append(acquire_file(root / 'runtime/locks' / f'gpu-{gpu}.lock'))
        handles.append(acquire_file(root / 'runtime/locks' / f'run-{run_id}.lock'))
        if gpu in legacy_gpu_reservations(root):
            raise ResourceBusy(f'GPU {gpu} is reserved by the legacy experiment controller.')
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def inherit_file(fd, path):
    import os
    if fd is None or fd < 3:
        raise ValueError('Internal worker requires an inherited resource lock.')
    expected, actual = Path(path).stat(), os.fstat(fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError('Inherited descriptor does not match the declared resource lock.')
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return os.fdopen(fd, 'a+')
