"""Fail before an update when the fixed FULL gradient contract is broken."""
import hashlib
import math

import torch
import torch.distributed as dist

from scripts.tect_diff.common import atomic_json, timestamp


# These eight inherited MMFF modules are constructed but never called by
# model/net.py:MMFF_att.forward. Keep their original None-gradient semantics.
MAIN_UNUSED = frozenset(
    f'network.mmff{level}.MMFF_att.{stream}_spatial_attention.conv1.weight'
    for level in range(1, 5) for stream in ('rgb', 'de'))


def require_gradient_sync(model, norm, failure_path=None, allowed_unused=MAIN_UNUSED):
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    missing = {name for name, p in parameters if p.grad is None}
    packet = torch.stack((norm.detach().double(), norm.new_tensor(len(parameters)-len(missing)).double(),
                          norm.new_tensor(int(missing == set(allowed_unused))).double()))
    gathered = [torch.empty_like(packet) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, packet)
    rows = torch.stack(gathered).cpu().tolist()
    norms = [row[0] for row in rows]
    passed = (all(math.isfinite(x) for x in norms)
              and all(row[2] == 1 for row in rows)
              and len({row[1] for row in rows}) == 1
              and max(norms)-min(norms) <= 1e-6 * max(1., max(norms)))
    receipt = {'passed': passed, 'at': timestamp(),
               'gradient_norm_by_rank': [x if math.isfinite(x) else str(x) for x in norms],
               'gradient_tensors_by_rank': [int(row[1]) for row in rows],
               'coverage_pass_by_rank': [bool(row[2]) for row in rows]}
    if not passed:
        missing_by_rank = [None] * dist.get_world_size()
        dist.all_gather_object(missing_by_rank, sorted(missing))
        receipt['missing_by_rank'] = missing_by_rank
        if failure_path is not None:
            atomic_json(failure_path, receipt)
        raise RuntimeError('Main gradient coverage or DDP synchronization failed before optimizer.step: ' + str(receipt))
    return receipt


def require_parameter_sync(model, receipt_path):
    # BN running statistics are intentionally rank-local; compare parameters,
    # including unused trainable parameters, without comparing those buffers.
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, digest.hexdigest())
    receipt = {'at': timestamp(), 'passed': len(set(hashes)) == 1, 'parameter_sha256_by_rank': hashes,
               'scope': 'trainable parameters; registered rank-local BN buffers excluded'}
    if dist.get_rank() == 0:
        atomic_json(receipt_path, receipt)
    if not receipt['passed']:
        raise RuntimeError('DDP parameters differ; checkpoint/evaluation blocked')
    return hashes[0]
