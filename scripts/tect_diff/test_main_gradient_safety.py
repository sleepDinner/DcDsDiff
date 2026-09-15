"""CPU regressions for the no-grad AMP cache and pre-update DDP guards."""
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from model.tect_diff.amp_context import self_condition_no_grad
from scripts.tect_diff.gradient_safety import require_gradient_sync, require_parameter_sync


def distributed_checks(rank, directory):
    dist.init_process_group('gloo', init_method='file://' + str(Path(directory) / 'rendezvous'), rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = nn.Linear(2, 2)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        receipt = require_gradient_sync(model, norm, allowed_unused=frozenset())
        assert receipt['passed']
        for case in ('missing', 'different_norm', 'nonfinite_norm'):
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            if case == 'missing':
                model.weight.grad = None
            value = norm.clone()
            if case == 'different_norm' and rank == 1:
                value += 1
            if case == 'nonfinite_norm' and rank == 1:
                value.fill_(float('inf'))
            before = [p.detach().clone() for p in model.parameters()]
            try:
                require_gradient_sync(model, value, Path(directory)/f'{case}-rank{rank}.json', allowed_unused=frozenset())
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unsafe gradients accepted: ' + case)
            assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))
        require_parameter_sync(model, Path(directory)/'parameters.json')
        if rank == 1:
            with torch.no_grad():
                model.weight[0, 0] += 1
        try:
            require_parameter_sync(model, Path(directory)/'parameters-corrupt.json')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Divergent model parameters accepted')
    finally:
        dist.destroy_process_group()


class MainGradientSafetyTests(unittest.TestCase):
    def test_pilot_preserves_values_and_all_training_gradients(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 1))
        x = torch.ones(2, 4)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with self_condition_no_grad():
                pilot = model(x)
                self.assertFalse(pilot.requires_grad)
            output = model(x)
        self.assertTrue(torch.equal(pilot, output))
        self.assertEqual(output.dtype, torch.bfloat16)
        output.sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_nested_cache_and_grad_mode_restore_after_exception(self):
        for enabled in (False, True):
            with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=enabled):
                with self.assertRaises(ValueError):
                    with self_condition_no_grad():
                        self.assertFalse(torch.is_grad_enabled())
                        self.assertFalse(torch.is_autocast_cache_enabled())
                        raise ValueError('injected')
                self.assertTrue(torch.is_grad_enabled())
                self.assertEqual(torch.is_autocast_cache_enabled(), enabled)

    def test_distributed_guards(self):
        root = Path(__file__).resolve().parents[2]
        (root/'runtime').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='tect-gradient-safety-', dir=root/'runtime') as directory:
            mp.spawn(distributed_checks, args=(directory,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
