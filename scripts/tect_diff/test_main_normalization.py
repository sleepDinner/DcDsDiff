"""Bounded CPU checks for the explicitly versioned TECT GN8 architecture."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.tect_diff.normalization import (
    MAIN_ARCHITECTURE_GN8, MAIN_ARCHITECTURE_V1, TASK_BATCHNORMS,
    configure_main_normalization, require_main_normalization,
)


def task_bn_fixture():
    model = nn.Module()
    for path, channels in TASK_BATCHNORMS.items():
        parent = model
        parts = path.split('.')
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, nn.Module())
            parent = getattr(parent, part)
        parent.add_module(parts[-1], nn.BatchNorm2d(channels))
    return model


class MainNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_invalid_or_mismatched_versions_are_rejected(self):
        for value in (None, '', ' ', 'gn', 'GroupNorm8', [], 8):
            with self.subTest(normalization=value), self.assertRaises(ValueError):
                require_main_normalization(value, MAIN_ARCHITECTURE_GN8)
        for value in (None, '', ' ', 'unknown', [], 8):
            with self.subTest(architecture=value), self.assertRaises(ValueError):
                require_main_normalization('groupnorm8', value)
        for norm, version in (('batchnorm', MAIN_ARCHITECTURE_GN8),
                              ('groupnorm8', MAIN_ARCHITECTURE_V1)):
            with self.subTest(norm=norm), self.assertRaisesRegex(ValueError, 'do not match'):
                require_main_normalization(norm, version)

    def test_default_bn_preserves_module_identity_values_and_rng(self):
        model = task_bn_fixture()
        before = copy.deepcopy(model.state_dict())
        identities = {name: id(module) for name, module in model.named_modules()}
        rng = torch.get_rng_state().clone()
        configure_main_normalization(model, 'batchnorm', MAIN_ARCHITECTURE_V1)
        self.assertEqual(identities, {name: id(module) for name, module in model.named_modules()})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))

    def test_inventory_drift_fails_before_any_replacement(self):
        for fault in ('missing', 'unexpected', 'channels', 'epsilon', 'affine', 'syncbn'):
            with self.subTest(fault=fault):
                model = task_bn_fixture()
                parent = model.get_submodule('mask.pred2.0')
                if fault == 'missing':
                    del parent.bn
                elif fault == 'unexpected':
                    model.extra = nn.BatchNorm2d(8)
                elif fault == 'channels':
                    parent.bn = nn.BatchNorm2d(31)
                elif fault == 'epsilon':
                    parent.bn.eps = 1e-3
                elif fault == 'affine':
                    parent.bn = nn.BatchNorm2d(32, affine=False)
                else:
                    parent.bn = nn.SyncBatchNorm(32)
                before = {name: id(module) for name, module in model.named_modules()}
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    configure_main_normalization(model, 'groupnorm8', MAIN_ARCHITECTURE_GN8)
                self.assertEqual(before, {name: id(module) for name, module in model.named_modules()})

    def test_gn_has_no_running_statistics_and_is_batch_and_mode_independent(self):
        model = task_bn_fixture()
        rng = torch.get_rng_state().clone()
        configure_main_normalization(model, 'groupnorm8', MAIN_ARCHITECTURE_GN8)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(model.named_buffers()), [])
        self.assertFalse(any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                             for layer in model.modules()))
        for name, channels in TASK_BATCHNORMS.items():
            with self.subTest(layer=name):
                layer = model.get_submodule(name)
                self.assertIsInstance(layer, nn.GroupNorm)
                self.assertEqual((layer.num_groups, layer.num_channels, layer.eps, layer.affine),
                                 (8, channels, 1e-5, True))
                image = torch.randn(1, channels, 3, 3)
                others = torch.randn(5, channels, 3, 3) * 20 + 100
                layer.train()
                train = layer(image)
                layer.eval()
                torch.testing.assert_close(layer(image), train, atol=0, rtol=0)
                torch.testing.assert_close(layer(torch.cat((image, others)))[:1], train,
                                           atol=1e-6, rtol=1e-6)
                train.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in layer.parameters()))
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_old_checkpoint_cannot_silently_load_into_gn_architecture(self):
        legacy = task_bn_fixture()
        revised = task_bn_fixture()
        configure_main_normalization(revised, 'groupnorm8', MAIN_ARCHITECTURE_GN8)
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            revised.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            legacy.load_state_dict(revised.state_dict(), strict=True)


class TaskNetworkNormalizationIntegrationTests(unittest.TestCase):
    """Use real fusion/MIB modules; omit expensive PVT and weight-file loading."""

    @staticmethod
    def make_network(**kwargs):
        from model.tect_diff.network import TECTNetwork

        def backbone(**_):
            result = nn.Module()
            result.patch_embed1 = nn.Module()
            result.patch_embed1.mask_proj = nn.Conv2d(1, 8, 1)
            return result

        with patch('model.tect_diff.network.pvt_v2_b2', side_effect=backbone), \
                patch('model.tect_diff.network.BaselineNet._init_weights'):
            return TECTNetwork(None, **kwargs)

    def test_actual_fusion_mib_inventory_and_mmcv_norm_lookup(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        legacy = self.make_network()
        self.assertEqual(legacy.architecture_version, MAIN_ARCHITECTURE_V1)
        self.assertEqual({name for name, module in legacy.named_modules()
                          if isinstance(module, nn.BatchNorm2d)}, set(TASK_BATCHNORMS))
        torch.manual_seed(42)
        revised = self.make_network(normalization='groupnorm8',
                                    architecture_version=MAIN_ARCHITECTURE_GN8)
        self.assertEqual(revised.architecture_version, MAIN_ARCHITECTURE_GN8)
        self.assertFalse(any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                             for layer in revised.modules()))
        # GN initialization must preserve all unrelated initial parameters/RNG.
        for name, value in legacy.named_parameters():
            torch.testing.assert_close(dict(revised.named_parameters())[name], value, atol=0, rtol=0)
        for path in ('mask.down2.0', 'mask.down2.2', 'mask.up2.0',
                     'mask.up2.2', 'mask.up2.4', 'mask.pred2.0'):
            block = revised.get_submodule(path)
            self.assertEqual(block.norm_name, 'bn')
            self.assertIs(block.norm, block.bn)
            self.assertIsInstance(block.norm, nn.GroupNorm)
            image = torch.randn(2, block.conv.in_channels, 8, 8)
            output = block(image)
            self.assertTrue(torch.isfinite(output).all())


if __name__ == '__main__':
    unittest.main()
