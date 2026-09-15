"""Run model regression checks in the project's server environment."""
import unittest

import torch

from model.tect_diff.reference import (
    REFERENCE_ARCHITECTURE_V1, REFERENCE_ARCHITECTURE_V2,
    ReferenceDenoiser, TimeResidual,
)


class ReferenceModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_projection_skip_does_not_amplify_large_input_in_v2(self):
        legacy = TimeResidual(128, 64)
        stable = TimeResidual(128, 64, stabilized=True)
        stable.load_state_dict(legacy.state_dict(), strict=True)
        image = torch.randn(2, 128, 16, 16) * 100000
        times = torch.randn(2, 256)
        with torch.no_grad():
            old = legacy(image, times).square().mean().sqrt().item()
            fixed = stable(image, times).square().mean().sqrt().item()
        self.assertGreater(old, 1000)
        self.assertLess(fixed, 10)

    def test_v2_gradient_and_frozen_boundary(self):
        model = ReferenceDenoiser(architecture_version=REFERENCE_ARCHITECTURE_V2)
        image = torch.randn(2, 3, 64, 64)
        times = torch.tensor([2., 5.])
        prediction, features = model(image, times)
        prediction.square().mean().backward()
        for module in (model.entry, model.downsample_norm[0], model.upsample_norm[0], model.feature):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        self.assertEqual(features.shape, (2, 256, 16, 16))
        model.freeze().train()
        prediction, features = model(image.requires_grad_(), times)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertFalse(prediction.requires_grad or features.requires_grad)
        self.assertFalse(prediction.is_inference())

    def test_versions_reject_incompatible_state(self):
        legacy = ReferenceDenoiser()
        self.assertEqual(legacy.architecture_version, REFERENCE_ARCHITECTURE_V1)
        stable = ReferenceDenoiser(architecture_version=REFERENCE_ARCHITECTURE_V2)
        with self.assertRaises(RuntimeError):
            stable.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaises(ValueError):
            ReferenceDenoiser(architecture_version='unknown')

    def test_shared_image_adapter_block_keeps_legacy_behavior(self):
        implicit = TimeResidual(128, 64)
        explicit = TimeResidual(128, 64, stabilized=False)
        explicit.load_state_dict(implicit.state_dict(), strict=True)
        image, times = torch.randn(2, 128, 8, 8), torch.randn(2, 256)
        with torch.no_grad():
            self.assertTrue(torch.equal(implicit(image, times), explicit(image, times)))


if __name__ == '__main__':
    unittest.main()
