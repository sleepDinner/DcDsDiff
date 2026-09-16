"""Small CPU loss/gradient tests; no task-network construction or GPU calls."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from model.loss import structure_loss
from model.tect_diff.mask_loss import tect_mask_loss


class MaskLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(42)
        self.logits = torch.randn(3, 1, 33, 35, generator=generator, dtype=torch.float64)
        self.targets = torch.zeros_like(self.logits)
        self.targets[1, :, 3:17, 4:21] = 1
        self.targets[2, :, 19:25, 13:27] = 1

    @staticmethod
    def loss_and_gradient(logits, function):
        prediction = logits.detach().clone().requires_grad_(True)
        value = function(prediction)
        return value.detach(), torch.autograd.grad(value, prediction)[0]

    def test_original_policy_preserves_exact_loss_and_gradient(self):
        expected = self.loss_and_gradient(self.logits, lambda pred: structure_loss(pred, self.targets))
        actual = self.loss_and_gradient(self.logits, lambda pred: tect_mask_loss(pred, self.targets, 'structure_v1'))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(expected, actual)))

    def test_all_nonempty_preserves_exact_original_loss_and_gradient(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                logits, targets = self.logits[1:].to(dtype), self.targets[1:].to(dtype)
                expected = self.loss_and_gradient(logits, lambda pred: structure_loss(pred, targets))
                actual = self.loss_and_gradient(logits, lambda pred: tect_mask_loss(pred, targets, 'empty_target_bce_v1'))
                self.assertTrue(all(torch.equal(left, right) for left, right in zip(expected, actual)))

    def test_empty_target_is_bce_with_its_normal_batch_mean_gradient(self):
        target = torch.zeros_like(self.logits)
        expected = self.loss_and_gradient(self.logits, lambda pred: F.binary_cross_entropy_with_logits(pred, target))
        actual = self.loss_and_gradient(self.logits, lambda pred: tect_mask_loss(pred, target, 'empty_target_bce_v1'))
        self.assertTrue(torch.allclose(actual[0], expected[0], rtol=1e-14, atol=0))
        self.assertTrue(torch.allclose(actual[1], expected[1], rtol=1e-14, atol=0))
        self.assertTrue(bool((actual[1] > 0).all()))

    def test_mixed_batch_keeps_all_images_in_denominator_and_tp_gradients_exact(self):
        def expected_loss(pred):
            return (F.binary_cross_entropy_with_logits(pred[:1], self.targets[:1])
                    + 2*structure_loss(pred[1:], self.targets[1:]))/3
        expected = self.loss_and_gradient(self.logits, expected_loss)
        actual = self.loss_and_gradient(self.logits, lambda pred: tect_mask_loss(pred, self.targets, 'empty_target_bce_v1'))
        original = self.loss_and_gradient(self.logits, lambda pred: structure_loss(pred, self.targets))
        self.assertTrue(torch.allclose(actual[0], expected[0], rtol=1e-14, atol=0))
        self.assertTrue(torch.allclose(actual[1], expected[1], rtol=1e-14, atol=0))
        self.assertTrue(torch.equal(actual[1][1:], original[1][1:]))
        self.assertFalse(torch.equal(actual[1][:1], original[1][:1]))

    def test_any_nonzero_soft_target_retains_iou_without_a_foreground_threshold(self):
        target = torch.zeros_like(self.logits[:1])
        target[0, 0, 16, 17] = 1e-12
        expected = self.loss_and_gradient(self.logits[:1], lambda pred: structure_loss(pred, target))
        actual = self.loss_and_gradient(self.logits[:1], lambda pred: tect_mask_loss(pred, target, 'empty_target_bce_v1'))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(expected, actual)))
        self.assertGreater(float(actual[0]), float(F.binary_cross_entropy_with_logits(self.logits[:1], target)))

    def test_unknown_missing_or_whitespace_policy_is_rejected(self):
        for policy in (None, '', ' ', 'structure_v1 ', 'EMPTY_TARGET_BCE_V1', 'unknown', [], {}, 1, True):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                tect_mask_loss(self.logits, self.targets, policy)

    def test_actual_diffusion_loss_expression_defaults_to_original_and_reads_policy(self):
        # Execute the real dispatch expression without importing/constructing a CUDA model.
        source = Path(__file__).resolve().parents[2]/'model/tect_diff/diffusion.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TECTDiffusion')
        forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        assignment = next(node for node in ast.walk(forward) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'mask_loss' for target in node.targets))
        expression = compile(ast.Expression(assignment.value), str(source), 'eval')
        for settings, policy in (({}, 'structure_v1'), ({'mask_loss':'structure_v1'}, 'structure_v1'),
                                 ({'mask_loss':'empty_target_bce_v1'}, 'empty_target_bce_v1')):
            with self.subTest(settings=settings):
                actual = eval(expression, {'self':SimpleNamespace(config={'training':settings}),
                    'controlled':self.logits, 'gt':self.targets, 'tect_mask_loss':tect_mask_loss})
                expected = tect_mask_loss(self.logits.float(), self.targets.float(), policy)
                self.assertTrue(torch.equal(actual, expected))
        with self.assertRaises(ValueError):
            eval(expression, {'self':SimpleNamespace(config={'training':{'mask_loss':None}}),
                'controlled':self.logits, 'gt':self.targets, 'tect_mask_loss':tect_mask_loss})


if __name__ == '__main__':
    unittest.main()
