"""CPU regressions for geometry reuse; no model, data or trained artifact loads."""
from dataclasses import asdict, replace
import random
import unittest

import numpy as np
import torch

from model.tect_diff.evidence import (
    MEASUREMENT_VERSION, EvidenceConfig, FixedTrajectoryEvidence, _edge_valid,
    _grid_candidates, _knn, _positions, calibration_hash, content_descriptor,
    tensor_tree_hash,
)


def fixture(size=24, **options):
    config = EvidenceConfig(candidate_limit=64, query_chunk=37, **options)
    buffers = {'content_center': torch.linspace(-.1, .1, 6),
               'content_std': torch.linspace(.5, 1., 6),
               'mu': torch.arange(4 * 7 * 9).reshape(4, 7, 9).float() / 1000,
               'd0': torch.tensor(1.), 'content_beta': torch.linspace(-.2, .2, 7)}
    for prefix in range(1, 5):
        buffers.update({f'W_{prefix}': torch.linspace(.1, 1., 9 * (2 * prefix - 1)),
                        f'anomaly_scale_{prefix}': torch.tensor(1.),
                        f'v0_{prefix}': torch.tensor(.7),
                        f'joint_beta_{prefix}': torch.linspace(-.3, .3, 14)})
    artifact = {'format_version': MEASUREMENT_VERSION, 'config': asdict(config),
                'metadata': {'reference_trained': True, 'train_root': '/data0/hl/FinalTrainData',
                             'input_resolution': size},
                'fit_receipt': {'images': 2, 'normal_pixels': 64, 'tampered_pixels': 64},
                'prefix_metadata': {str(k): {} for k in range(1, 5)}, 'buffers': buffers}
    artifact['artifact_hash'] = calibration_hash(artifact)
    return FixedTrajectoryEvidence(artifact), artifact


def uncached_prepare(model, image):
    """Pre-change preparation, retaining the original uncached _knn branch."""
    batch, _, height, width = image.shape
    content = ((content_descriptor(image) - model.content_center[None, :, None, None]) /
               model.content_std[None, :, None, None])
    candidates = _grid_candidates(height, width, model.config, image.device)
    positions = _positions(height, width, image.device)
    edge = _edge_valid(height, width, model.config.invalid_border, image.device)
    matches = []
    for b in range(batch):
        flat = content[b].flatten(1).T
        matches.append(_knn(flat, flat[candidates], positions, positions[candidates],
                            height, width, model.config, edge))
    context = {key: torch.stack([entry[key] for entry in matches]) for key in matches[0]}
    context.update(content=content, candidates=candidates, height=height, width=width,
                   batch=batch, valid=edge.reshape(1, 1, height, width).expand(batch, -1, -1, -1),
                   artifact_hash=model.artifact_hash)
    return context


def chunked_relative_response(model, context, pair, scale):
    """Pre-change response path, including the original query-chunk loop."""
    content = context['content']
    design = torch.cat((torch.ones_like(content[:, :1]), content), 1)
    mean = torch.einsum('bchw,cd->bdhw', design, model.mu[scale])
    residual = pair.float() - mean[None]
    result = residual.clone()
    for replica in range(2):
        for batch in range(context['batch']):
            flat = residual[replica, batch].flatten(1).T
            candidate_values = flat[context['candidates']]
            target = result[replica, batch].flatten(1).T
            for start in range(0, context['height'] * context['width'], model.config.query_chunk):
                stop = min(start + model.config.query_chunk, context['height'] * context['width'])
                indices = context['indices'][batch, start:stop]
                weights = context['weights'][batch, start:stop]
                target[start:stop] -= (candidate_values[indices] * weights[..., None]).sum(1)
    return result


class GeometryCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_tensor_bits_equal(self, expected, actual, label='tensor'):
        self.assertEqual(expected.shape, actual.shape, label)
        self.assertEqual(expected.dtype, actual.dtype, label)
        self.assertTrue(torch.equal(expected.contiguous().reshape(-1).view(torch.uint8),
                                    actual.contiguous().reshape(-1).view(torch.uint8)), label)

    def assert_tensors_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                self.assert_tensor_bits_equal(value, actual[key], key)
            else:
                self.assertEqual(value, actual[key], key)

    def test_old_and_cached_matching_and_all_prefixes_are_exact(self):
        generator = torch.Generator().manual_seed(341)
        for size, batch in ((7, 1), (11, 2), (24, 2)):
            image = torch.rand(batch, 3, size, size, generator=generator) * 2 - 1
            responses = torch.randn(2, 4, batch, 9, size, size, generator=generator)
            model, _ = fixture(size)
            for observed in (image, image.flip(-1), torch.zeros_like(image)):
                with self.subTest(size=size, batch=batch):
                    original = uncached_prepare(model, observed)
                    cached = model.prepare(observed)
                    self.assert_tensors_equal(original, cached)
                    original_u = [chunked_relative_response(model, original, responses[:, scale], scale)
                                  for scale in range(4)]
                    for scale in range(4):
                        self.assert_tensor_bits_equal(original_u[scale],
                            model._relative_response(cached, responses[:, scale], scale))
                    for prefix in range(1, 5):
                        self.assert_tensors_equal(model._evaluate_u(original, original_u[:prefix]),
                                                  model.prefix(cached, responses[:, :prefix]))

    def test_reuse_geometry_invalidation_and_module_device_application(self):
        model, _ = fixture()
        image = torch.zeros(2, 3, 24, 24)
        first = model.prepare(image)
        cache = model._geometry_cache
        self.assertIsNotNone(cache)
        second = model.prepare(image + .3)
        self.assertIs(cache, model._geometry_cache)
        self.assertEqual(first['candidates'].data_ptr(), second['candidates'].data_ptr())
        model.config = replace(model.config, exclusion_divisor=4.)
        changed = model.prepare(image)
        self.assertIsNot(cache, model._geometry_cache)
        self.assert_tensors_equal(uncached_prepare(model, image), changed)
        cache = model._geometry_cache
        model._geometry(25, 24, torch.device('cpu'))
        self.assertIsNot(cache, model._geometry_cache)
        model.to('cpu')
        self.assertIsNone(model._geometry_cache)
        self.assert_tensors_equal(uncached_prepare(model, image), model.prepare(image))

    def test_cache_does_not_change_rng_calibration_or_state_dict(self):
        model, artifact = fixture()
        state_hash = tensor_tree_hash(model.state_dict())
        torch_rng, python_rng, numpy_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
        context = model.prepare(torch.zeros(2, 3, 24, 24))
        responses = torch.arange(2 * 4 * 2 * 9 * 24 * 24).float().reshape(2, 4, 2, 9, 24, 24) / 10000
        for prefix in range(1, 5):
            model.prefix(context, responses[:, :prefix])
        self.assertIsNotNone(model._geometry_cache)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], current_numpy[0])
        self.assertTrue(np.array_equal(numpy_rng[1], current_numpy[1]))
        self.assertEqual(numpy_rng[2:], current_numpy[2:])
        self.assertEqual(state_hash, tensor_tree_hash(model.state_dict()))
        self.assertEqual(state_hash, model.assert_frozen())
        self.assertEqual(artifact['artifact_hash'], calibration_hash(artifact))
        restored, _ = fixture()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(state_hash, restored.assert_frozen())
        self.assertIsNone(restored._geometry_cache)

    def test_unavailable_neighbors_match_original(self):
        model, _ = fixture(exclusion_divisor=.5)
        image = torch.zeros(2, 3, 24, 24)
        original, cached = uncached_prepare(model, image), model.prepare(image)
        self.assert_tensors_equal(original, cached)
        self.assertEqual(torch.count_nonzero(cached['weights']), 0)
        self.assertEqual(torch.count_nonzero(cached['availability']), 0)


if __name__ == '__main__':
    unittest.main()
