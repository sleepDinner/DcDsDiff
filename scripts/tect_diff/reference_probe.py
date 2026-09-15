"""Fixed-final, training-only reference check without any weight selection.

The caller records the returned result before enforcing its health verdict.
This probe never trains, freezes, replaces, or selects reference weights.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from model.tect_diff.diffusion import coefficients, stable_noise
from scripts.tect_diff.common import restore_rng, rng_state, tensor_hash
from scripts.tect_diff.data import ReferenceDataset
from scripts.tect_diff.reference_health import (
    POLICY_ID, FINAL_PROBE_ID, FINAL_PROBE_NOISE_PURPOSES, summarize_reference_epoch,
)


PROBE_ID = FINAL_PROBE_ID
NOISE_PURPOSES = FINAL_PROBE_NOISE_PURPOSES


def final_reference_probe(worker, reference):
    """Check the actual fixed final weights with the registered 16 images.

    Two independent ID-derived noise draws are shared across the four scales.
    Moments are channel/pixel means in FP32, averaged over replicas per image,
    then summed in FP64 across images and ranks. Thus each scale's sample count
    is 16 images, corresponding to 32 image/noise observations. The selected
    images remain in reference and main training; this creates no held-out set.
    """
    config = worker.config['reference']
    count, seed = config['health_probe_images'], config['health_probe_seed']
    micro, lambdas = config['micro_batch'], worker.config['image_diffusion']['lambdas']
    if (config.get('health_policy') != POLICY_ID or count != 16 or seed != 42
            or micro != 2 or worker.config['resolution'] != 512
            or list(lambdas) != [2, 3, 4, 5]
            or worker.config['image_diffusion']['replicas'] != 2):
        raise ValueError('Final reference probe settings differ from the registered policy')
    if (worker.world != 2 or dist.get_world_size() != worker.world
            or dist.get_rank() != worker.rank):
        raise ValueError('Final reference probe requires the registered two ranks')
    if reference.architecture_version != config['architecture_version']:
        raise ValueError('Final reference probe architecture does not match its config')
    records = worker.bundle['reference'][:count]
    selected_ids = [row['id'] for row in records]
    if len(records) != count or len(set(selected_ids)) != count:
        raise ValueError('Final reference probe requires 16 distinct manifest image IDs')
    dataset = ReferenceDataset(records[worker.rank::worker.world], size=512, training=False)
    before_hash = tensor_hash(reference)
    saved_rng = rng_state()
    saved_modes = [(module, module.training) for module in reference.modules()]
    sums = torch.zeros(4, 5, device=worker.device, dtype=torch.float64)
    observed_ids = []
    try:
        reference.eval()
        with torch.no_grad():
            for offset in range(0, len(dataset), micro):
                rows = [dataset[index] for index in range(offset, min(offset + micro, len(dataset)))]
                ids = [row['id'] for row in rows]
                image = torch.stack([row['image'] for row in rows]).to(worker.device)
                noises = [stable_noise(image, ids, seed, purpose) for purpose in NOISE_PURPOSES]
                for scale, lam in enumerate(lambdas):
                    times = torch.full((len(image),), float(lam), device=worker.device)
                    a, sigma = coefficients(times)
                    moments = torch.zeros(len(image), 4, device=worker.device, dtype=torch.float32)
                    for noise in noises:
                        noisy = a[:, None, None, None] * image + sigma[:, None, None, None] * noise
                        with worker.amp():
                            predicted, _ = reference(noisy, times)
                        prediction = predicted.float()
                        moments += torch.stack([
                            (prediction - noise).square().flatten(1).mean(1),
                            noise.square().flatten(1).mean(1),
                            prediction.square().flatten(1).mean(1),
                            (prediction * noise).flatten(1).mean(1),
                        ], dim=1) / len(noises)
                    sums[scale, :4] += moments.double().sum(0)
                    sums[scale, 4] += len(image)
                observed_ids.extend(ids)
        after_hash = tensor_hash(reference)
        rank_receipts = [None] * worker.world
        dist.all_gather_object(rank_receipts, {
            'rank': worker.rank, 'ids': observed_ids,
            'before_hash': before_hash, 'after_hash': after_hash,
        })
        all_ids = [sample_id for row in rank_receipts for sample_id in row['ids']]
        if sorted(all_ids) != sorted(selected_ids) or len(set(all_ids)) != count:
            raise RuntimeError('Final reference probe DDP image coverage is not exact')
        if any(row['before_hash'] != before_hash or row['after_hash'] != before_hash
               for row in rank_receipts):
            raise RuntimeError('Final reference probe weights changed or differ across ranks')
        dist.all_reduce(sums)
        metrics = summarize_reference_epoch(config['epochs'] - 1, sums.tolist(), final=True)
        if metrics['samples_by_scale'] != [count] * len(lambdas):
            raise RuntimeError('Final reference probe has incomplete per-scale image counts')
        metrics['reference_health']['statistics_source'] = 'fixed_final_weights_training_reference_probe'
    finally:
        for module, mode in saved_modes:
            module.training = mode
        restore_rng(saved_rng)
    return {
        'probe_id': PROBE_ID,
        'model_parameter_hash': before_hash,
        'architecture_version': reference.architecture_version,
        'reference_manifest_sha256': worker.bundle['manifest_hashes']['reference'],
        'selection': 'first_16_in_frozen_reference_manifest_order',
        'selected_reference_ids': selected_ids,
        'image_count': count,
        'image_noise_observations_per_scale': count * len(NOISE_PURPOSES),
        'seed': seed,
        'micro_batch': micro,
        'resolution': 512,
        'lambdas': list(lambdas),
        'replicas': len(NOISE_PURPOSES),
        'noise_purposes': list(NOISE_PURPOSES),
        'shared_noise_across_scales': True,
        'training_augmentation': False,
        'ddp_padding': False,
        'parameters_unchanged': True,
        'rng_and_modes_restored': True,
        'checkpoint_selection': 'fixed_final_only_no_reselection',
        'scope': 'training_reference_images_only_no_validation_split_or_test_data',
        'metrics': metrics,
    }
