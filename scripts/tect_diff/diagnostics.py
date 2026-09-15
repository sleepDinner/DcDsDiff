"""Bounded engineering gates, separate from all formal optimizer histories.

The preflight has an explicitly synthetic, disposable reference/evidence path;
it never creates an artifact that could satisfy Stage A or B. The main probe
uses the actual trained reference and frozen calibration, then restores every
initial parameter, buffer, module mode and random-number state before training.
"""
from __future__ import annotations

import copy
import io
import inspect
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from model.loss import structure_loss
from model.tect_diff.diffusion import coefficients, ddim_step, stable_noise
from model.tect_diff.evidence import measure, tensor_tree_hash
from model.tect_diff.network import TECTNetwork
from denoising_diffusion_pytorch.simple_diffusion import logsnr_schedule_cosine, logsnr_schedule_shifted
from model.tect_diff.amp_context import self_condition_no_grad
from scripts.tect_diff.common import atomic_json, rng_state, restore_rng, timestamp, seed_all
from scripts.tect_diff.data import ManifestDataset


def _hash(module):
    return tensor_tree_hash(module.state_dict())


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite diagnostic {name}")


def _training_batch(worker, count, offset):
    records = sorted(worker.bundle['train'], key=lambda row: row['id'])
    if len(records) < offset + count:
        raise RuntimeError('Insufficient unique training samples for the bounded probe')
    dataset = ManifestDataset(records[offset:offset+count], worker.config['resolution'], training=False)
    rows = [dataset[index] for index in range(len(dataset))]
    batch = {key: torch.stack([row[key] for row in rows]).to(worker.device)
             for key in ('image', 'trace', 'mask', 'valid')}
    batch['id'] = [row['id'] for row in rows]
    return batch


def _gradient_receipt(module, scale=1.0):
    groups = {'rgb_pvt': 'network.backbone.', 'hf_pvt': 'network.backbone_t.',
              'mask': 'network.mask.', 'image_task': 'network.image_task_adapter.',
              'image_to_mask': 'network.image_to_mask.',
              'mask_to_image': 'network.image_task_adapter.mask_to_image.',
              'controller': 'gamma_logit'}
    receipt = {name: {'nonzero_tensors': 0, 'gradient_tensors': 0, 'squared_norm': 0.0}
               for name in groups}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith('reference.'):
            raise RuntimeError('Frozen reference has received a gradient')
        grad = parameter.grad.detach().float() / scale
        _finite(grad, f'gradient {name}')
        squared = float(grad.square().sum())
        for group, prefix in groups.items():
            if name.startswith(prefix):
                receipt[group]['gradient_tensors'] += 1
                receipt[group]['nonzero_tensors'] += int(squared > 0)
                receipt[group]['squared_norm'] += squared
    for group, values in receipt.items():
        values['norm'] = math.sqrt(values.pop('squared_norm'))
        if not values['nonzero_tensors'] or not math.isfinite(values['norm']):
            raise RuntimeError(f'TECT mechanism {group} has no finite nonzero gradient')
    return receipt


def _numeric_control_probe(model, device):
    original = model.gamma_logit.detach().clone()
    logits = torch.linspace(-20., 20., 16, device=device).reshape(1, 1, 4, 4)
    evidence = {'q': torch.ones_like(logits), 'ell': torch.where(logits >= 0, 5., -5.)}
    with torch.no_grad():
        try:
            model.gamma_logit.fill_(-10000.)
            zero, _ = model.control(logits, evidence, 9, 4)
            if not torch.equal(zero, logits):
                raise RuntimeError('gamma=0 does not preserve the base logits')
            model.gamma_logit.fill_(10000.)
            controlled, gamma = model.control(logits, evidence, 9, 4)
            _finite(controlled, 'extreme controlled logits')
            if not 0 <= float(gamma) <= 2:
                raise RuntimeError('Controller gain escaped its registered bounds')
            prefix_one, _ = model.control(logits, evidence, 9, 1)
            if not torch.equal(prefix_one, logits):
                raise RuntimeError('Prefix-one trajectory control is active')
            noisy = torch.randn_like(logits)
            probability = controlled.sigmoid()
            final = ddim_step(noisy, probability, torch.tensor([5.], device=device),
                              torch.tensor([float('inf')], device=device), final=True)
            if not torch.equal(final, 2 * probability - 1):
                raise RuntimeError('Final zero-noise DDIM step does not return controlled x0')
            stepped = ddim_step(noisy, probability, torch.tensor([-5.], device=device),
                                torch.tensor([5.], device=device))
            _finite(stepped, 'controlled DDIM step')
            return {'gamma_zero_identity': True, 'prefix_one_disabled': True,
                    'extreme_evidence_finite': True, 'final_zero_noise_identity': True}
        finally:
            model.gamma_logit.copy_(original)


def _noise_probe(image, ids, seed):
    first = stable_noise(image, ids, seed, 'image-0')
    second = stable_noise(image, ids, seed, 'image-1')
    mask = stable_noise(image, ids, seed, 'mask')
    if torch.equal(first, second) or torch.equal(first, -second):
        raise RuntimeError('The image replicas are equal or antithetic')
    if torch.equal(first[:, :1], mask) or torch.equal(second[:, :1], mask):
        raise RuntimeError('Image and mask noises are shared')
    def correlation(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        a, b = a-a.mean(), b-b.mean()
        return float((a*b).sum() / (a.norm()*b.norm()).clamp_min(1e-12))
    # Distinct seeds are the structural independence check; finite-sample
    # correlation is reported without pretending it proves independence.
    return {'different_replica_seeds': True, 'different_mask_seed': True,
            'replica_correlation': correlation(first, second),
            'image_mask_correlation': correlation(first[:, :1], mask),
            'noise_seed_rule': 'SHA256(seed | stable sample id | independent purpose)'}


class _SyntheticEngineeringProbe(nn.Module):
    """Disposable complete network exercise; never a FULL inference model."""
    synthetic_engineering_only = True

    def __init__(self, network, reference, config):
        super().__init__()
        self.network, self.reference, self.config = network, reference.freeze(), config
        self.gamma_logit = nn.Parameter(torch.tensor(math.log(.05/.95)))
        self.schedule = logsnr_schedule_shifted(logsnr_schedule_cosine, config['resolution'], 64)
        self.last_diagnostics = {}

    def train(self, mode=True):
        super().train(mode)
        self.reference.eval()
        return self

    def control(self, logits, evidence, j, m):
        gamma = 2 * self.gamma_logit.float().sigmoid() * (float(j) / 9)
        gamma = gamma if m >= 2 else gamma * 0
        return logits.float() + gamma * evidence['q'] * evidence['ell'], gamma

    @torch.no_grad()
    def observations(self, image, ids, keep_all):
        observed = image.float() * 2 - 1
        noise = torch.stack([stable_noise(image, ids, 728, f'image-{r}') for r in range(2)])
        features, estimates, responses = [], [], []
        for index, lam in enumerate(self.config['image_diffusion']['lambdas']):
            times = torch.full((len(image),), float(lam), device=image.device)
            a, sigma = coefficients(times)
            ff, ee, dd = [], [], []
            for replica in range(2):
                noisy = a[:, None, None, None]*observed + sigma[:, None, None, None]*noise[replica]
                epsilon, feature = self.reference(noisy, times)
                dd.append(measure(epsilon, noise[replica]))
                if keep_all or index == 3:
                    ee.append(epsilon.detach())
                    ff.append(feature.detach())
            responses.append(torch.stack(dd))
            if ff:
                features.append(torch.stack(ff, 1))
                estimates.append(torch.stack(ee, 1))
        # Keep the full raw pixel responses to exercise their real memory shape.
        responses = torch.stack(responses, 1)
        # Fixed bounded synthetic evidence tests controller wiring only. It is
        # intentionally not saved, calibrated, or labeled as trajectory evidence.
        ell = torch.linspace(-5., 5., image.shape[-1], device=image.device)[None, None, None]
        ell = ell.expand(len(image), 1, *image.shape[-2:])
        evidence = {'q': torch.full_like(ell, .5), 'ell': ell}
        return features, estimates, noise, responses, evidence

    def forward(self, image, trace, gt, ids):
        features, estimates, noise, responses, evidence = self.observations(image, ids, False)
        logsnr = self.schedule(torch.full((len(image),), .1, device=image.device))
        a, sigma = coefficients(logsnr)
        noisy_mask = a[:, None, None, None]*(2*gt-1) + sigma[:, None, None, None]*torch.randn_like(gt)
        image_time = torch.full((len(image),), 5., device=image.device)
        was_training = self.network.training
        self.network.eval()
        with self_condition_no_grad():
            pilot = self.network(image, noisy_mask, logsnr, features[-1], estimates[-1], image_time, trace=trace)
            pilot_ctrl, _ = self.control(pilot['logits_base'], evidence, 9, 4)
            previous = (F.interpolate(pilot_ctrl.sigmoid(), (32, 32), mode='bilinear', align_corners=False) >= .5).float()
        self.network.train(was_training)
        output = self.network(image, noisy_mask, logsnr, features[-1], estimates[-1], image_time,
                              previous=previous, trace=trace)
        controlled, gamma = self.control(output['logits_base'], evidence, 9, 4)
        with torch.autocast(device_type=image.device.type, enabled=False):
            mask_loss = structure_loss(controlled.float(), gt.float())
            image_loss = F.mse_loss(output['epsilon_joint'].float(), noise.permute(1, 0, 2, 3, 4).float())
            loss = mask_loss + image_loss
        self.last_diagnostics = {'mask_loss': float(mask_loss.detach()), 'image_loss': float(image_loss.detach()),
            'joint_ref_mse': float((output['epsilon_joint'].float()-estimates[-1].float()).square().mean().detach()),
            'reference_features': list(features[-1].shape), 'F_joint': list(output['F_joint'].shape),
            'mask_logits': list(output['logits_base'].shape), 'image_epsilon': list(output['epsilon_joint'].shape),
            'raw_response_shape': list(responses.shape), 'gamma': float(gamma.detach())}
        return loss

    @torch.no_grad()
    def sample(self, image, trace, ids):
        features, estimates, _, responses, evidence = self.observations(image, ids, True)
        noisy = stable_noise(image, ids, 728, 'mask')
        steps = torch.linspace(1., 0., 11, device=image.device).mul(math.pi/2).sin()
        previous = None
        for j in range(10):
            prefix = min(4, 1+4*j//10)
            log_t = self.schedule(steps[j].expand(len(image)))
            log_s = self.schedule(steps[j+1].expand(len(image)))
            image_time = torch.full((len(image),), float(self.config['image_diffusion']['lambdas'][prefix-1]), device=image.device)
            output = self.network(image, noisy, log_t, features[prefix-1], estimates[prefix-1],
                                  image_time, previous=previous, trace=trace)
            controlled, _ = self.control(output['logits_base'], evidence, j, prefix)
            probability = controlled.sigmoid()
            noisy = ddim_step(noisy, probability, log_t, log_s, final=j == 9)
            previous = (F.interpolate(probability, (32, 32), mode='bilinear', align_corners=False) >= .5).float()
        _finite(probability, 'synthetic ten-step sampling')
        return probability


def _gather_receipts(worker, receipt, name):
    atomic_json(worker.run / f'{name}_rank{worker.rank}.json', receipt)
    gathered = [None] * worker.world
    dist.all_gather_object(gathered, receipt)
    return gathered


def _checkpoint_roundtrip(model, optimizer, scheduler, scaler):
    """One in-memory exact-state round trip, without extra CUDA allocations."""
    saved_rng = rng_state()
    model_hash = _hash(model)
    optimizer_hash = tensor_tree_hash(optimizer.state_dict())
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
               'rng': saved_rng, 'epoch': 0, 'next_epoch': 1,
               'epoch_complete': True, 'evaluation_complete': True,
               'optimizer_step': 1, 'synthetic_engineering_only': True}
    memory = io.BytesIO()
    torch.save(payload, memory)
    serialized_bytes = memory.tell()
    del payload
    memory.seek(0)
    restored = torch.load(memory, map_location='cpu', weights_only=False)
    memory.close()
    if tensor_tree_hash(restored['model']) != model_hash:
        raise RuntimeError('Checkpoint model tensor round trip failed')
    if tensor_tree_hash(restored['optimizer']) != optimizer_hash:
        raise RuntimeError('Checkpoint optimizer tensor round trip failed')
    # Exercise actual strict model loading; no additional model is constructed.
    model.load_state_dict(restored['model'], strict=True)
    # A CPU optimizer with exactly matching parameter shapes validates its
    # restore interface without allocating a second set of CUDA Adam states.
    cpu_groups = []
    for group in optimizer.param_groups:
        options = {key: value for key, value in group.items() if key != 'params'}
        options['params'] = [nn.Parameter(torch.empty_like(parameter, device='cpu')) for parameter in group['params']]
        cpu_groups.append(options)
    restored_optimizer = torch.optim.AdamW(cpu_groups)
    restored_optimizer.load_state_dict(restored['optimizer'])
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, 100, eta_min=1e-6)
    restored_scheduler.load_state_dict(restored['scheduler'])
    if tensor_tree_hash(restored_optimizer.state_dict()) != optimizer_hash:
        raise RuntimeError('Loaded optimizer state differs from the serialized checkpoint')
    if restored_scheduler.state_dict() != scheduler.state_dict():
        raise RuntimeError('Scheduler state did not restore exactly')
    before_scaler = scaler.state_dict()
    scaler.load_state_dict(restored['scaler'])
    if scaler.state_dict() != before_scaler:
        raise RuntimeError('AMP scaler state did not restore exactly')
    if restored['next_epoch'] != 1 or restored['optimizer_step'] != 1 or not restored['epoch_complete']:
        raise RuntimeError('Checkpoint epoch/step recovery metadata is inconsistent')
    if [group['lr'] for group in restored_optimizer.param_groups] != [group['lr'] for group in optimizer.param_groups]:
        raise RuntimeError('Restored optimizer group learning rates differ')
    # Restoring from the loaded RNG (including CUDA states) exercises recovery;
    # restore the same state again after this diagnostic to make its scope clear.
    restore_rng(restored['rng'])
    restored_rng = rng_state()
    rng_equal = (torch.equal(restored_rng['torch'], saved_rng['torch']) and
                 all(torch.equal(a, b) for a, b in zip(restored_rng['cuda'], saved_rng['cuda'])) and
                 restored_rng['python'] == saved_rng['python'] and
                 np.array_equal(restored_rng['numpy'][1], saved_rng['numpy'][1]) and
                 restored_rng['numpy'][2:] == saved_rng['numpy'][2:])
    if not rng_equal:
        raise RuntimeError('Checkpoint random-number states did not restore exactly')
    receipt = {'status': 'PASS', 'storage': 'BytesIO only; no diagnostic weights saved',
               'serialized_bytes': serialized_bytes, 'model_sha256': model_hash,
               'optimizer_sha256': optimizer_hash, 'scheduler_restored': True,
               'scaler_restored': True, 'rng_restored': True, 'next_epoch': 1,
               'optimizer_step': 1, 'cpu_optimizer_restore': True}
    del restored, restored_optimizer, restored_scheduler, cpu_groups
    restore_rng(saved_rng)
    return receipt


def preflight(worker):
    """Both ranks: full-resolution accumulated DDP update and ten-step sampling."""
    start = time.monotonic()
    worker.status('PREFLIGHT', synthetic_engineering_only=True, substage='constructing_models')
    seed_all(worker.config['seed'])
    network = TECTNetwork(worker.config['pretrained_path'], worker.config['model']['gradient_checkpointing']).to(worker.device)
    reference = worker.reference_model()
    probe = _SyntheticEngineeringProbe(network, reference, worker.config).to(worker.device)
    ddp = DDP(probe, device_ids=[worker.device.index], find_unused_parameters=True, broadcast_buffers=False)
    reference_hash = _hash(reference)
    initial_network_hash = _hash(network)
    optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=worker.dtype == torch.float16)
    micro = worker.config['training']['micro_batch']
    accumulation = worker.config['training']['accumulation_steps']
    batch = _training_batch(worker, micro, worker.rank * micro)
    torch.cuda.reset_peak_memory_stats(worker.device)
    ddp.train()
    from contextlib import nullcontext
    for index in range(accumulation):
        context = nullcontext() if index == accumulation-1 else ddp.no_sync()
        with context:
            with worker.amp():
                loss = ddp(batch['image'], batch['trace'], batch['mask'], batch['id'])
            _finite(loss, 'preflight loss')
            scaler.scale(loss / accumulation).backward()
    gradients = _gradient_receipt(probe, scaler.get_scale())
    norm, skipped = worker.optimizer_update(probe, optimizer, scaler)
    if skipped or initial_network_hash == _hash(network):
        raise RuntimeError('Disposable preflight did not execute an actual finite optimizer update')
    if reference_hash != _hash(reference) or any(p.grad is not None for p in reference.parameters()):
        raise RuntimeError('Preflight reference parameters or buffers changed')
    scheduler.step()
    checkpoint = _checkpoint_roundtrip(probe, optimizer, scheduler, scaler)
    torch.cuda.synchronize(worker.device)
    train_peak = torch.cuda.max_memory_allocated(worker.device)
    worker.status('PREFLIGHT', synthetic_engineering_only=True, substage='optimizer_step_complete',
                  optimizer_step=1, loss=float(loss), gradient_norm=norm, peak_memory_bytes=train_peak)
    probe.eval()
    with torch.no_grad(), worker.amp():
        probability = probe.sample(batch['image'][:1], batch['trace'][:1], batch['id'][:1])
        # The frozen reference's forward signature cannot accept either mask.
        times = torch.tensor([5.], device=worker.device)
        noisy = torch.randn_like(batch['image'][:1])
        first, _ = reference(noisy, times)
        network.image_task_adapter(features=torch.zeros(1, 2, 256, worker.config['resolution']//4,
                    worker.config['resolution']//4, device=worker.device),
                    epsilon_ref=torch.zeros(1, 2, 3, worker.config['resolution'], worker.config['resolution'], device=worker.device),
                    image_logsnr=times, previous=torch.ones(1, 1, 32, 32, device=worker.device))
        second, _ = reference(noisy, times)
        if not torch.equal(first, second):
            raise RuntimeError('Mask task interaction changed a frozen reference prediction')
        padded_window = network.image_to_mask(torch.randn(1, 256, 9, 11, device=worker.device),
                                               torch.randn(1, 256, 9, 11, device=worker.device))
        _finite(padded_window, 'partially padded local attention window')
        if tuple(padded_window.shape) != (1, 256, 9, 11):
            raise RuntimeError('Padded-window attention returned an incorrect query grid')
    expected = (1, 1, worker.config['resolution'], worker.config['resolution'])
    if tuple(probability.shape) != expected:
        raise RuntimeError('Ten-step sample has the wrong output grid')
    numeric = _numeric_control_probe(probe, worker.device)
    noise = _noise_probe(batch['image'][:1], batch['id'][:1], worker.config['evaluation']['seed'])
    torch.cuda.synchronize(worker.device)
    receipt = {'status': 'PASS', 'rank': worker.rank, 'completed_at': timestamp(),
        'synthetic_engineering_only': True,
        'scope': 'Disposable untrained reference and fixed synthetic controller evidence; no Stage A/B artifact or scientific score',
        'resolution': worker.config['resolution'], 'micro_batch': micro, 'accumulation_steps': accumulation,
        'diagnostic_effective_global_batch': micro*accumulation*worker.world,
        'diagnostic_optimizer_steps': 1, 'formal_optimizer_steps': 0, 'finite_loss': float(loss),
        'gradients': gradients, 'reference_parameters_and_buffers_unchanged': True,
        'reference_mask_isolation': True, 'pretrained_load_report': network.pretrained_load_report,
        'checkpoint_roundtrip': checkpoint,
        'partial_window_padding_shape': list(padded_window.shape),
        'network_shapes': probe.last_diagnostics, 'control_numerics': numeric, 'noise': noise,
        'training_peak_allocated_bytes': train_peak, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(worker.device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(worker.device), 'elapsed_seconds': time.monotonic()-start,
        'sample_shape': list(probability.shape), 'sampling_steps': 10, 'training_ids': batch['id'],
        'limitation': 'Trained calibration and its actual pixel evidence are checked once in main_probe after stages A/B'}
    ranks = _gather_receipts(worker, receipt, 'preflight')
    ids = [item for rank in ranks for item in rank['training_ids']]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Preflight DDP training diagnostic IDs are duplicated across ranks')
    worker.receipt('preflight', synthetic_engineering_only=True, ranks=ranks,
                   no_diagnostic_weights_saved=True, same_resolution_real_artifact_probe_required=True)
    worker.status('PREFLIGHT_COMPLETED', synthetic_engineering_only=True, optimizer_step=1)
    return receipt


def _diagnostic_visualization(path, image, gt, output):
    """Server-only fixed training example; display transforms do not feed metrics."""
    rgb = image[0].detach().float().cpu().permute(1, 2, 0).numpy()
    arrays = [('observed RGB', rgb), ('training GT', gt[0, 0].detach().cpu().numpy()),
              ('P_base', output['P_base'][0, 0].detach().cpu().numpy()),
              ('P_ctrl', output['P_ctrl'][0, 0].detach().cpu().numpy()),
              ('signed A: tanh display', (.5+.5*torch.tanh(output['A'][0, 0])).cpu().numpy()),
              ('reliability q', output['q'][0, 0].detach().cpu().numpy())]
    side = min(384, image.shape[-1])
    canvas = Image.new('RGB', (3*side, 2*(side+24)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(arrays):
        array = (np.clip(array, 0, 1)*255).round().astype(np.uint8)
        tile = Image.fromarray(array).convert('RGB').resize((side, side), Image.Resampling.BILINEAR)
        x, y = index % 3 * side, index // 3 * (side+24)
        draw.text((x+4, y+4), label, fill='black')
        canvas.paste(tile, (x, y+24))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _prefix_four_seed(device, base):
    # Force the actual forward's random-time branch into its full prefix for
    # this disposable diagnostic, without replacing torch.rand or the sampler.
    for candidate in range(base, base+64):
        torch.manual_seed(candidate)
        torch.cuda.manual_seed(candidate)
        t = torch.rand((), device=device).clamp(1e-5, 1-1e-5)
        progress = 1 - 2 / math.pi * torch.asin(t)
        j = min(9, int((progress*10).item()))
        if min(4, 1+4*j//10) == 4:
            torch.manual_seed(candidate)
            torch.cuda.manual_seed(candidate)
            return candidate
    raise RuntimeError('Could not deterministically select the full-prefix diagnostic branch')


def main_probe(worker, model):
    """Probe actual trained artifacts once; restore initialization before return.

    Call on the unwrapped model before constructing DDP/optimizer, and only when
    there is no main-training checkpoint to resume. Each rank owns one of the
    first two sorted training IDs, so no test labels enter this diagnostic.
    """
    if isinstance(model, DDP) or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('main_probe expects an unwrapped model with no existing gradients')
    start = time.monotonic()
    saved_rng = rng_state()
    saved_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    saved_modes = {name: module.training for name, module in model.named_modules()}
    saved_config, saved_diagnostics = model.config, model.last_diagnostics
    initial_hash = _hash(model)
    reference_hash = _hash(model.reference)
    calibration_hash = model.evidence.assert_frozen()
    receipt = None
    worker.status('MAIN_PROBE', substage='actual_artifact_inference')
    try:
        if any(parameter.requires_grad for parameter in model.reference.parameters()):
            raise RuntimeError('Actual reference is not frozen')
        batch = _training_batch(worker, 1, worker.rank)
        model.eval()
        torch.cuda.reset_peak_memory_stats(worker.device)
        if 'gt' in inspect.signature(model.sample).parameters or 'mask' in inspect.signature(model.sample).parameters:
            raise RuntimeError('Inference API accepts forbidden ground-truth conditioning')

        def predict_from_evaluation_batch(item):
            # This is the same image/trace/id-only projection used by evaluate.
            return model.sample(item['image'], item['trace'], item['id'], return_diagnostics=True)

        with torch.no_grad(), worker.amp():
            probability, diagnostic = predict_from_evaluation_batch(batch)
            shuffled = dict(batch)
            shuffled['mask'] = 1-batch['mask']
            second, second_diagnostic = predict_from_evaluation_batch(shuffled)
        gt_change_max_abs = float((probability-second).abs().max())
        if not torch.equal(probability, second):
            raise RuntimeError('Changing an external test-style GT changed inference')
        for name in ('A', 'q', 'ell'):
            if not torch.equal(diagnostic[name], second_diagnostic[name]):
                raise RuntimeError(f'Changing external GT changed frozen {name}')
        if tuple(probability.shape) != (1, 1, worker.config['resolution'], worker.config['resolution']):
            raise RuntimeError('Actual FULL inference grid is incorrect')
        _finite(probability, 'actual FULL sample')
        sample_peak = torch.cuda.max_memory_allocated(worker.device)
        reference_context = model.evidence.prepare(batch['image']*2-1)
        original_indices = reference_context['indices'].clone()
        # Test actual Mask->Image interaction against an identical frozen input.
        with torch.no_grad(), worker.amp():
            noise = torch.stack([stable_noise(batch['image'], batch['id'], 0, f'image-{r}') for r in range(2)])
            responses, features, estimates, _ = model.reference_path(batch['image'], 4, noise)
            logsnr = model.schedule(torch.full((1,), .1, device=worker.device))
            mask_state = stable_noise(batch['image'], batch['id'], 0, 'mask')
            out0 = model.task(batch['image'], batch['trace'], mask_state, logsnr, features[-1], estimates[-1], 4,
                              previous=torch.zeros(1, 1, 32, 32, device=worker.device))
            out1 = model.task(batch['image'], batch['trace'], mask_state, logsnr, features[-1], estimates[-1], 4,
                              previous=torch.ones(1, 1, 32, 32, device=worker.device))
            history_effect = float((out0['epsilon_joint'].float()-out1['epsilon_joint'].float()).abs().mean())
            # Same frozen noisy observation after both task calls, scale 4/replica 0.
            time_i = torch.full((1,), 5., device=worker.device)
            a, sigma = coefficients(time_i)
            epsilon_again, _ = model.reference(a[:, None, None, None]*(batch['image']*2-1)+
                                               sigma[:, None, None, None]*noise[0], time_i)
        if not torch.equal(epsilon_again, estimates[-1][:, 0]):
            raise RuntimeError('Mask feedback contaminated the reference epsilon')
        changed_context = model.evidence.prepare(batch['image']*2-1)
        if not torch.equal(original_indices, changed_context['indices']):
            raise RuntimeError('Mask feedback changed image-only reference indices')
        measured_after = model.evidence.prefix(changed_context, responses)
        measured_before = model.evidence.prefix(reference_context, responses)
        if not torch.equal(measured_after['A'], measured_before['A']):
            raise RuntimeError('Mask feedback changed the frozen trajectory anomaly')
        if not history_effect > 0:
            raise RuntimeError('Restricted predicted-mask feedback has no effect on the trainable Image task')
        _diagnostic_visualization(worker.run/'diagnostics'/f'training-{worker.rank}.png',
                                  batch['image'], batch['mask'], diagnostic)
        evidence_summary = {key: {'mean': float(diagnostic[key].float().mean()),
                                  'min': float(diagnostic[key].float().min()),
                                  'max': float(diagnostic[key].float().max())}
                            for key in ('A', 'q', 'ell')}
        for key in ('A', 'q', 'ell', 'gamma'):
            _finite(diagnostic[key], f'actual diagnostic {key}')
        valid = batch['valid']
        target = batch['mask'].bool()
        def pixel_f1(probabilities):
            prediction = probabilities >= worker.config['evaluation']['threshold']
            tp = int((prediction & target & valid).sum())
            fp = int((prediction & ~target & valid).sum())
            fn = int((~prediction & target & valid).sum())
            denominator = 2*tp+fp+fn
            return 2*tp/denominator if denominator else 1.0
        base_f1, controlled_f1 = pixel_f1(diagnostic['P_base']), pixel_f1(diagnostic['P_ctrl'])
        sample_profile = diagnostic.get('sample_profile', {})
        gamma_value = float(diagnostic['gamma'])
        q_zero = bool((diagnostic['q'] == 0).all())
        ell_zero = bool((diagnostic['ell'] == 0).all())
        # Actual data may produce weak evidence. Report these mechanisms honestly;
        # the non-empty trained artifact, finite path and gradient gates remain.
        del second, second_diagnostic, reference_context, changed_context, original_indices
        del responses, features, estimates, out0, out1, measured_after, measured_before
        model.zero_grad(set_to_none=True)
        model.config = copy.deepcopy(saved_config)
        model.config['model']['self_condition_probability'] = 1.0
        # Both ranks use the same two predeclared training examples for this
        # unoptimized backward memory check; each rank's inference example is
        # different. These are diagnostic passes, with no formal sample counts.
        backward_batch = _training_batch(worker, worker.config['training']['micro_batch'], 0)
        diagnostic_seed = _prefix_four_seed(worker.device, 30042+worker.rank*100)
        model.train()
        with worker.amp():
            loss = model(backward_batch['image'], backward_batch['trace'], backward_batch['mask'])
        _finite(loss, 'actual full-prefix main loss')
        if model.last_diagnostics['prefix'] != 4:
            raise RuntimeError('Main mechanism probe did not reach the full trajectory prefix')
        scaler = torch.cuda.amp.GradScaler(enabled=worker.dtype == torch.float16)
        scaler.scale(loss).backward()
        gradients = _gradient_receipt(model, scaler.get_scale())
        from scripts.tect_diff.gradient_safety import MAIN_UNUSED
        missing = {name for name, p in model.named_parameters() if p.requires_grad and p.grad is None}
        if missing != MAIN_UNUSED:
            raise RuntimeError('Actual main probe has missing trainable gradients: ' + str(sorted(missing)))
        if reference_hash != _hash(model.reference) or calibration_hash != model.evidence.assert_frozen():
            raise RuntimeError('Actual reference or calibration changed during backward')
        if float(model.last_diagnostics['image_loss']) <= 0 or float(model.last_diagnostics['joint_ref_mse']) <= 0:
            raise RuntimeError('The supervised Image task is empty or identical to the frozen output')
        numerics = _numeric_control_probe(model, worker.device)
        torch.cuda.synchronize(worker.device)
        receipt = {'status': 'PASS', 'rank': worker.rank, 'completed_at': timestamp(),
            'synthetic_engineering_only': False, 'training_ids': batch['id'],
            'resolution': worker.config['resolution'], 'calibration_artifact_hash': model.evidence.artifact_hash,
            'reference_state_hash': reference_hash, 'calibration_state_hash': calibration_hash,
            'sample_shape': list(probability.shape), 'sampling_steps': 10,
            'external_gt_inversion_prediction_max_abs': gt_change_max_abs,
            'external_gt_invariance': True, 'reference_mask_isolation': True, 'reference_index_invariance': True,
            'anomaly_mask_invariance': True, 'history_epsilon_joint_mean_abs_change': history_effect,
            'image_loss': float(model.last_diagnostics['image_loss']),
            'joint_ref_mse': float(model.last_diagnostics['joint_ref_mse']),
            'finite_loss': float(loss.detach()), 'gradients': gradients, 'evidence': evidence_summary,
            'full_gradient_coverage': True, 'registered_unused_parameters': sorted(missing),
            'q_all_zero_on_training_probe': q_zero, 'ell_all_zero_on_training_probe': ell_zero,
            'gamma': gamma_value, 'sample_profile': sample_profile,
            'fixed_training_probe_pixel_f1': {'P_base': base_f1, 'P_ctrl': controlled_f1,
                                             'control_minus_base': controlled_f1-base_f1,
                                             'coordinates': 'fixed resized training grid; diagnostic only'},
            'backward_micro_batch': len(backward_batch['image']),
            'backward_training_ids': backward_batch['id'],
            'control_numerics': numerics, 'sampling_peak_allocated_bytes': sample_peak,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(worker.device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(worker.device),
            'elapsed_seconds': time.monotonic()-start, 'diagnostic_seed': diagnostic_seed,
            'diagnostic_override': 'Full prefix and one predicted self-conditioning pass; restored before formal training',
            'formal_optimizer_steps': 0, 'visualization_server_only': str(worker.run/'diagnostics'/f'training-{worker.rank}.png')}
    finally:
        model.load_state_dict(saved_state, strict=True)
        model.zero_grad(set_to_none=True)
        model.config, model.last_diagnostics = saved_config, saved_diagnostics
        for name, module in model.named_modules():
            module.training = saved_modes[name]
        restore_rng(saved_rng)
        if _hash(model) != initial_hash:
            raise RuntimeError('Main probe failed to restore the exact model initialization')
        model.evidence.assert_frozen()
    receipt['initial_state_and_rng_restored'] = True
    ranks = _gather_receipts(worker, receipt, 'main_probe')
    all_ids = [sample_id for item in ranks for sample_id in item['training_ids']]
    if len(set(all_ids)) != 2:
        raise RuntimeError('Main probe must cover exactly two fixed, unique training diagnostic images')
    worker.receipt('main_probe', ranks=ranks, initial_state_and_rng_restored=True,
                   formal_optimizer_steps=0, no_diagnostic_weights_saved=True)
    worker.status('MAIN_PROBE_COMPLETED', formal_optimizer_step=0, initialized_state_restored=True)
    return receipt
