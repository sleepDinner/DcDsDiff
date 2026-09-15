"""Two-process workers for the fixed A -> B -> C TECT protocol."""
import argparse
import csv
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'denoisingdiffusionpytorch'))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.nn.functional as F

from scripts.tect_diff.common import (atomic_json, atomic_torch_save, append_json, read_json,
    sha256, json_hash, seed_all, rng_state, restore_rng, tensor_hash, timestamp)
from scripts.tect_diff.data import load_bundle, ManifestDataset, ReferenceDataset, evaluation_collate
from scripts.tect_diff.metrics import per_image_metrics, aggregate_dataset, aggregate_all8
from scripts.tect_diff.reference_health import (
    POLICY_ID, summarize_reference_epoch, require_healthy_reference, require_final_reference_probe,
)
from model.tect_diff.reference import ReferenceDenoiser, REFERENCE_ARCHITECTURE_V1
from model.tect_diff.network import TECTNetwork
from model.tect_diff.diffusion import TECTDiffusion, coefficients, stable_noise
from model.tect_diff.evidence import CalibrationFitter, measure, save_calibration, FixedTrajectoryEvidence


class Worker:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.config = read_json(self.run / 'resolved_config.json')
        self.config_hash = json_hash(self.config)
        if sha256(self.config['pretrained_path']) != self.config['pretrained_sha256']:
            raise RuntimeError('Registered PVT initialization checksum changed')
        self.bundle = load_bundle(self.run / 'data_bundle.json')
        self.rank = int(os.environ['RANK'])
        self.world = int(os.environ['WORLD_SIZE'])
        if self.world != 2:
            raise RuntimeError('TECT formal workers require exactly two ranks')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        self.device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl', timeout=timedelta(hours=6))
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        seed_all(self.config['seed'] + self.rank)
        self.start = time.monotonic()

    def amp(self):
        return torch.autocast('cuda', dtype=self.dtype)

    def reference_model(self, artifact=None):
        reference = ReferenceDenoiser(
            self.config['model']['gradient_checkpointing'],
            architecture_version=self.config['reference'].get('architecture_version', REFERENCE_ARCHITECTURE_V1),
        ).to(self.device)
        if artifact is not None:
            reference.load_state_dict(artifact['model'], strict=True)
            if self.config['reference'].get('health_policy'):
                require_final_reference_probe(artifact.get('final_health_probe'), self.config['reference'],
                                              tensor_hash(reference), self.bundle['manifest_hashes']['reference'])
        return reference

    def status(self, stage, **values):
        value = {'updated_at': timestamp(), 'stage': stage, 'rank': self.rank,
                 'pid': os.getpid(), 'elapsed_seconds': time.monotonic()-self.start, **values}
        atomic_json(self.run / f'progress_rank{self.rank}.json', value)
        if self.rank == 0:
            atomic_json(self.run / 'progress.json', value)
            print(json.dumps(value, allow_nan=False), flush=True)

    def loader(self, dataset, epoch, micro, training=True):
        generator = torch.Generator().manual_seed(self.config['seed'] + epoch * 997 + self.rank)
        workers = self.config['data']['loader_workers']
        sampler = DistributedSampler(dataset, self.world, self.rank, shuffle=True,
                                     seed=self.config['seed'], drop_last=False)
        sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=micro, sampler=sampler, num_workers=workers,
                            pin_memory=True, persistent_workers=workers > 0,
                            prefetch_factor=self.config['data']['prefetch_factor'] if workers else None,
                            generator=generator, drop_last=False)
        return loader, sampler

    def receipt(self, name, **values):
        if self.rank == 0:
            atomic_json(self.run / f'{name}_receipt.json', {
                'status': 'COMPLETED', 'protocol_id': self.config['protocol_id'],
                'config_hash': self.config_hash, 'completed_at': timestamp(),
                'elapsed_seconds': time.monotonic()-self.start, **values})

    def read_artifact(self, name):
        receipt = read_json(self.run / f'{name}_receipt.json')
        if receipt['status'] != 'COMPLETED' or receipt['config_hash'] != self.config_hash:
            raise ValueError(f'{name} protocol/phase mismatch')
        path = Path(receipt['artifact_path'])
        if sha256(path) != receipt['sha256']:
            raise ValueError(f'{name} artifact file hash mismatch')
        artifact = torch.load(path, map_location='cpu')
        if name == 'reference':
            expected = self.config['reference'].get('architecture_version', REFERENCE_ARCHITECTURE_V1)
            if artifact.get('architecture_version', REFERENCE_ARCHITECTURE_V1) != expected:
                raise ValueError('Reference architecture does not match this protocol')
        return artifact, receipt

    def training_checkpoint(self, model, optimizer, scheduler, scaler, epoch, step,
                            evaluation_complete, **extra):
        if extra.get('selection_protocol') == 'test_selected':
            from scripts.tect_diff.gradient_safety import require_parameter_sync
            extra['trainable_parameter_sha256'] = require_parameter_sync(
                model, self.run / 'main_parameter_agreement.json')
        local_rng = rng_state()
        states = [None] * self.world
        dist.all_gather_object(states, local_rng)
        if self.rank != 0:
            return None
        return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'rng_by_rank': states, 'sampler': {'epoch': epoch, 'seed': self.config['seed'],
                                                'world_size': self.world, 'position': 'epoch_boundary'},
                'epoch': epoch, 'next_epoch': epoch+1, 'epoch_complete': True,
                'evaluation_complete': evaluation_complete, 'optimizer_step': step,
                'config': self.config, 'config_hash': self.config_hash,
                'source_commit': read_json(self.run / 'provenance.json')['commit'],
                'manifest_hashes': self.bundle['manifest_hashes'],
                'manifest_hash_definition': 'canonical JSON, sort_keys, compact separators, UTF-8, no trailing newline', **extra}

    def restore(self, checkpoint, model, optimizer, scheduler, scaler):
        if checkpoint['source_commit'] != read_json(self.run / 'provenance.json')['commit']:
            raise RuntimeError('Exact resume source commit mismatch')
        if checkpoint['config_hash'] != self.config_hash or checkpoint['manifest_hashes'] != self.bundle['manifest_hashes']:
            raise RuntimeError('Exact resume protocol/data mismatch')
        if checkpoint['sampler']['world_size'] != self.world:
            raise RuntimeError('Exact resume world size mismatch')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng_by_rank'][self.rank])

    def optimizer_update(self, model, optimizer, scaler, check_main_sync=False):
        scaler.unscale_(optimizer)
        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        # Keep flags on device until the single all-rank decision. Python all()
        # would synchronize once per parameter through Tensor.__bool__.
        checks = [torch.isfinite(p.grad).all() for p in parameters]
        finite = (torch.stack(checks).all().to(dtype=torch.int64) if checks
                  else torch.ones((), device=self.device, dtype=torch.int64))
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError('Non-finite gradient; no optimizer update made')
        maximum = self.config['training']['gradient_clip_norm']
        norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf') if maximum is None else maximum)
        if check_main_sync:
            from scripts.tect_diff.gradient_safety import require_gradient_sync
            self.last_gradient_sync = require_gradient_sync(
                model, norm, self.run / f'gradient_sync_failure_rank{self.rank}.json')
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < old_scale
        optimizer.zero_grad(set_to_none=True)
        return float(norm), bool(skipped)

    def reference(self):
        config = self.config['reference']
        health_policy = config.get('health_policy')
        if health_policy not in (None, POLICY_ID):
            raise ValueError('Unknown reference health policy')
        seed_all(self.config['seed'])
        reference = self.reference_model()
        initial_hash = tensor_hash(reference)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.cuda.amp.GradScaler(enabled=self.dtype == torch.float16)
        dataset = ReferenceDataset(self.bundle['reference'], self.config['resolution'])
        start_epoch, step, history = 0, 0, []
        last_path = self.run / 'reference_last.pth'
        if last_path.exists():
            last = torch.load(last_path, map_location='cpu')
            self.restore(last, reference, optimizer, scheduler, scaler)
            start_epoch, step, history = last['next_epoch'], last['optimizer_step'], last['history']
            initial_hash = last['initial_parameter_hash']
            if health_policy and history:
                require_healthy_reference(history[-1])
        model = DDP(reference, device_ids=[self.device.index], broadcast_buffers=False)
        seed_all(self.config['seed'] + self.rank) if start_epoch == 0 else None
        micro, accumulation = config['micro_batch'], config['accumulation_steps']
        for epoch in range(start_epoch, config['epochs']):
            reference.train()
            loader, sampler = self.loader(dataset, epoch, micro)
            sums = torch.zeros(4, 5, device=self.device, dtype=torch.float64)
            for index, batch in enumerate(loader):
                group_start = index // accumulation * accumulation
                group_count = min(accumulation * micro, len(sampler)-group_start*micro) * self.world
                synchronize = (index + 1) % accumulation == 0 or index == len(loader)-1
                context = nullcontext() if synchronize else model.no_sync()
                image = batch['image'].to(self.device, non_blocking=True)
                scale_ids = (torch.arange(len(image), device=self.device) + index*micro*self.world + self.rank*micro) % 4
                times = torch.tensor(self.config['image_diffusion']['lambdas'], device=self.device)[scale_ids]
                a, sigma = coefficients(times)
                noise = torch.randn_like(image)
                noisy = a[:, None, None, None]*image + sigma[:, None, None, None]*noise
                with context:
                    with self.amp():
                        predicted, _ = model(noisy, times)
                    loss_per_image = (predicted.float()-noise).square().flatten(1).mean(1)
                    loss = loss_per_image.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite reference MSE')
                    scaler.scale(loss * (len(image)*self.world/group_count)).backward()
                with torch.no_grad():
                    zero_mse = noise.square().flatten(1).mean(1)
                    energy = predicted.float().square().flatten(1).mean(1)
                    cross = (predicted.float()*noise).flatten(1).mean(1)
                    # The registered micro-batch has distinct scale IDs, so each
                    # row receives the same FP64 per-image addition as before.
                    # Avoid dynamic boolean indexing and its device-host syncs.
                    moments = torch.stack((loss_per_image.detach(), zero_mse, energy, cross,
                                           torch.ones_like(loss_per_image)), dim=1).double()
                    sums.index_add_(0, scale_ids, moments)
                if synchronize:
                    norm, skipped = self.optimizer_update(reference, optimizer, scaler)
                    step += int(not skipped)
                    if step == 1 or step % self.config['training']['diagnostic_step_interval'] == 0:
                        self.status('REFERENCE_TRAINING', epoch=epoch, optimizer_step=step, loss=float(loss),
                                    gradient_norm=norm, amp_skipped=skipped, micro_batch=index+1,
                                    micro_batches_per_epoch=len(loader), peak_memory_bytes=torch.cuda.max_memory_allocated())
            scheduler.step()
            dist.all_reduce(sums)
            if health_policy:
                metrics = summarize_reference_epoch(epoch, sums.tolist(), history, final=epoch == config['epochs']-1)
            else:
                metrics = {'epoch': epoch, 'mse_by_scale': (sums[:, 0]/sums[:, 4]).tolist(),
                           'zero_predictor_mse_by_scale': (sums[:, 1]/sums[:, 4]).tolist(),
                           'samples_by_scale': sums[:, 4].long().tolist()}
            metrics.update(optimizer_step=step, padded_training_samples=len(sampler)*self.world-len(dataset))
            history.append(metrics)
            last = self.training_checkpoint(reference, optimizer, scheduler, scaler, epoch, step, True,
                                            history=history, initial_parameter_hash=initial_hash)
            if self.rank == 0:
                atomic_torch_save(last_path, last)
                append_json(self.run / 'reference_metrics.jsonl', metrics)
                if health_policy:
                    atomic_json(self.run / 'reference_health.json', metrics)
            dist.barrier()
            self.status('REFERENCE_TRAINING', epoch=epoch, epoch_complete=True, optimizer_step=step, metrics=metrics)
            if health_policy:
                require_healthy_reference(metrics)
        final_hash = tensor_hash(reference)
        final_mse = np.mean(history[-1]['mse_by_scale'])
        zero_mse = np.mean(history[-1]['zero_predictor_mse_by_scale'])
        if initial_hash == final_hash or step < 1 or not np.isfinite(final_mse) or final_mse >= zero_mse:
            raise RuntimeError('Reference did not demonstrate finite learned noise prediction within fixed budget')
        if health_policy:
            require_healthy_reference(history[-1])
            if not history[-1]['reference_health']['final_endpoint_checked']:
                raise RuntimeError('Reference final per-scale health check is missing')
        final_probe = None
        if health_policy:
            from scripts.tect_diff.reference_probe import final_reference_probe
            final_probe = final_reference_probe(self, reference)
            if self.rank == 0:
                atomic_json(self.run / 'reference_final_health.json', final_probe)
            dist.barrier()
            require_healthy_reference(final_probe['metrics'])
            require_final_reference_probe(final_probe, config, final_hash, self.bundle['manifest_hashes']['reference'])
        artifact_path = Path(self.config['project_root']) / 'artifacts/reference' / self.run.name / 'reference_final.pth'
        if self.rank == 0:
            atomic_torch_save(artifact_path, {'model': reference.state_dict(), 'reference_trained': True,
                'config': self.config, 'config_hash': self.config_hash, 'epoch': config['epochs']-1,
                'architecture_version': reference.architecture_version,
                'final_health_probe': final_probe,
                'fit_manifest_sha256': self.bundle['manifest_hashes']['reference'],
                'initial_parameter_hash': initial_hash, 'final_parameter_hash': final_hash,
                'reference_source_mode': self.bundle['reference_source_mode'], 'history': history})
            self.receipt('reference', artifact_path=str(artifact_path), sha256=sha256(artifact_path),
                         optimizer_step=step, epoch=config['epochs']-1, initial_parameter_hash=initial_hash,
                         final_parameter_hash=final_hash, mean_epsilon_mse=float(final_mse),
                         zero_predictor_mse=float(zero_mse), sample_count=len(dataset),
                         reference_source_mode=self.bundle['reference_source_mode'],
                         architecture_version=reference.architecture_version,
                         health_metrics=history[-1] if health_policy else None,
                         final_health_probe=final_probe)
        dist.barrier()

    def calibration(self):
        artifact, receipt = self.read_artifact('reference')
        if artifact.get('reference_trained') is not True:
            raise ValueError('A trained reference is mandatory')
        reference = self.reference_model(artifact)
        reference.freeze()
        del artifact
        fitter = CalibrationFitter(self.config)
        dataset = ManifestDataset(self.bundle['calibration'][self.rank::self.world], self.config['resolution'], include_trace=False)
        for i in range(len(dataset)):
            batch = dataset[i]
            image = batch['image'][None].to(self.device) * 2 - 1
            gt = batch['mask'][None].to(self.device)
            noises = [stable_noise(image, [batch['id']], self.config['seed'], f'calibration-{r}') for r in range(2)]
            responses = []
            with torch.no_grad(), self.amp():
                for lam in self.config['image_diffusion']['lambdas']:
                    times = torch.tensor([lam], device=self.device)
                    a, sigma = coefficients(times)
                    scale = []
                    for r in range(2):
                        prediction, _ = reference(a[:, None, None, None]*image+sigma[:, None, None, None]*noises[r], times)
                        scale.append(measure(prediction.float(), noises[r]))
                    responses.append(torch.stack(scale))
            fitter.add(image, torch.stack(responses, 1), gt, batch['id'], batch['valid'][None])
            if i == 0 or (i+1) % 16 == 0:
                self.status('CALIBRATION_MEASUREMENT', images_complete=i+1, images_assigned=len(dataset))
        part = self.run / 'calibration_parts' / f'rank{self.rank}.pth'
        atomic_torch_save(part, {'records': fitter.records, 'config_hash': self.config_hash})
        dist.barrier()
        if self.rank == 0:
            for rank in range(1, self.world):
                incoming = torch.load(self.run/'calibration_parts'/f'rank{rank}.pth', map_location='cpu')
                if incoming['config_hash'] != self.config_hash:
                    raise RuntimeError('Calibration shard protocol mismatch')
                fitter.records.extend(incoming['records'])
            fitter.records.sort(key=lambda row: row['sample_id'])
            fitter.ids = {row['sample_id'] for row in fitter.records}
            if fitter.ids != {row['id'] for row in self.bundle['calibration']} or len(fitter.records) != len(fitter.ids):
                raise RuntimeError('Calibration DDP coverage failed')
            self.status('CALIBRATION_FITTING', images_complete=len(fitter.records))
            calibrated = fitter.finalize({'train_root': self.config['data']['train_root'],
                'training_manifest_sha256': self.bundle['manifest_hashes']['train'],
                'fit_manifest_sha256': self.bundle['manifest_hashes']['calibration'],
                'reference_sha256': receipt['sha256'], 'reference_trained': True,
                'input_resolution': self.config['resolution'], 'protocol_id': self.config['protocol_id'],
                'reference_source_mode': self.bundle['reference_source_mode']})
            artifact_path = Path(self.config['project_root'])/'artifacts/calibration'/self.run.name/'calibration.pth'
            save_calibration(calibrated, artifact_path)
            FixedTrajectoryEvidence(calibrated).assert_frozen()
            self.receipt('calibration', artifact_path=str(artifact_path), sha256=sha256(artifact_path),
                artifact_hash=calibrated['artifact_hash'], fit_receipt=calibrated['fit_receipt'],
                prefix_metadata=calibrated['prefix_metadata'])
        dist.barrier()

    def evaluate(self, model, epoch):
        # Broadcast buffers once, then bypass DDP forward for uneven no-padding shards.
        for buffer in model.buffers():
            dist.broadcast(buffer, 0)
        model.eval()
        saved_rng = rng_state()
        epoch_dir = self.run/'epochs'/f'epoch-{epoch:03d}'
        start = time.monotonic()
        outputs = {}
        try:
            for name, records in self.bundle['tests'].items():
                dataset = ManifestDataset(records, self.config['resolution'], original_gt=True)
                loader = DataLoader(Subset(dataset, list(range(self.rank, len(dataset), self.world))),
                    batch_size=self.config['evaluation']['micro_batch'], shuffle=False,
                    num_workers=self.config['data']['loader_workers'], pin_memory=True,
                    collate_fn=evaluation_collate)
                rows = []
                for batch in loader:
                    image = batch['image'].to(self.device, non_blocking=True)
                    trace = batch['trace'].to(self.device, non_blocking=True)
                    with torch.no_grad(), self.amp():
                        prediction = model.sample(image, trace, batch['id'])
                    for i, sample_id in enumerate(batch['id']):
                        probability = F.interpolate(prediction[i:i+1].float(), size=batch['original_hw'][i],
                            mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                        rows.append({'id': sample_id, **per_image_metrics(probability, batch['gt'][i], batch['gt_valid'][i],
                            self.config['evaluation']['threshold'], self.config['evaluation']['boundary_diagonal_ratio'])})
                    if len(rows) == 1 or len(rows) % 32 == 0:
                        self.status('MAIN_EVALUATION', epoch=epoch, dataset=name, evaluated_on_rank=len(rows), assigned=len(loader.dataset))
                atomic_json(epoch_dir/f'{name}.rank{self.rank}.json', rows)
                dist.barrier()
                if self.rank == 0:
                    combined = []
                    for rank in range(self.world):
                        combined.extend(read_json(epoch_dir/f'{name}.rank{rank}.json'))
                    combined.sort(key=lambda row: row['id'])
                    outputs[name] = aggregate_dataset(combined, [row['id'] for row in records])
                    atomic_json(epoch_dir/f'{name}.json', {'metrics': outputs[name], 'per_image': combined})
                    csv_path = epoch_dir/f'{name}.csv'
                    csv_temp = csv_path.with_suffix('.csv.tmp')
                    with csv_temp.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
                        writer.writeheader()
                        writer.writerows(combined)
                    os.replace(csv_temp, csv_path)
            if self.rank == 0:
                result = {'epoch': epoch, 'evaluation_complete': True, 'datasets': outputs,
                          **aggregate_all8(outputs), 'evaluation_seconds': time.monotonic()-start,
                          'inference_output': 'last_step_P_ctrl', 'noise_seed': 0, 'threshold': 0.5}
                atomic_json(epoch_dir/'evaluation.json', result)
            dist.barrier()
            return read_json(epoch_dir/'evaluation.json')
        finally:
            restore_rng(saved_rng)

    def main_training(self):
        reference_artifact, reference_receipt = self.read_artifact('reference')
        calibration, calibration_receipt = self.read_artifact('calibration')
        if calibration['metadata']['reference_sha256'] != reference_receipt['sha256']:
            raise ValueError('Calibration is not bound to this trained reference')
        seed_all(self.config['seed'])
        reference = self.reference_model(reference_artifact)
        reference.freeze()
        network = TECTNetwork(self.config['pretrained_path'], self.config['model']['gradient_checkpointing']).to(self.device)
        model = TECTDiffusion(network, reference, calibration, self.config).to(self.device)
        del reference_artifact
        frozen_hash = tensor_hash(reference)
        config = self.config['training']
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'],
            weight_decay=config['weight_decay'], betas=tuple(config['betas']), eps=config['eps'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['minimum_learning_rate'])
        scaler = torch.cuda.amp.GradScaler(enabled=self.dtype == torch.float16)
        last_path = self.run/'last.pth'
        start_epoch, step, best, pending = 0, 0, None, None
        if last_path.exists():
            pending = torch.load(last_path, map_location='cpu')
            self.restore(pending, model, optimizer, scheduler, scaler)
            start_epoch = pending['next_epoch'] if pending['evaluation_complete'] else pending['epoch']
            step, best = pending['optimizer_step'], pending['best']
        if pending is None:
            from scripts.tect_diff.diagnostics import main_probe
            main_probe(self, model)
        ddp = DDP(model, device_ids=[self.device.index], find_unused_parameters=True, broadcast_buffers=False)
        if pending is None:
            seed_all(self.config['seed']+self.rank)
        dataset = ManifestDataset(self.bundle['train'], self.config['resolution'], training=True)
        micro, accumulation = config['micro_batch'], config['accumulation_steps']
        training_seconds = pending.get('training_seconds', 0.0) if pending else 0.0
        evaluation_seconds = pending.get('evaluation_seconds', 0.0) if pending else 0.0
        for epoch in range(start_epoch, config['epochs']):
            already_trained = pending is not None and pending['epoch'] == epoch and not pending['evaluation_complete']
            epoch_start = time.monotonic()
            if not already_trained:
                ddp.train()
                loader, sampler = self.loader(dataset, epoch, micro)
                for index, batch in enumerate(loader):
                    group_start = index // accumulation * accumulation
                    group_count = min(accumulation*micro, len(sampler)-group_start*micro)*self.world
                    synchronize = (index+1) % accumulation == 0 or index == len(loader)-1
                    context = nullcontext() if synchronize else ddp.no_sync()
                    with context:
                        with self.amp():
                            loss = ddp(batch['image'].to(self.device, non_blocking=True),
                                batch['trace'].to(self.device, non_blocking=True), batch['mask'].to(self.device, non_blocking=True))
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Non-finite TECT loss')
                        scaler.scale(loss * (len(batch['image'])*self.world/group_count)).backward()
                    if synchronize:
                        # Frozen reference and calibration are never in optimizer.
                        if any(p.grad is not None for p in reference.parameters()):
                            raise RuntimeError('Reference received a gradient')
                        norm, skipped = self.optimizer_update(model, optimizer, scaler, check_main_sync=True)
                        step += int(not skipped)
                        if step == 1 or step % config['diagnostic_step_interval'] == 0:
                            diagnostics = {key: float(value) if torch.is_tensor(value) else value for key, value in model.last_diagnostics.items()}
                            diagnostics['gradient_sync'] = self.last_gradient_sync
                            self.status('MAIN_TRAINING', epoch=epoch, optimizer_step=step, loss=float(loss), gradient_norm=norm,
                                amp_skipped=skipped, micro_batch=index+1, micro_batches_per_epoch=len(loader),
                                peak_memory_bytes=torch.cuda.max_memory_allocated(), **diagnostics)
                            metric_row = {'epoch': epoch, 'optimizer_step': step, 'loss': float(loss),
                                'gradient_norm': norm, 'amp_skipped': skipped,
                                'peak_memory_bytes': torch.cuda.max_memory_allocated(), **diagnostics}
                            append_json(self.run/f'training_metrics.rank{self.rank}.jsonl', metric_row)
                            if self.rank == 0:
                                append_json(self.run/'training_metrics.jsonl', metric_row)
                scheduler.step()
                training_seconds += time.monotonic()-epoch_start
                pending = self.training_checkpoint(model, optimizer, scheduler, scaler, epoch, step, False,
                    best=best, calibration_artifact=calibration, reference_receipt=reference_receipt,
                    calibration_receipt=calibration_receipt, training_seconds=training_seconds,
                    evaluation_seconds=evaluation_seconds, selection_protocol='test_selected')
                if self.rank == 0:
                    atomic_torch_save(last_path, pending)
                dist.barrier()
            if tensor_hash(reference) != frozen_hash:
                raise RuntimeError('Reference parameters or buffers changed')
            model.evidence.assert_frozen()
            evaluation = self.evaluate(model, epoch)
            evaluation_seconds += evaluation['evaluation_seconds']
            score = evaluation['all8_macro_pixel_f1']
            is_best = best is None or score > best['metrics']['all8_macro_pixel_f1']
            if is_best:
                best = {'epoch': epoch, 'path': str(self.run/'best.pth'), 'metrics': evaluation}
            complete = self.training_checkpoint(model, optimizer, scheduler, scaler, epoch, step, True,
                best=best, calibration_artifact=calibration, reference_receipt=reference_receipt,
                calibration_receipt=calibration_receipt, metrics=evaluation,
                training_seconds=training_seconds, evaluation_seconds=evaluation_seconds, selection_protocol='test_selected')
            if self.rank == 0:
                # Commit best/final before last marks this epoch transaction complete.
                if is_best:
                    atomic_torch_save(self.run/'best.pth', complete)
                if epoch == 99:
                    atomic_torch_save(self.run/'final.pth', complete)
                atomic_torch_save(last_path, complete)
                atomic_json(self.run/'epoch_summary.json', {'epoch': epoch, 'epoch_complete': True,
                    'evaluation_complete': True, 'optimizer_step': step, 'best': best, 'metrics': evaluation,
                    'padded_training_samples': math.ceil(len(dataset)/self.world)*self.world-len(dataset)})
            dist.barrier()
            pending = None
            self.status('MAIN_EPOCH_COMPLETE', epoch=epoch, optimizer_step=step, metrics=evaluation)
        if self.rank == 0:
            evaluation = read_json(self.run/'epochs'/'epoch-099'/'evaluation.json')
            best['sha256'] = sha256(self.run/'best.pth')
            final = {'epoch': 99, 'path': str(self.run/'final.pth'), 'sha256': sha256(self.run/'final.pth'), 'metrics': evaluation}
            self.receipt('main', best=best, final=final, optimizer_step=step,
                costs={'training_seconds': training_seconds, 'evaluation_seconds': evaluation_seconds},
                selection_protocol='test_selected', reference_sha256=reference_receipt['sha256'],
                calibration_sha256=calibration_receipt['sha256'], pretrained_load_report=network.pretrained_load_report)
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--stage', choices=['preflight', 'reference', 'calibration', 'main'], required=True)
    args = parser.parse_args()
    worker = Worker(args.run_dir)
    try:
        if args.stage == 'preflight':
            from scripts.tect_diff.diagnostics import preflight
            preflight(worker)
        else:
            {'reference': worker.reference, 'calibration': worker.calibration, 'main': worker.main_training}[args.stage]()
    except BaseException as error:
        worker.status('FAILED', failed_stage=args.stage, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
