"""CASIA2 stabilization worker; frozen fitting dependencies and bounded two-set tests."""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE), str(SOURCE/'denoisingdiffusionpytorch')]
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from scripts.tect_diff.worker import Worker
from scripts.tect_diff.common import (atomic_json, atomic_torch_save, append_json, read_json,
    sha256, json_hash, seed_all, rng_state, restore_rng, tensor_hash)
from scripts.tect_diff.data import ManifestDataset, evaluation_collate
from scripts.tect_diff.pilot_data import load_pilot_bundle
from scripts.tect_diff.metrics import aggregate_dataset
from scripts.tect_diff.evaluation_pipeline import EvaluationMetricPipeline
from scripts.tect_diff.pilot_metrics import (aggregate_test2, build_epoch_record,
    upsert_epoch_jsonl, pilot_health_gate)
from scripts.tect_diff.pilot_state import (ensure_evaluation_binding, load_completed_dataset,
    validate_confirmation)
from scripts.tect_diff.gradient_safety import require_parameter_sync
from model.tect_diff.network import TECTNetwork
from model.tect_diff.diffusion import TECTDiffusion
from model.tect_diff.evidence import tensor_tree_hash


class PilotWorker(Worker):
    def __init__(self, run):
        self.run = Path(run).resolve(strict=True)
        self.config = read_json(self.run/'resolved_config.json')
        self.config_hash = json_hash(self.config)
        if sha256(self.config['pretrained_path']) != self.config['pretrained_sha256']:
            raise RuntimeError('Registered ImageNet weights changed')
        self.bundle = load_pilot_bundle(self.run/'data_bundle.json')
        self.rank, self.world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
        if self.world != 2:
            raise RuntimeError('Pilot requires two registered DDP ranks')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        self.device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl', timeout=timedelta(minutes=15))
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Registered pilot requires BF16 support')
        self.dtype = torch.bfloat16
        self.start = time.monotonic()
        seed_all(self.config['seed']+self.rank)

    def create_model(self):
        reference_artifact, reference_receipt = self.read_artifact('reference')
        calibration, calibration_receipt = self.read_artifact('calibration')
        if calibration['metadata']['reference_sha256'] != reference_receipt['sha256']:
            raise RuntimeError('Calibration is not bound to the registered reference')
        seed_all(self.config['seed'])
        reference = self.reference_model(reference_artifact).freeze()
        settings = self.config['model']
        network = TECTNetwork(self.config['pretrained_path'], settings['gradient_checkpointing'],
                              normalization=settings['normalization'],
                              architecture_version=settings['architecture_version']).to(self.device)
        if any(isinstance(layer, torch.nn.modules.batchnorm._BatchNorm) for layer in network.modules()):
            raise RuntimeError('The registered GN8 task network still contains BatchNorm')
        model = TECTDiffusion(network, reference, calibration, self.config).to(self.device)
        return model, calibration, reference_receipt, calibration_receipt

    def optimizer(self, model):
        settings = self.config['training']
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=settings['learning_rate'], weight_decay=settings['weight_decay'],
            betas=tuple(settings['betas']), eps=settings['eps'])

    def engineering_probe(self, model):
        from scripts.tect_diff.diagnostics import main_probe
        main_probe(self, model)
        saved = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        saved_rng, initial_hash = rng_state(), tensor_hash(model)
        reference_hash = tensor_hash(model.reference)
        batch_check = self.batch_independence_probe(model)
        optimizer = self.optimizer(model)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        ddp = DDP(model, device_ids=[self.device.index], find_unused_parameters=True, broadcast_buffers=False)
        dataset = ManifestDataset(self.bundle['preflight_train'], self.config['resolution'], training=True)
        loader, _ = self.loader(dataset, 0, self.config['training']['micro_batch'])
        iterator = iter(loader)
        receipts, started = [], time.monotonic()
        try:
            seed_all(self.config['seed']+self.rank)
            ddp.train()
            for step in range(self.config['pilot']['engineering_updates']):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                with self.amp():
                    loss = ddp(batch['image'].to(self.device), batch['trace'].to(self.device), batch['mask'].to(self.device))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite engineering loss')
                scaler.scale(loss).backward()
                norm, skipped = self.optimizer_update(model, optimizer, scaler, check_main_sync=True)
                if skipped:
                    raise RuntimeError('Engineering update skipped')
                receipts.append({'step': step+1, 'loss': float(loss), 'gradient_norm': norm,
                                 'gradient_sync': self.last_gradient_sync})
            digest = tensor_tree_hash(optimizer.state_dict())
            hashes = [None]*self.world
            dist.all_gather_object(hashes, digest)
            if len(set(hashes)) != 1 or tensor_hash(model.reference) != reference_hash:
                raise RuntimeError('Engineering Adam rank agreement or frozen reference failed')
            require_parameter_sync(model, self.run/'engineering_parameter_agreement.json')
            if tensor_hash(model) == initial_hash:
                raise RuntimeError('Engineering updates did not change the task network')
            model.evidence.assert_frozen()
            receipt = {'updates': receipts, 'adam_hashes': hashes,
                       'fp32_batch_independence': batch_check,
                       'elapsed_seconds': time.monotonic()-started,
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device)}
        finally:
            del iterator, loader, ddp, optimizer
            model.load_state_dict(saved, strict=True)
            model.zero_grad(set_to_none=True)
            restore_rng(saved_rng)
            if tensor_hash(model) != initial_hash:
                raise RuntimeError('Engineering probe did not restore the fresh initialization')
        ranks = [None]*self.world
        dist.all_gather_object(ranks, receipt)
        self.receipt('engineering', **receipt, ranks=ranks, fresh_initialization_restored=True,
                     formal_optimizer_steps=0, no_diagnostic_weights_saved=True)
        dist.barrier()

    def batch_independence_probe(self, model):
        """Training images only: isolate mathematical batch coupling from BF16 rounding."""
        records = self.bundle['preflight_train'][self.rank*2:self.rank*2+2]
        dataset = ManifestDataset(records, self.config['resolution'], training=False)
        examples = [dataset[index] for index in range(2)]
        images, traces = [torch.stack([row[key] for row in examples]).to(self.device) for key in ('image', 'trace')]
        ids = [row['id'] for row in examples]
        modes = {module: module.training for module in model.modules()}
        tf32 = torch.backends.cudnn.allow_tf32
        before = rng_state()
        try:
            model.eval()
            torch.backends.cudnn.allow_tf32 = False
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                batched = model.sample(images, traces, ids)
                singles = torch.cat([model.sample(images[i:i+1], traces[i:i+1], ids[i:i+1]) for i in range(2)])
            difference = float((batched-singles).abs().max())
            if not math.isfinite(difference) or difference > 1e-5:
                raise RuntimeError(f'FP32 per-image independence gate failed: {difference}')
            return {'training_ids': ids, 'max_probability_difference': difference, 'budget': 1e-5,
                    'passed': True, 'formal_precision_unchanged': True}
        finally:
            torch.backends.cudnn.allow_tf32 = tf32
            for module, training in modes.items():
                module.training = training
            restore_rng(before)

    def evaluation_binding(self, checkpoint):
        return {'epoch': checkpoint['epoch'], 'config_hash': self.config_hash,
                'run_id': self.run.name, 'source_commit': read_json(self.run/'provenance.json')['commit'],
                'checkpoint_parameter_sha256': checkpoint['trainable_parameter_sha256']}

    def evaluate_groups(self, model, groups, directory, binding, reuse_directory=None):
        """Original-size metrics; rank shards never pad. Reuse exact existing IDs only."""
        epoch = binding['epoch']
        population_binding = {**binding, 'populations': {name: json_hash(rows) for name, rows in groups.items()}}
        if self.rank == 0:
            ensure_evaluation_binding(directory, population_binding)
        dist.barrier()
        if reuse_directory is not None:
            quick_groups = {**self.bundle['tests'], 'CASIA2_authentic_probe': self.bundle['authentic_probe']}
            expected = {**binding, 'populations': {name: json_hash(rows) for name, rows in quick_groups.items()}}
            if read_json(reuse_directory/'evaluation_binding.json') != expected:
                raise RuntimeError('Reused evaluation belongs to a different model or population')
        for buffer in model.buffers():
            dist.broadcast(buffer, 0)
        model.eval()
        saved_rng = rng_state()
        outputs, evaluation_seconds = {}, 0.0
        try:
            for name, records in groups.items():
                completed = load_completed_dataset(directory, name, [row['id'] for row in records])
                if completed is not None:
                    outputs[name] = completed['metrics']
                    evaluation_seconds += completed['evaluation_seconds']
                    continue
                started = time.monotonic()
                reused = []
                if reuse_directory is not None:
                    cached = load_completed_dataset(reuse_directory, name, [row['id'] for row in quick_groups[name]])
                    if cached is None:
                        raise RuntimeError('Confirmation requires a complete quick evaluation')
                    reused = cached['per_image']
                    allowed = {row['id'] for row in records}
                    if any(row['id'] not in allowed for row in reused):
                        raise RuntimeError('Reused evaluation is outside the confirmation population')
                used = {row['id'] for row in reused}
                remaining = [row for row in records if row['id'] not in used]
                loader = []
                if remaining:
                    dataset = ManifestDataset(remaining, self.config['resolution'], original_gt=True)
                    loader = DataLoader(Subset(dataset, list(range(self.rank, len(dataset), self.world))),
                        batch_size=self.config['evaluation']['micro_batch'], shuffle=False,
                        num_workers=self.config['data']['loader_workers'], pin_memory=True,
                        collate_fn=evaluation_collate)
                with EvaluationMetricPipeline(self.config['evaluation']['metric_workers'],
                        self.config['evaluation']['metric_queue_limit']) as pipeline:
                    for batch in loader:
                        with torch.no_grad(), self.amp():
                            prediction = model.sample(batch['image'].to(self.device, non_blocking=True),
                                batch['trace'].to(self.device, non_blocking=True), batch['id'])
                        for index, sample_id in enumerate(batch['id']):
                            probability = F.interpolate(prediction[index:index+1].float(),
                                size=batch['original_hw'][index], mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                            pipeline.submit(sample_id, probability, batch['gt'][index], batch['gt_valid'][index],
                                self.config['evaluation']['threshold'], self.config['evaluation']['boundary_diagonal_ratio'])
                        if pipeline.submitted % 32 == 0:
                            self.status('PILOT_EVALUATION', epoch=epoch, dataset=name,
                                        inferred_on_rank=pipeline.submitted, assigned=len(loader.dataset))
                    pipeline.finish()
                    atomic_json(directory/f'{name}.rank{self.rank}.json', pipeline.rows)
                dist.barrier()
                if self.rank == 0:
                    combined = reused + [row for rank in range(self.world)
                                          for row in read_json(directory/f'{name}.rank{rank}.json')]
                    combined.sort(key=lambda row: row['id'])
                    outputs[name] = aggregate_dataset(combined, [row['id'] for row in records])
                    elapsed = time.monotonic()-started
                    evaluation_seconds += elapsed
                    atomic_json(directory/f'{name}.json', {'metrics': outputs[name], 'per_image': combined,
                                'reused_same_epoch_images': len(reused), 'new_inference_images': len(remaining),
                                'evaluation_seconds': elapsed})
                dist.barrier()
            if self.rank == 0:
                atomic_json(directory/'groups.json', {'datasets': outputs, 'epoch': epoch,
                            'evaluation_seconds': evaluation_seconds, 'evaluation_complete': True})
            dist.barrier()
            return read_json(directory/'groups.json')
        finally:
            restore_rng(saved_rng)

    def expected_counts(self, full=False):
        groups = self.bundle['tests_full' if full else 'tests']
        return {name: len(rows) for name, rows in groups.items()}

    def foreground_baselines(self, full=False):
        groups = self.bundle['tests_full' if full else 'tests']
        result = {}
        for name, records in groups.items():
            values = []
            for row in records:
                positive = row['positive_pixels']
                valid = math.prod(row['mask_hw'])-row['ignored_pixels']
                if valid <= 0:
                    raise RuntimeError('Invalid baseline mask denominator')
                values.append(2*positive/(valid+positive))
            result[name] = math.fsum(values)/len(values)
        return result

    def epoch_row(self, epoch, result, training, selected=False, full=False):
        datasets = {name: result['datasets'][name] for name in self.expected_counts(full)}
        row = build_epoch_record(epoch, self.config['training']['epochs'], datasets, self.expected_counts(full),
            selection_authority='Fixed pilot development gates; no independent generalization claim',
            selection_scope='Casiav1+Columbia complete confirmation' if full else 'Casiav1+Columbia fixed pilot subsets',
            training=training, health={'runtime_healthy': True,
                                      'authentic_probe': result['datasets']['CASIA2_authentic_probe']}, selected=selected)
        row.update(run_id=self.run.name, source_commit=read_json(self.run/'provenance.json')['commit'],
                   config_hash=self.config_hash, evaluation_seconds=result['evaluation_seconds'])
        return row

    def write_epoch(self, checkpoint):
        if self.rank == 0:
            row = checkpoint['pilot_epoch_record']
            upsert_epoch_jsonl(self.run/'metrics_per_epoch.jsonl', row)
            atomic_json(self.run/'epoch_summary.json', {'epoch': checkpoint['epoch'],
                'optimizer_step': checkpoint['optimizer_step'], 'epoch_complete': True,
                'evaluation_complete': True, 'best': checkpoint['best'], 'metrics': row})
        dist.barrier()

    def checkpoint_metadata(self, checkpoint):
        keys = ('epoch', 'optimizer_step', 'best', 'training_summary', 'pilot_epoch_record',
                'trainable_parameter_sha256', 'training_seconds', 'evaluation_seconds')
        payload = [{key: checkpoint[key] for key in keys if key in checkpoint} if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def gate_and_confirmation(self, model, checkpoint):
        rows = [json.loads(line) for line in (self.run/'metrics_per_epoch.jsonl').read_text().splitlines()]
        source = read_json(self.run/'provenance.json')['commit']
        if ([row['epoch_index'] for row in rows] != list(range(checkpoint['epoch']+1))
                or any(row.get('run_id') != self.run.name or row.get('config_hash') != self.config_hash
                       or row.get('source_commit') != source for row in rows)
                or rows[-1] != checkpoint['pilot_epoch_record']):
            raise RuntimeError('Pilot gate history is not bound to the current checkpoint and run')
        gate = pilot_health_gate(rows, self.foreground_baselines(), thresholds=self.config['pilot']['gate'],
                                expected_counts=self.expected_counts(),
                                authentic_expected_count=len(self.bundle['authentic_probe']))
        if self.rank == 0:
            atomic_json(self.run/'pilot_gate.json', gate)
        if not gate['passed']:
            required = self.config['pilot']['consecutive_degenerate_epochs']
            latest = rows[-required:]
            def degenerate(row):
                summary = row['summary']
                fraction = (summary['tp']+summary['fp'])/summary['valid_pixels']
                return fraction >= .99 or fraction <= .00001
            if len(latest) == required and all(degenerate(row) for row in latest):
                gate['stop_reason'] = 'Persistent near-total foreground or background across three complete epochs'
                return 'NO_GO', gate
            return None, gate
        epoch = checkpoint['epoch']
        binding = self.evaluation_binding(checkpoint)
        thresholds = {**self.config['pilot']['gate'], 'consecutive_epochs': 1}
        previous = self.run/'confirmation_result.json'
        if previous.exists():
            result = read_json(previous)
            confirmed_gate = validate_confirmation(result, binding, self.foreground_baselines(True),
                self.expected_counts(True), thresholds, len(self.bundle['authentic_probe']))
            return ('READY_FOR_FULL' if confirmed_gate['passed'] else 'NO_GO'), confirmed_gate
        directory = self.run/'confirmation'/f'epoch-{epoch:03d}'
        # Reuse current-epoch subset rows and its authentic probe, never another checkpoint.
        groups = {**self.bundle['tests_full'], 'CASIA2_authentic_probe': self.bundle['authentic_probe']}
        result = self.evaluate_groups(model, groups, directory, binding,
                                      reuse_directory=self.run/'epochs'/f'epoch-{epoch:03d}')
        row = self.epoch_row(epoch, result, checkpoint['training_summary'], full=True)
        confirmation = pilot_health_gate([row], self.foreground_baselines(True), thresholds=thresholds,
                                         expected_counts=self.expected_counts(True),
                                         authentic_expected_count=len(self.bundle['authentic_probe']))
        if self.rank == 0:
            atomic_json(directory/'metrics.json', row)
            atomic_json(directory/'gate.json', confirmation)
            atomic_json(self.run/'confirmation_result.json', {**binding, 'gate': confirmation, 'metrics': row})
        # A failed complete confirmation ends this bounded pilot; no repeated test tuning.
        return ('READY_FOR_FULL' if confirmation['passed'] else 'NO_GO'), confirmation

    def persist_outcome(self, outcome, checkpoint, gate):
        if self.rank == 0:
            best = checkpoint['best']
            confirmation_path = self.run/'confirmation_result.json'
            confirmation_seconds = (read_json(confirmation_path)['metrics']['evaluation_seconds']
                                    if confirmation_path.exists() else 0.0)
            best_receipt = {'epoch': best['epoch'], 'path': str(self.run/'best.pth'),
                            'sha256': sha256(self.run/'best.pth'), 'score': best['score']}
            self.receipt('pilot', outcome=outcome,
                source_commit=read_json(self.run/'provenance.json')['commit'], gate=gate,
                checkpoint={'epoch': checkpoint['epoch'], 'path': str(self.run/'last.pth'),
                            'sha256': sha256(self.run/'last.pth')}, best=best_receipt,
                optimizer_step=checkpoint['optimizer_step'],
                costs={'training_seconds': checkpoint['training_seconds'],
                       'quick_evaluation_seconds': checkpoint['evaluation_seconds'],
                       'confirmation_seconds': confirmation_seconds,
                       'evaluation_seconds': checkpoint['evaluation_seconds']+confirmation_seconds},
                selection_protocol='test_selected', full_data_training_started=False)
        dist.barrier()

    def main_training(self):
        settings = self.config['training']
        if settings['accumulation_steps'] != 1 or settings['global_batch'] != 2*settings['micro_batch']:
            raise ValueError('Registered pilot uses accumulation1 and global batch twice per-rank batch')
        model, calibration, reference_receipt, calibration_receipt = self.create_model()
        last_path = self.run/'last.pth'
        if not last_path.exists():
            self.engineering_probe(model)
        optimizer = self.optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings['scheduler_epochs'],
                                                              eta_min=settings['minimum_learning_rate'])
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        pending, start_epoch, step, best = None, 0, 0, None
        training_seconds = evaluation_seconds = 0.0
        if last_path.exists():
            pending = torch.load(last_path, map_location='cpu')
            self.restore(pending, model, optimizer, scheduler, scaler)
            start_epoch = pending['next_epoch'] if pending['evaluation_complete'] else pending['epoch']
            step, best = pending['optimizer_step'], pending['best']
            training_seconds, evaluation_seconds = pending['training_seconds'], pending['evaluation_seconds']
            if pending['evaluation_complete']:
                self.write_epoch(pending)
                outcome, gate = self.gate_and_confirmation(model, pending)
                if outcome is not None or start_epoch >= settings['epochs']:
                    self.persist_outcome(outcome or 'NO_GO', pending, gate)
                    return
        reference_hash = tensor_hash(model.reference)
        ddp = DDP(model, device_ids=[self.device.index], find_unused_parameters=True, broadcast_buffers=False)
        if pending is None:
            seed_all(self.config['seed']+self.rank)
        dataset = ManifestDataset(self.bundle['train'], self.config['resolution'], training=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        groups = {**self.bundle['tests'], 'CASIA2_authentic_probe': self.bundle['authentic_probe']}
        for epoch in range(start_epoch, settings['epochs']):
            already_trained = pending is not None and pending['epoch'] == epoch and not pending['evaluation_complete']
            if not already_trained:
                ddp.train()
                loader, sampler = self.loader(dataset, epoch, settings['micro_batch'])
                started = time.monotonic()
                sums = torch.zeros(3, device=self.device, dtype=torch.float64)
                prefix_counts = [0]*4
                for index, batch in enumerate(loader):
                    with self.amp():
                        loss = ddp(batch['image'].to(self.device, non_blocking=True),
                                   batch['trace'].to(self.device, non_blocking=True),
                                   batch['mask'].to(self.device, non_blocking=True))
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError('Non-finite pilot training loss')
                    scaler.scale(loss).backward()
                    if any(parameter.grad is not None for parameter in model.reference.parameters()):
                        raise RuntimeError('Frozen reference received a gradient')
                    norm, skipped = self.optimizer_update(model, optimizer, scaler, check_main_sync=True)
                    if skipped:
                        raise RuntimeError('Pilot AMP optimizer update was skipped')
                    step += 1
                    count = len(batch['image'])
                    sums[0] += loss.detach().double()*count
                    sums[1] += model.last_diagnostics['mask_loss'].double()*count
                    sums[2] += count
                    prefix_counts[model.last_diagnostics['prefix']-1] += 1
                    if step == 1 or step % settings['diagnostic_step_interval'] == 0:
                        diagnostics = {key: float(value) if torch.is_tensor(value) else value
                                       for key, value in model.last_diagnostics.items()}
                        row = {'epoch': epoch, 'optimizer_step': step, 'loss': float(loss),
                               'gradient_norm': norm, 'amp_skipped': False,
                               'gradient_sync': self.last_gradient_sync, **diagnostics}
                        append_json(self.run/f'training_metrics.rank{self.rank}.jsonl', row)
                        if self.rank == 0:
                            append_json(self.run/'training_metrics.jsonl', row)
                        self.status('PILOT_TRAINING', micro_batch=index+1, micro_batches_per_epoch=len(loader),
                                    peak_memory_bytes=torch.cuda.max_memory_allocated(self.device), **row)
                dist.all_reduce(sums)
                elapsed = time.monotonic()-started
                training_seconds += elapsed
                scheduler.step()
                training = {'loss': float(sums[0]/sums[2]), 'mask_loss': float(sums[1]/sums[2]),
                            'images': int(sums[2]), 'seconds': elapsed, 'images_per_second': float(sums[2])/elapsed,
                            'prefix_updates_on_rank0': prefix_counts if self.rank == 0 else None,
                            'optimizer_step': step, 'learning_rate_next_epoch': optimizer.param_groups[0]['lr'],
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device)}
                pending = self.training_checkpoint(model, optimizer, scheduler, scaler, epoch, step, False,
                    best=best, calibration_artifact=calibration, reference_receipt=reference_receipt,
                    calibration_receipt=calibration_receipt, training_seconds=training_seconds,
                    evaluation_seconds=evaluation_seconds, training_summary=training, selection_protocol='test_selected')
                if self.rank == 0:
                    atomic_torch_save(last_path, pending)
                dist.barrier()
            # Share only small metadata; do not reread the full optimizer checkpoint each epoch.
            metadata = self.checkpoint_metadata(pending)
            training = metadata['training_summary']
            pending = None
            if tensor_hash(model.reference) != reference_hash:
                raise RuntimeError('Frozen reference changed')
            model.evidence.assert_frozen()
            result = self.evaluate_groups(model, groups, self.run/'epochs'/f'epoch-{epoch:03d}',
                                          self.evaluation_binding(metadata))
            evaluation_seconds += result['evaluation_seconds']
            score = aggregate_test2({name: result['datasets'][name] for name in self.expected_counts()},
                                   self.expected_counts())['test2_macro_pixel_f1']
            is_best = best is None or score > best['score']
            if is_best:
                best = {'epoch': epoch, 'score': score}
            row = self.epoch_row(epoch, result, training, selected=is_best)
            complete = self.training_checkpoint(model, optimizer, scheduler, scaler, epoch, step, True,
                best=best, calibration_artifact=calibration, reference_receipt=reference_receipt,
                calibration_receipt=calibration_receipt, training_seconds=training_seconds,
                evaluation_seconds=evaluation_seconds, training_summary=training, pilot_epoch_record=row,
                selection_protocol='test_selected')
            if self.rank == 0:
                if is_best:
                    atomic_torch_save(self.run/'best.pth', complete)
                if epoch == settings['epochs']-1:
                    atomic_torch_save(self.run/'pilot_final.pth', complete)
                atomic_torch_save(last_path, complete)
            dist.barrier()
            complete = self.checkpoint_metadata(complete)
            self.write_epoch(complete)
            self.status('PILOT_EPOCH_COMPLETE', epoch=epoch, optimizer_step=step, pixel_f1=row['pixel_f1'],
                        average_test2=row['average_test2'], image_counts=row['image_counts'])
            outcome, gate = self.gate_and_confirmation(model, complete)
            if outcome is not None or epoch == settings['epochs']-1:
                self.persist_outcome(outcome or 'NO_GO', complete, gate)
                return
            pending = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    worker = PilotWorker(args.run_dir)
    try:
        worker.main_training()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
