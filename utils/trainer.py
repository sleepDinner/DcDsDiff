"""Single-process training with epoch-boundary recovery and fixed-final reporting."""
import functools
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from model.train_val_forward import modification_train_val_forward
from utils.import_utils import fill_args_from_dict
from utils.logger_utils import create_logger
from utils.train_utils import set_random_seed


def normalize_gt_mask(gt):
    # GT is loaded from an 8-bit PIL L image. Preserve soft labels as ToTensor does.
    return np.asarray(gt, dtype=np.float32) / 255.0


def tracker_config_from_cfg(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def run_on_seed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        state = capture_rng_state()
        try:
            set_random_seed(0)
            return func(*args, **kwargs)
        finally:
            restore_rng_state(state)
    return wrapper


def atomic_write(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def finite(value, label):
    if not math.isfinite(float(value)):
        raise FloatingPointError(f'Non-finite {label}: {value}')
    return float(value)


class Trainer:
    def __init__(self, model, train_loader, test_loader=None,
                 train_val_forward_fn=modification_train_val_forward,
                 gradient_accumulate_every=1, optimizer=None, scheduler=None,
                 train_num_epoch=100, results_folder='./results', amp=False,
                 fp16=False, split_batches=True, log_with=None, cfg=None):
        if gradient_accumulate_every != 1:
            raise ValueError('The registered single-GPU protocol requires gradient_accumulate_every=1.')
        self.accelerator = Accelerator(
            mixed_precision='fp16' if (fp16 or amp) else 'no',
            gradient_accumulation_steps=1,
            log_with='wandb' if log_with and log_with != 'none' else None,
        )
        if self.accelerator.num_processes != 1:
            raise RuntimeError('This trainer requires exactly one process; DDP is not supported.')
        self.cfg = tracker_config_from_cfg(cfg) or {}
        self.results_folder = Path(results_folder or './results')
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.logger = create_logger(log_file=str(self.results_folder / 'train.log'))
        if log_with and log_with != 'none':
            self.accelerator.init_trackers(self.cfg.get('project_name', 'DcDsDiff'), config=self.cfg)
        self.model = self.accelerator.prepare(model)
        unwrapped = self.accelerator.unwrap_model(self.model)
        network = getattr(unwrapped, 'model', unwrapped)
        self.model_provenance = {
            'architecture_version': getattr(network, 'architecture_version', None),
            'pretrained_load_report': getattr(network, 'pretrained_load_report', None),
        }
        self.opt = self.accelerator.prepare(optimizer) if optimizer is not None else None
        # Keep the epoch scheduler outside Accelerator: exactly one step per completed epoch.
        self.scheduler = scheduler
        self.train_loader = self.accelerator.prepare(train_loader) if train_loader is not None else None
        self.test_loader = self.accelerator.prepare(test_loader) if test_loader is not None else None
        self.train_val_forward_fn = train_val_forward_fn
        self.train_num_epoch = int(train_num_epoch)
        self.cur_epoch = -1
        self.next_epoch = 0
        self.global_step = 0
        self.best_mae = float('inf')
        self.best_epoch = None
        self.epoch_records = []
        self.save_every = int(self.cfg.get('save_every', 10))
        self.status_interval = max(1.0, float(self.cfg.get('status_interval_seconds', 60)))
        self._last_status_time = 0.0
        self.contract = {key: self.cfg.get(key) for key in (
            'model', 'cond_uvit', 'diffusion_model', 'optimizer', 'train_dataset', 'test_dataset',
            'num_epoch', 'batch_size', 'num_workers', 'gradient_accumulate_every', 'seed', 'lr_min',
            'fp16', 'protocol_id', 'train_val_forward_fn',
        )}
        self.contract['train_num_epoch'] = self.train_num_epoch
        self.contract['mixed_precision'] = self.accelerator.mixed_precision
        if self.train_loader is not None:
            atomic_write(self.results_folder / 'resolved_config.yaml', OmegaConf.to_yaml(OmegaConf.create(self.cfg)))

    def write_status(self, state, force=True, **fields):
        now = time.monotonic()
        if not force and now - self._last_status_time < self.status_interval:
            return
        self._last_status_time = now
        record = {
            'state': state, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'pid': os.getpid(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'epoch': self.cur_epoch, 'next_epoch': self.next_epoch,
            'global_step': self.global_step, 'total_epochs': self.train_num_epoch,
            'best_mae': self.best_mae if math.isfinite(self.best_mae) else None,
            'best_epoch': self.best_epoch,
            'primary_selection': 'FIXED_FINAL_EPOCH',
            'best_selection': 'TEST_SELECTED_MAE_DIAGNOSTIC', **self.model_provenance, **fields,
        }
        atomic_write(self.results_folder / 'training_status.json', json.dumps(record, indent=2, allow_nan=False) + '\n')

    def _checkpoint_payload(self, selection):
        return {
            'checkpoint_format': 2, 'epoch': self.cur_epoch, 'next_epoch': self.next_epoch,
            'global_step': self.global_step, 'best_mae': self.best_mae,
            'best_epoch': self.best_epoch, 'selection': selection, 'contract': self.contract,
            **self.model_provenance,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict() if self.opt is not None else None,
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
            'rng': capture_rng_state(), 'epoch_records': self.epoch_records,
        }

    def save(self, epoch, max_to_keep=None):
        selection = 'TEST_SELECTED_MAE_DIAGNOSTIC' if epoch == 'best' else (
            'FIXED_FINAL_EPOCH' if epoch == 'final' else 'EPOCH_BOUNDARY_RECOVERY')
        atomic_checkpoint(self.results_folder / f'model-{epoch}.pt', self._checkpoint_payload(selection))

    def _write_metrics(self):
        # Full small history is in the committed checkpoint; reconcile after interrupted writes.
        atomic_write(self.results_folder / 'metrics.jsonl', ''.join(
            json.dumps(record, allow_nan=False) + '\n' for record in self.epoch_records))

    def load(self, resume_path=None, pretrained_path=None):
        if bool(resume_path) == bool(pretrained_path):
            raise ValueError('Specify exactly one of resume_path and pretrained_path.')
        # These are explicitly selected local project checkpoints, including Python/NumPy RNG state.
        data = torch.load(resume_path or pretrained_path, map_location='cpu', weights_only=False)
        if resume_path:
            required = {'checkpoint_format', 'next_epoch', 'opt', 'scheduler', 'rng', 'best_mae',
                        'best_epoch', 'global_step', 'contract', 'epoch_records', 'scaler', 'epoch'}
            if not required <= data.keys() or data['checkpoint_format'] != 2:
                raise ValueError('Legacy checkpoint cannot resume exactly; use --pretrained for an explicitly new run.')
            if data['contract'] != self.contract:
                raise ValueError('Resume configuration differs from the checkpoint training contract.')
            if any(data.get(key) != value for key, value in self.model_provenance.items()):
                raise ValueError('Resume architecture or loaded pretrained provenance differs from the checkpoint.')
            if self.opt is None or data['opt'] is None:
                raise ValueError('Exact resume requires optimizer state.')
            if (self.scheduler is None) != (data['scheduler'] is None):
                raise ValueError('Resume scheduler state does not match the trainer.')
        self.accelerator.unwrap_model(self.model).load_state_dict(data['model'], strict=True)
        if resume_path:
            self.opt.load_state_dict(data['opt'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(data['scheduler'])
            if self.accelerator.scaler is not None:
                if data['scaler'] is None:
                    raise ValueError('Mixed-precision resume requires scaler state.')
                self.accelerator.scaler.load_state_dict(data['scaler'])
            self.cur_epoch = int(data['epoch'])
            self.next_epoch = int(data['next_epoch'])
            if self.next_epoch != self.cur_epoch + 1 or not 0 <= self.next_epoch <= self.train_num_epoch:
                raise ValueError('Invalid checkpoint epoch boundary.')
            self.global_step = int(data['global_step'])
            self.best_mae, self.best_epoch = float(data['best_mae']), data['best_epoch']
            self.epoch_records = data['epoch_records']
            if len(self.epoch_records) != self.next_epoch:
                raise ValueError('Checkpoint metric history does not match completed epochs.')
            self._write_metrics()
            restore_rng_state(data['rng'])
            self.logger.info('Resuming at epoch %s, global step %s', self.next_epoch, self.global_step)

    @torch.inference_mode()
    @run_on_seed
    def _evaluate(self, model, test_data_loader, time_ensemble, thresholding=False, save_to=None):
        model = self.accelerator.unwrap_model(model)
        model.eval()
        maes = []
        if save_to is not None:
            Path(save_to).mkdir(parents=True, exist_ok=True)
        for data in tqdm(test_data_loader, disable=not sys.stderr.isatty(), mininterval=30):
            gt = [normalize_gt_mask(np.asarray(item, dtype=np.float32)) for item in data['gt']]
            image = data['image'].to(self.accelerator.device).squeeze(1)
            trace = data['trace'].to(self.accelerator.device).squeeze(1)
            options = {'time_ensemble': time_ensemble, 'verbose': False}
            if time_ensemble:
                options['gt_sizes'] = [item.shape for item in gt]
            outputs = self.train_val_forward_fn(model, image=image, trace=trace, **options)
            for target, prediction, name in zip(gt, outputs['pred_gt'], data['name']):
                if not time_ensemble:
                    prediction = F.interpolate(prediction.unsqueeze(0), size=target.shape,
                                               mode='bilinear', align_corners=False)
                    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
                prediction = prediction.detach().cpu().numpy().reshape(target.shape)
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(f'Non-finite prediction: {name}')
                if thresholding:
                    prediction = (prediction > 0.5).astype(np.float32)
                maes.append(finite(np.abs(prediction - target).mean(), 'validation MAE'))
                if save_to is not None:
                    # Store the actual probability image with fixed [0,1] scaling.
                    pixels = np.rint(np.clip(prediction, 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(pixels).save(Path(save_to) / (Path(name).stem + '.png'))
            if self.train_loader is not None:
                self.write_status('VALIDATING', force=False, validation_images=len(maes))
        if not maes or len(maes) != len(test_data_loader.dataset):
            raise RuntimeError('Validation did not cover every dataset image exactly once.')
        mae = finite(np.mean(maes), 'mean validation MAE')
        return mae, min(self.best_mae, mae)

    def val_time_ensemble(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        return self._evaluate(model, test_data_loader, True, thresholding, save_to)

    def val(self, model, test_data_loader, accelerator, thresholding=False, save_to=None):
        return self._evaluate(model, test_data_loader, False, thresholding, save_to)

    def val_batch_ensemble(self, *args, **kwargs):
        raise ValueError('Batch ensemble is outside this reproduction protocol; use time ensemble.')

    def train(self):
        if self.train_loader is None or self.test_loader is None or self.opt is None:
            raise ValueError('Training requires train/test loaders and an optimizer.')
        if len(self.train_loader) == 0 or len(self.test_loader) == 0:
            raise ValueError('Training and diagnostic datasets must be nonempty.')
        self.write_status('RUNNING')
        try:
            for epoch in range(self.next_epoch, self.train_num_epoch):
                self.cur_epoch = epoch
                self.train_loader.set_epoch(epoch)
                self.model.train()
                started = time.monotonic()
                loss_sum, image_count = 0.0, 0
                lr = finite(self.opt.param_groups[0]['lr'], 'learning rate')
                for batch_index, data in enumerate(self.train_loader):
                    self.opt.zero_grad(set_to_none=True)
                    with self.accelerator.autocast():
                        loss = fill_args_from_dict(self.train_val_forward_fn, data)(model=self.model)
                    loss_value = finite(loss.detach().item(), 'training loss')
                    self.accelerator.backward(loss)
                    norm = self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    finite(norm, 'gradient norm')
                    self.opt.step()
                    self.global_step += 1
                    batch_size = int(data['image'].shape[0])
                    loss_sum += loss_value * batch_size
                    image_count += batch_size
                    self.write_status('RUNNING', force=batch_index == 0, batch=batch_index + 1,
                                      batches_per_epoch=len(self.train_loader), loss=loss_value, lr=lr)
                self.opt.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.write_status('VALIDATING', training_loss=loss_sum / image_count, lr=lr)
                mae, _ = self.val_time_ensemble(self.model, self.test_loader, self.accelerator)
                improved = mae < self.best_mae
                if improved:
                    self.best_mae, self.best_epoch = mae, epoch
                self.next_epoch = epoch + 1
                record = {
                    'epoch': epoch, 'completed_epochs': self.next_epoch, 'global_step': self.global_step,
                    'train_images': image_count, 'train_loss': finite(loss_sum / image_count, 'epoch loss'),
                    'lr': lr, 'next_lr': finite(self.opt.param_groups[0]['lr'], 'next learning rate'),
                    'diagnostic_test_mae': mae, 'test_selected_best_mae': self.best_mae,
                    'test_selected_best_epoch': self.best_epoch,
                    'elapsed_seconds': time.monotonic() - started,
                    'primary_selection': 'FIXED_FINAL_EPOCH',
                }
                self.epoch_records.append(record)
                if improved:
                    self.save('best')
                self.save('last')
                if self.save_every > 0 and self.next_epoch < self.train_num_epoch and self.next_epoch % self.save_every == 0:
                    self.save(epoch)
                self._write_metrics()
                self.logger.info('Epoch %d/%d loss=%.6f diagnostic_test_mae=%.6f lr=%.9g',
                                 epoch, self.train_num_epoch - 1, record['train_loss'], mae, lr)
                self.accelerator.log(record, step=self.global_step)
                self.write_status('RUNNING', **{'last_epoch_metrics': record})
            self.save('final')
            final_alias = self.results_folder / f'model-{self.train_num_epoch - 1}.pt'
            temporary_alias = final_alias.with_name(final_alias.name + f'.{os.getpid()}.tmp')
            if temporary_alias.exists():
                temporary_alias.unlink()
            os.link(self.results_folder / 'model-final.pt', temporary_alias)
            os.replace(temporary_alias, final_alias)
            self.write_status('COMPLETED', primary_checkpoint=str(self.results_folder / 'model-final.pt'))
            self.logger.info('Training complete; fixed-final checkpoint: model-final.pt')
        except BaseException as error:
            self.write_status('FAILED', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.accelerator.end_training()
