import os
import sys

# Keep the project environment authoritative, including for worker subprocesses.
if __name__ == '__main__':
    os.environ['PYTHONNOUSERSITE'] = '1'
    if not sys.flags.no_user_site:
        os.execv(sys.executable, [sys.executable, '-s', *sys.argv])

from utils import init_env

import argparse
from pathlib import Path

import torch
from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from torch.utils.data import DataLoader
from utils.trainer import Trainer

def get_loader(cfg):
    # 训练集配置在 cfg.train_dataset 中，实际类名和参数都由 YAML 决定。
    train_dataset = instantiate_from_config(cfg.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True)

    # Mix 仅用于诊断，主结果固定使用训练结束的 checkpoint；每张图只评估一次。
    test_dataset = instantiate_from_config(cfg.test_dataset.Mix)

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./', help='Run directory for checkpoints, metrics and status.')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_with', choices=('none', 'wandb'), default='none')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--status_interval_seconds', type=float, default=60)

    cfg = add_args(parser)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise RuntimeError('This reproduction requires one process and one GPU.')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one available GPU before launch.')
    if cfg.num_epoch < 1 or cfg.batch_size < 1 or cfg.gradient_accumulate_every != 1:
        raise ValueError('Require positive epochs/batch size and gradient_accumulate_every=1.')
    if cfg.resume and cfg.pretrained:
        raise ValueError('Specify either --resume or --pretrained, not both.')
    result_path = Path(cfg.results_folder)
    if not cfg.resume and (result_path / 'model-last.pt').exists():
        raise FileExistsError('Existing training checkpoint: use --resume or a new results folder.')
    set_random_seed(cfg.seed)

    config_pretty(cfg)

    # cond_uvit 是遗留的配置接口；当前默认配置中它是 EmptyObject，不承担实际网络计算。
    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    # 主干网络由 cfg.model 指定，默认是 model.net.net，即两路 PVT + 融合 + 双分支解码器。
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)
    # 扩散外壳负责给 gt/de 加噪、采样时间步、计算训练目标和损失。
    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)


    train_loader, test_loader = get_loader(cfg)

    # 优化器只更新主网络参数；diffusion_model 只是包装器，本身不额外定义可训练主干。
    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epoch, eta_min=cfg.lr_min)

    # Trainer 统一封装训练循环、验证、checkpoint、accelerate 和日志记录。
    trainer = Trainer(
        diffusion_model, train_loader, test_loader,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer, scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        fp16=cfg.fp16,
        log_with=None if cfg.log_with == 'none' else cfg.log_with,
        cfg=cfg,
    )
    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        # resume 完整恢复训练状态，从下一个 epoch 继续；pretrained 只加载权重。
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)
    trainer.train()
