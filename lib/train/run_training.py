import argparse
import importlib
import os
import random

import cv2 as cv
import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist

import _init_paths
import lib.train.admin.settings as ws_settings


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Random seed: {seed}')


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None,
                 base_seed=42, use_lmdb=False, use_wandb=False):
    """Run one LMFTrack training configuration."""
    cv.setNumThreads(0)
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if save_dir is None:
        save_dir = os.path.join('output', config_name)
    save_dir = os.path.abspath(save_dir)

    process_seed = base_seed + local_rank if local_rank != -1 else base_seed
    init_seeds(process_seed)

    print(f'script_name: {script_name}.py  config_name: {config_name}.yaml')
    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.project_path = f'train/{script_name}/{config_name}'
    settings.local_rank = local_rank
    settings.save_dir = save_dir
    settings.use_lmdb = bool(use_lmdb)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    settings.cfg_file = os.path.join(project_root, 'experiments', script_name, f'{config_name}.yaml')
    settings.use_wandb = bool(use_wandb)

    expression_module = importlib.import_module('lib.train.train_script')
    expression_module.run(settings)


def main():
    parser = argparse.ArgumentParser(description='Run LMFTrack training.')
    parser.add_argument('--script', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cudnn_benchmark', type=int, choices=[0, 1], default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('LMFTrack training requires a CUDA-capable GPU.')

    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)

    run_training(
        args.script,
        args.config,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        local_rank=args.local_rank,
        save_dir=args.save_dir,
        base_seed=args.seed,
        use_lmdb=bool(args.use_lmdb),
        use_wandb=bool(args.use_wandb),
    )


if __name__ == '__main__':
    main()
