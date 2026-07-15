import argparse
import os
import random
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Train LMFTrack.')
    parser.add_argument('--script', default='lmftrack', choices=['lmftrack'])
    parser.add_argument('--config', default='lmftrack_lasher', help='Configuration name without .yaml.')
    parser.add_argument('--save_dir', default='./output/lmftrack_lasher',
                        help='Root directory for checkpoints, logs, and TensorBoard files.')
    parser.add_argument('--mode', choices=['single', 'multiple'], default='single')
    parser.add_argument('--nproc_per_node', type=int, default=1, help='Number of GPUs for distributed training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_lmdb', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    run_script = os.path.join('lib', 'train', 'run_training.py')
    common_args = [
        '--script', args.script,
        '--config', args.config,
        '--save_dir', args.save_dir,
        '--seed', str(args.seed),
        '--use_lmdb', str(int(args.use_lmdb)),
        '--use_wandb', str(int(args.use_wandb)),
    ]

    if args.mode == 'single':
        command = [sys.executable, run_script] + common_args
    else:
        if args.nproc_per_node < 1:
            raise ValueError('--nproc_per_node must be at least 1.')
        command = [
            sys.executable, '-m', 'torch.distributed.launch',
            '--nproc_per_node', str(args.nproc_per_node),
            '--master_port', str(random.randint(10000, 50000)),
            run_script,
        ] + common_args

    print('Running:', ' '.join(command))
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
