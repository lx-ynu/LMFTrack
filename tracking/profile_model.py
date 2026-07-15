import argparse
import importlib
import os
import sys
import time

import torch
from thop import profile
from thop.utils import clever_format

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description='Profile LMFTrack FLOPs, parameters, and inference speed.')
    parser.add_argument('--script', default='lmftrack', choices=['lmftrack'])
    parser.add_argument('--config', default='lmftrack_lasher', help='Configuration name without .yaml.')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    return parser.parse_args()


def build_inputs(batch_size, template_size, search_size, device):
    template = [
        torch.randn(batch_size, 3, template_size, template_size, device=device),
        torch.randn(batch_size, 3, template_size, template_size, device=device),
    ]
    search = [
        torch.randn(batch_size, 3, search_size, search_size, device=device),
        torch.randn(batch_size, 3, search_size, search_size, device=device),
    ]
    return template, search


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Model profiling requires a CUDA-capable GPU.')

    config_path = os.path.join(PROJECT_ROOT, 'experiments', args.script, f'{args.config}.yaml')
    config_module = importlib.import_module(f'lib.config.{args.script}.config')
    config_module.update_config_from_file(config_path)
    cfg = config_module.cfg

    from lib.models import build_lmftrack

    device = torch.device('cuda:0')
    model = build_lmftrack(cfg, training=False).to(device).eval()
    template, search = build_inputs(1, cfg.TEST.TEMPLATE_SIZE, cfg.TEST.SEARCH_SIZE, device)

    macs, params = profile(model, inputs=(template, search), verbose=False)
    macs_text, params_text = clever_format([macs, params], '%.3f')
    print(f'MACs: {macs_text}')
    print(f'Parameters: {params_text}')

    with torch.no_grad():
        for _ in range(args.warmup):
            model(template, search)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(args.iterations):
            model(template, search)
        torch.cuda.synchronize()

    average_latency = (time.time() - start) / args.iterations
    print(f'Average latency: {average_latency * 1000:.2f} ms')
    print(f'FPS: {1.0 / average_latency:.2f}')


if __name__ == '__main__':
    main()
