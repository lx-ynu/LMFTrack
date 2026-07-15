#!/usr/bin/env bash
set -euo pipefail
python tracking/train.py --script lmftrack --config lmftrack_lasher --save_dir ./output/lmftrack_lasher --mode single
