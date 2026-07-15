#!/usr/bin/env bash
set -euo pipefail
python tracking/test.py lmftrack lmftrack_lasher --dataset_name gtot --threads 4 --num_gpus 1 --runid 15
