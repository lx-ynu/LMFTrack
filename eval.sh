#!/usr/bin/env bash
set -euo pipefail
python tracking/analysis_results.py \
    --tracker_name lmftrack \
    --tracker_param lmftrack_lasher \
    --dataset_name lasher_test \
    --runid 15 \
    --display_name LMFTrack
