#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "Installing the LMFTrack environment"
echo "========================================"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Error: Python was not found. Please activate a Python 3.9 environment first."
    exit 1
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Detected Python ${PYTHON_VERSION}"

if [[ "${PYTHON_VERSION}" != "3.9" ]]; then
    echo "Warning: LMFTrack was tested with Python 3.9."
fi

${PYTHON_BIN} -m pip install --upgrade "pip==23.3.1" "setuptools==68.2.2" "wheel==0.41.2"

# PyTorch 1.13.1 with CUDA 11.7.
# Change cu117 to a compatible build when using a different CUDA environment.
${PYTHON_BIN} -m pip install \
    "torch==1.13.1+cu117" \
    "torchvision==0.14.1+cu117" \
    "torchaudio==0.13.1" \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Core scientific-computing and computer-vision dependencies.
${PYTHON_BIN} -m pip install \
    "numpy==1.24.3" \
    "scipy==1.10.1" \
    "pandas==2.0.3" \
    "matplotlib==3.7.5" \
    "opencv-python==4.9.0.80" \
    "pillow==10.2.0" \
    "PyYAML==6.0.1" \
    "tqdm==4.65.0"

# Core tracking and model dependencies.
${PYTHON_BIN} -m pip install \
    "timm==0.9.16" \
    "einops==0.7.0" \
    "yacs==0.1.8" \
    "easydict==1.12" \
    "attributee==0.1.8" \
    "lmdb==1.4.1" \
    "pycocotools==2.0.7" \
    "jpeg4py==0.1.4" \
    "fvcore==0.1.5.post20221221" \
    "iopath==0.1.10" \
    "tensorboardX==2.6.2.2" \
    "thop==0.1.1.post2209072238"

# Dependencies required by OpenAI CLIP.
${PYTHON_BIN} -m pip install \
    "ftfy==6.2.3" \
    "regex==2024.9.11"

# Official OpenAI CLIP implementation.
# The installed environment reports the package as clip==1.0.
${PYTHON_BIN} -m pip install "git+https://github.com/openai/CLIP.git"

echo "========================================"
echo "LMFTrack dependencies were installed."
echo "Verify the environment with:"
echo "python -c \"import torch, clip; print(torch.__version__); print(clip.__file__)\""
echo "========================================"
