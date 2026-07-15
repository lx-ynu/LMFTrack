#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "Installing the LMFTrack environment"
echo "========================================"

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: Conda was not found. Please install Miniconda or Anaconda first."
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo "Error: Python was not found. Please activate the lmftrack environment first."
    echo "  conda create -n lmftrack python=3.9 -y"
    echo "  conda activate lmftrack"
    exit 1
fi

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Detected Python ${PYTHON_VERSION}"
if [[ "${PYTHON_VERSION}" != "3.9" ]]; then
    echo "Warning: the paper experiments used Python 3.9."
fi

# PyTorch environment used for the paper experiments.
conda install -y \
    pytorch==1.10.0 \
    torchvision==0.11.0 \
    torchaudio==0.10.0 \
    cudatoolkit=11.3 \
    -c pytorch \
    -c conda-forge

python -m pip install --upgrade "pip==23.3.1" "setuptools==68.2.2" "wheel==0.41.2"
python -m pip install -r requirements.txt

echo "========================================"
echo "LMFTrack dependencies were installed."
echo "Verify the environment with:"
echo "python -c \"import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__)\""
echo "========================================"
