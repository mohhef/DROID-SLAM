#!/bin/bash
# Install script for DROID-SLAM
# This script sets up the conda environment and downloads model weights

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== DROID-SLAM Installation ==="

# Initialize git submodules if needed
if [ ! -f "thirdparty/lietorch/setup.py" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

# Create conda environment if it doesn't exist
CONDA_ENV="droidslam"
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    echo "Creating conda environment: ${CONDA_ENV}"
    conda create -n ${CONDA_ENV} python=3.10 -y
fi

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install CUDA toolkit to match PyTorch CUDA version (fixes build issues)
echo "Installing CUDA toolkit..."
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y

# Install requirements
echo "Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Install third-party modules (this takes a while due to CUDA compilation)
# Force system GCC for CUDA compatibility (CUDA 11.8 requires GCC <= 11)
# This prevents issues when other conda envs have GCC 12+ installed
export CUDAHOSTCXX=/usr/bin/gcc
export CXX=/usr/bin/g++
export CC=/usr/bin/gcc

echo "Installing lietorch (CUDA-accelerated Lie algebra)..."
# Clean any cached builds to ensure fresh compilation with correct compiler
rm -rf thirdparty/lietorch/build thirdparty/lietorch/*.egg-info
pip install --no-build-isolation thirdparty/lietorch

echo "Installing pytorch_scatter..."
rm -rf thirdparty/pytorch_scatter/build thirdparty/pytorch_scatter/*.egg-info
pip install --no-build-isolation thirdparty/pytorch_scatter

# Install droid-backends
echo "Installing DROID-SLAM backends..."
pip install --no-build-isolation -e .

# Download model weights
if [ ! -f "droid.pth" ]; then
    echo "Downloading model weights..."
    pip install gdown
    gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh
else
    echo "Model weights already exist."
fi

echo ""
echo "=== Installation Complete ==="
echo "DROID-SLAM is ready to use."
echo ""
echo "Conda environment: ${CONDA_ENV}"
echo ""
echo "To test manually:"
echo "  conda activate ${CONDA_ENV}"
echo "  python demo.py --imagedir=data/sfm_bench/rgb --calib=calib/eth.txt --disable_vis"
