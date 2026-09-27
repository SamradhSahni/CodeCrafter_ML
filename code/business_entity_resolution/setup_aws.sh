#!/bin/bash
# ==============================================================================
# AWS EC2 Turnkey Setup Script for Business Entity Resolution
# Target Instance: g5.4xlarge or g5.2xlarge (Ubuntu 22.04 / Deep Learning AMI)
# ==============================================================================
set -e

echo "=== [1/5] Updating system packages and installing prerequisites ==="
sudo apt-get update -y
sudo apt-get install -y build-essential python3-dev python3-pip python3-venv tmux htop git zip unzip libgomp1

echo "=== [2/5] Checking GPU status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed."
fi

echo "=== [3/5] Setting up Python virtual environment ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "=== [4/5] Installing PyTorch (CUDA 12) & FAISS-GPU ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Install GPU FAISS for CUDA 12
pip install faiss-gpu-cu12 || pip install faiss-gpu || pip install faiss-cpu

echo "=== [5/5] Installing project requirements ==="
pip install -r requirements.txt

echo "=== [Verification] Checking environment ==="
python3 -c "
import torch, faiss, psutil, os
print('=' * 50)
print(f'PyTorch Version  : {torch.__version__}')
print(f'CUDA Available   : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device       : {torch.cuda.get_device_name(0)}')
    print(f'GPU VRAM         : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')
print(f'System RAM       : {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'CPU Cores        : {os.cpu_count()}')
print(f'FAISS Version    : {faiss.__version__}')
has_gpu_faiss = hasattr(faiss, 'StandardGpuResources')
print(f'FAISS GPU Support: {has_gpu_faiss}')
print('=' * 50)
print('ALL DEPENDENCIES INSTALLED AND READY!')
"

echo "=============================================================================="
echo " Setup complete! To run the full pipeline:"
echo "   source venv/bin/activate"
echo "   ./run_aws.sh"
echo "=============================================================================="
