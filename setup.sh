#!/bin/bash
set -e  # Stop on error

echo "🚀 Setting up environment..."

# Ensure Python 3.11 is installed
if ! python3.11 --version &>/dev/null; then
    echo "⚠️ Python 3.11 not found! Installing..."
    sudo apt update && sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
fi

# Use Python 3.11 explicitly
PYTHON=python3.11

# Create & activate virtual environment
$PYTHON -m venv thinkfast-env
source thinkfast-env/bin/activate

# Ensure pip is up to date
pip install --upgrade pip

# Install correct PyTorch version for CUDA
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ CUDA detected! Installing GPU-accelerated PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️ No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "To activate, run: source thinkfast-env/bin/activate"

