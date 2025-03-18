#!/bin/bash
set -e  # Stop on error

echo "🚀 Setting up environment..."

# Ensure Python & venv exist
if ! command -v python3 &>/dev/null; then
    echo "⚠️ Python3 not found, installing..."
    sudo apt update && sudo apt install -y python3 python3-venv python3-pip
fi

# Create & activate virtual environment
python3 -m venv thinkfast-env
source thinkfast-env/bin/activate

# Install correct PyTorch version for CUDA
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ CUDA detected! Installing GPU-accelerated PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️ No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install other dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "To activate, run: source thinkfast-env/bin/activate"
