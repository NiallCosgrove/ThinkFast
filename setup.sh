#!/bin/bash
set -e  # Stop on error

echo "🚀 Setting up environment..."

# Ensure Python 3.11 is installed
if ! python3.11 --version &>/dev/null; then
    echo "⚠️ Python 3.11 not found! Installing..."
    apt update && apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
fi

# needed to build llama.cpp  - see the build_llama.sh script
apt update && apt install -y build-essential cmake libssl-dev libfftw3-dev emacs   # emacs is technically optional (for some ^^)

# Use Python 3.11 explicitly
PYTHON=python3.11
export PATH="/usr/bin/python3.11:$PATH"

# Create & activate virtual environment
if [ ! -d "thinkfast-env" ]; then
    $PYTHON -m venv thinkfast-env
fi
source thinkfast-env/bin/activate

# Ensure pip is up to date
pip install --upgrade pip

# Install correct PyTorch version for CUDA
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "✅ CUDA detected! Installing GPU-accelerated PyTorch..."
    pip install torch transformers huggingface_hub unsloth trl accelerate vllm
else
    echo "⚠️ No CUDA detected,!!!  unsloth needs cuda!!!⚠️"
    exit 1
fi

# Install other dependencies
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo "To activate, run: source thinkfast-env/bin/activate"
echo "run build_llama.sh next or grpo.py wont be able to export to gguf!"
