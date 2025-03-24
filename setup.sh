#!/bin/bash
set -e  # Stop on error

echo "🚀 Setting up environment..."
#!/usr/bin/env bash
set -e

# Install dependencies
apt-get update
apt-get install -y \
  build-essential \
  cmake \
  libssl-dev \
  libfftw3-dev \
  libopenblas-dev \
  pkg-config \
  emacs \
  git

# Clone updated llama.cpp repo
if [ ! -d llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp

# Build with CUDA, CUBLAS, OpenBLAS
cmake -B build \
  -DGGML_CUDA=on \
  -DGGML_CUBLAS=on \
  -DGGML_BLAS=on \
  -DGGML_BLAS_VENDOR=OpenBLAS

cmake --build build --config Release -j $(nproc)

# Symlink llama-quantize to the root of llama.cpp
echo "symlinking llama_quantize"
ln -sf build/bin/llama-quantize ./llama-quantize
cd ..

echo "llama.cpp built successfully and llama-quantize symlinked."

# Create & activate virtual environment
echo "activating venv"
if [ ! -d "thinkfast-env" ]; then
    python -m venv thinkfast-env
fi
source thinkfast-env/bin/activate

# Ensure pip is up to date
echo "updating pip"
pip install --upgrade pip

# Install other dependencies
echo "installing requirements"
pip install -r requirements.txt

echo "✅ Environment setup complete!"

