#!/bin/bash
set -e  # Stop on error

echo "🚀 Setting up environment..."

# needed to build llama.cpp  - see the build_llama.sh script
echo "installing build essentials"
apt update && apt install -y build-essential cmake libssl-dev libfftw3-dev emacs   # emacs is technically optional (for some ^^)

# Clone llama.cpp if not already present
echo "cloning llama.cpp"
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp
fi

# Enter the llama.cpp directory
cd llama.cpp

# Build llama.cpp
echo "building llama.cpp"
cmake -B build
cmake --build build --config Release -j $(nproc)

# Symlink llama-quantize to the root of llama.cpp
echo "symlinking llama_quantize"
ln -sf build/bin/llama-quantize ./llama-quantize
cd ..

echo "llama.cpp built successfully and llama-quantize symlinked."

# Create & activate virtual environment
echo "activating venv"
if [ ! -d "thinkfast-env" ]; then
    $PYTHON -m venv thinkfast-env
fi
source thinkfast-env/bin/activate

# Ensure pip is up to date
echo "updating pip"
pip install --upgrade pip

# Install other dependencies
echo "installing requirements"
pip install -r requirements.txt

echo "✅ Environment setup complete!"

