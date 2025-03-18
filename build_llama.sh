#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Clone llama.cpp if not already present
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggml-org/llama.cpp
fi

# Enter the llama.cpp directory
cd llama.cpp

# Build llama.cpp
cmake -B build
cmake --build build --config Release -j $(nproc)

# Symlink llama-quantize to the root of llama.cpp
ln -sf build/bin/llama-quantize ./llama-quantize

echo "llama.cpp built successfully and llama-quantize symlinked."
