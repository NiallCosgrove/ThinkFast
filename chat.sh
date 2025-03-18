#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

./llama.cpp/build/bin/llama-cli -i --model ./model/unsloth.F16.gguf
