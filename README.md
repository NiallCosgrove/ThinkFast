# ThinkFast

## Fine-Tuning Qwen2.5 for Efficient, Compressed Reasoning

### 📌 Overview

ThinkFast is an experiment in fine-tuning Qwen2.5 to develop structured, compressed reasoning. Instead of enforcing verbose step-by-step explanations, this project explores whether LLMs can reason efficiently—producing the correct answer while minimizing token usage.

### 🛠 Features

- ✅ RLHF Training with GRPO – Reinforcing structured reasoning with reward shaping.

- ✅ Reasoning Compression – not yet fully implimented - optimizing thought processes.

- ✅ Unconstrained Inference Compute – Testing whether extra inference cycles improve reasoning.

- ✅ Multi-GPU Capable – Fully compatible with multi-GPU fine-tuning.

- ✅ Efficient GGUF Export – Convert fine-tuned models for llama.cpp inference.


***This is a work in progress***


### 📌 Quick Start

- 1. Set up env
'''
conda create -n thinkfast python=3.11 -y
conda activate thinkfast
pip install torch transformers huggingface_hub unsloth trl accelerate vllm
pip install -r requirements.txt
'''

There is a setup.sh that might work for you if you dont have conda - not fully tested  - ymmv

- 2. Run the GRPO fune tuner
'''
python grpo.py --model <hf-model-name>  --train_steps 1000
'''
- 3. gguf export is in grpo.py  - change False to True on the lines with your preferred export formats
you will need llama.cpp in your directory - todo arg parse this






## 🏆 Attribution
This project uses portions of code from [Unsloth](https://github.com/unslothai) under the Apache 2.0 license.
Some training loop components and techniques were adapted from Unsloth’s blog posts and codebase.
See the [`LICENSE`](LICENSE) file for details.

Apache2.0 Licence
