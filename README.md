ThinkFast 🚀

Fine-Tuning Qwen2.5 for Efficient, Compressed Reasoning
📌 Overview

ThinkFast is an experiment in fine-tuning Qwen2.5 to develop structured, compressed reasoning. Instead of enforcing verbose step-by-step explanations, this project explores whether LLMs can reason efficiently—producing the correct answer while minimizing token usage.
🛠 Features

✅ RLHF Training with GRPO – Reinforcing structured reasoning with reward shaping.
✅ Reasoning Compression – Inspired by DeepSeek-R0, optimizing thought processes.
✅ Unconstrained Inference Compute – Testing whether extra inference cycles improve reasoning.
✅ Multi-GPU Capable – Fully compatible with multi-GPU fine-tuning.
✅ Efficient GGUF Export – Convert fine-tuned models for llama.cpp inference.


This is a work in progress


🚀 Quick Start
1️⃣ et Up Environment

conda create -n thinkfast python=3.11 -y
conda activate thinkfast
pip install -r requirements.txt

2️⃣ ownload & Prepare Qwen

python setup_model.py --model Qwen/Qwen2.5-1.5B-Instruct

3️⃣ ine-Tune with GRPO

python grpo.py --train_steps 1000

4. gguf export is in the grpo.py  - change False to True
you will need llama.cpp 




## 🏆 Attribution
This project uses portions of code from [Unsloth](https://github.com/unslothai) under the Apache 2.0 license.  
Some training loop components and techniques were adapted from Unsloth’s blog posts and codebase.  
See the [`LICENSE`](LICENSE) file for details.

Apache2.0 Licence
