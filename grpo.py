# -*- coding: utf-8 -*-
#### Portions of this code are derived from Unsloth's blog and open-source repository
# Copyright 2024 Unsloth AI (Apache 2.0 License)
# Original source: https://github.com/unslothai

"""
Load model , and set parameters
"""

from unsloth import FastLanguageModel # FastQwen2Model reports missing LORA_REQUEST_ID - might be woth finding out why? 
import os
import glob
import argparse
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch.distributed as dist
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

#DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SAVE_PATH = "./model"
TOKENIZER_CONFIG_PATH = "qwen2.5-reasoning-template-config.json"

# for Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
def get_latest_checkpoint(output_dir="outputs"):
    """Finds the latest checkpoint directory based on the highest checkpoint number."""
    checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=os.path.getmtime, reverse=True)
    return checkpoints[0] if checkpoints else None  # Return the latest checkpoint, or None if none exist

def setup_model(model):
    """Ensure the model is cached locally and return its exact path."""

    # Step 1: Ensure model is cached locally
    if not os.path.exists(SAVE_PATH):
        print(f"Downloading model: {model} to {SAVE_PATH}")

        # Cache the model and get the actual path
        model_dir = snapshot_download(
            repo_id=model,
            cache_dir=SAVE_PATH,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            ignore_patterns=["*.bin"]
        )

        print(f"Model cached at: {model_dir}")
        exit()
    else:
        # Model directory already exists, find the latest snapshot by modification time
        base_path = os.path.join(SAVE_PATH, f"models--{model.replace('/', '--')}")
        snapshots_path = os.path.join(base_path, "snapshots")

        if os.path.exists(snapshots_path):
            snapshot_dirs = [os.path.join(snapshots_path, d) for d in os.listdir(snapshots_path)]
            snapshot_dirs = [d for d in snapshot_dirs if os.path.isdir(d)]
            
            # Sort by modification time (most recent first)
            latest_snapshot = max(snapshot_dirs, key=os.path.getmtime, default=None)

            if latest_snapshot:
                model_dir = latest_snapshot
            else:
                raise RuntimeError(f"No valid snapshots found in {snapshots_path}")
        else:
            raise RuntimeError(f"Could not determine model directory inside {SAVE_PATH}")

    # Step 2: Replace tokenizer config if it's a Qwen2.5 model
    if "Qwen2.5" in model:
        print(f"Replacing tokenizer_config.json in {model_dir}")
        os.system(f"cp {TOKENIZER_CONFIG_PATH} {model_dir}/tokenizer_config.json")
    else:
        print(f"Warning: Model {model} is not Qwen2.5. Skipping tokenizer modification.")

    print("Model setup complete.")
    return model_dir  # Pass back the exact model path

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def export_model(model=None, tokenizer=None):
    
    # Merge to 16bit
    if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")
    
    # Merge to 4bit
    if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")
    
    # Just LoRA adapters
    if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
    if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
    
    """
    
    ### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
    
    Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
    * `q8_0` - Fast conversion. High resource use, but generally acceptable.
    * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
    * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
    
    [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)
    """

    # Save to 8bit Q8_0
    if False: model.save_pretrained_gguf("model", tokenizer,)
    # Remember to go to https://huggingface.co/settings/tokens for a token!
    # And change hf to your username!
    if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")
    
    # Save to 16bit GGUF
    if True: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")
    
    # Save to q4_k_m GGUF
    if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
    if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")
    
    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "hf/model", # Change hf to your username!
            tokenizer,
            quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
            token = "",
        )


def load_model_and_tokenizer(model,max_seq_length,lora_rank, resume_from_checkpoint=False):
    model_name = setup_model(model)
    print(f"{model_name} should now be present and ready for fine tuning")

    print(f"Loading model from: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.95, # Reduce if out of memory
        enforce_eager = True   # SWA conflict so disable during training
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
        
    )

    model.config.sliding_window = None  # Disables SWA during training
    return model,tokenizer

def training_setup( model, tokenizer, training_dataset, max_seq_length, save_steps, max_steps=-1, num_train_epochs=1, resume_from_checkpoint = False):

    max_prompt_length = 256
    
    training_args = GRPOConfig(
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        logging_steps = 10,
        per_device_train_batch_size =1,
        gradient_accumulation_steps = 1, # Increase to 4 for smoother training
        num_generations = 2, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        num_train_epochs = num_train_epochs, # Set to 1 for a full training run
        max_steps = max_steps,
        save_steps = save_steps,
        max_grad_norm = 0.1,
        report_to = "none", # Can use Weights & Biases
        resume_from_checkpoint=resume_from_checkpoint,
        output_dir = "outputs",
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = training_dataset
    )
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen model with GRPO.")

    # Model and Training Control
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model name")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")

    # Training Hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max training steps (default: -1) set this to exit training early")
    parser.add_argument("--save_steps", type=int, default=250, help="Checkpoint save frequency")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Context length for the model")
    parser.add_argument("--lora_rank", type=int, default=8, help="Higher is smarter, but slower 2,4,8,16,32,64...")

    args = parser.parse_args()

    model,tokenizer = load_model_and_tokenizer(
        model=args.model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        resume_from_checkpoint=args.resume,
    )


    """### Data Prep
    leverage [@willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) for data prep and all reward functions.
    """
    dataset = get_gsm8k_questions()
    """
    ### Train the model
    Now set up GRPO Trainer and all configurations!
    """
    trainer = training_setup(
        resume_from_checkpoint=args.resume,
        model=model,
        tokenizer=tokenizer,
        training_dataset=dataset,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    
    # needed to make a graceful exit from multi-gpu
    if dist.is_initialized():
        print("Destroying NCCL process group before exit...")
        dist.destroy_process_group()

    print("Done.")


    #save the lora and export as gguf
    #todo: export model needs work
    model.save_lora("grpo_saved_lora")
    export_model(model=model,tokenizer=tokenizer)


if __name__ == "__main__":
    main()
