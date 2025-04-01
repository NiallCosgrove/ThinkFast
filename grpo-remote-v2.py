# -*- coding: utf-8 -*-
"""
Load model , and set parameters
"""

# Unsloth + GRPO
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# Core libraries
import os
import re
import glob
import json
import argparse
import random

# PyTorch & distributed training
import torch
import torch.distributed as dist

# Hugging Face & Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from datasets import load_dataset, Dataset


DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

SAVE_PATH = "model"
TOKENIZER_CONFIG_PATH = "llama3.2-reasoning-template-config.json"

SYSTEM_PROMPT1 = """
### Use chain-of-thought reasoning to solve the problem.
### Then explain your answer.
### Your reasoning should be delimited with <|reasoning|> and <|/reasoning|>.
### Your explanation to the user should be delimited with <|explanation|> and <|/explanation|>.
### Your final answer should be delimited with <|answer|> and <|/answer|>.
### There should be no text that is not delimited.

## Respond in exactly the following format:

<|reasoning|>
# Your internal reasoning goes here.
...
<|/reasoning|>

<|explanation|>
# Your explanation for the user goes here.
...
<|/explanation|>

<|answer|>
# Your final answer (usually a number, choice, or short sentence).
...
<|/answer|>
"""
SYSTEM_PROMPT2 = """
Respond in the following format:

<|reasoning|>
...
<|/reasoning|>
<|explanation|>
...
<|/explanation|>
<|answer|>
...
<|/answer|>
"""

XML_COT_FORMAT = """\
<|reasoning|>
{reasoning}
<|/reasoning|>
<|explanation|>
{explanation}
<|answer|>
{answer}
<|/answer|>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<|answer|>")[-1]
    answer = answer.split("<|/answer|>")[0]
    return answer.strip()

def extract_numerical_answer(text: str) -> str:
    ans = extract_xml_answer(text)
    numbers = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", ans)
    return numbers[-1].strip() if numbers else "0"

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def choose_prompt_by_index(i: int, total: int) -> str:
    frac = i / total
    if frac < 0.10:
        return SYSTEM_PROMPT1 if random.random() < 0.5 else SYSTEM_PROMPT2
    elif frac < 0.20:
        # Linear decay from 0.5 to 0.0
        p = 0.5 * (1 - (frac - 0.10) / 0.10)
        return SYSTEM_PROMPT1 if random.random() < p else SYSTEM_PROMPT2
    else:
        return SYSTEM_PROMPT2

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.shuffle()
    total = len(data)

    def add_prompt(x, i):
        return {
            'prompt': [
                {'role': 'user', 'content': choose_prompt_by_index(i, total) + "\n" + x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        }

    data = data.map(add_prompt, with_indices=True)  # type: ignore
    return data  # type: ignore

def normalise_number(text: str) -> float:
    # Strip formatting noise
    cleaned = text.replace(",", "").replace("$", "").strip()
    return float(cleaned)

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_numerical_answer(r) for r in responses]
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if normalise_number(r) ==normalise_number(a) else 0.0 for r, a in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks whether the completion adheres to strict tag format:
    - Starts with <|reasoning|>...</|reasoning|>
    - Followed by <|explanation|>...</|explanation|>
    - Followed by <|answer|>...</|/answer|>
    - Tags must be in order, with up to two optional blank lines between sections.
    """

    pattern = (
        r"^<\|reasoning\|\>\n[\s\S]*?\n<\|/reasoning\|\>(?:\n){0,2}"
        r"<\|explanation\|\>\n[\s\S]*?\n<\|/explanation\|\>(?:\n){0,2}"
        r"<\|answer\|\>\n[\s\S]*?\n<\|/answer\|\>\n?$"
    )

    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if reasoning, explanation, and answer blocks appear
    in the correct order anywhere in the output.
    """
    pattern = (
        r"<\|reasoning\|\>[\s\S]*?<\|/reasoning\|\>\s*"
        r"<\|explanation\|\>[\s\S]*?<\|/explanation\|\>\s*"
        r"<\|answer\|\>[\s\S]*?<\|/answer\|\>"
    )
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<|reasoning|>\n") == 1:
        count += 0.125
    if text.count("\n<|/reasoning|>\n") == 1:
        count += 0.125
    if text.count("<|explanation|>\n") == 1:
        count += 0.125
    if text.count("\n<|/explanation|>\n") == 1:
        count += 0.125
    if text.count("\n<|answer|>\n") == 1:
        count += 0.125
        count -= len(text.split("\n<|/answer|>\n")[-1]) * 0.001
    if text.count("\n<|/answer|>") == 1:
        count += 0.125
        count -= (len(text.split("\n<|/answer|>")[-1]) - 1) * 0.001
    # print(f"xml:{count}")
    return max(0, count)

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Applies count_xml() to each completion to compute partial format reward.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def export_model(model, tokenizer=None,export=False):
    
    # Merge to 16bit
    if export:
        model.save_pretrained_merged("model-merged", tokenizer, save_method = "merged_16bit",maximum_memory_usage=.01)
        

def get_latest_checkpoint(output_dir="outputs"):
    """Finds the latest checkpoint directory based on the highest checkpoint number."""
    """
    Unused at present — may be useful for automatically resuming from
    latest checkpoint in shell scripts or interactive sessions.
    """
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
            #raise RuntimeError(f"Could not determine model directory inside {SAVE_PATH}")
            print(f"Could not determine model directory inside {SAVE_PATH} - assuming single model.")
            model_dir = SAVE_PATH

    # Step 2: Replace tokenizer config 
    original_config = os.path.join(model_dir, "tokenizer_config.json")
    backup_config = os.path.join(model_dir, "tokenizer_config.original.json")

    if not os.path.exists(backup_config):
        print(f"Backing up original tokenizer config to {backup_config}")
        os.system(f"cp {original_config} {backup_config}")

    print(f"Replacing tokenizer_config.json in {model_dir}")
    os.system(f"cp {TOKENIZER_CONFIG_PATH} {model_dir}/tokenizer_config.json")

    print("Model setup complete.")
    return model_dir  # Pass back the exact model path


def load_model_and_tokenizer(model, max_seq_length, lora_rank):
    model_name = setup_model(model)
    print(f"{model_name} should now be present and ready for fine tuning")
    print(f"Loading model from: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,  # False for LoRA 16bit
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 1,  # Reduce if out of memory
    )

    SPECIAL_TAG_TOKENS = [
        "<|reasoning>|", "<|/reasoning|>",
        "<|explanation|>", "<|/explanation|>",
        "<|answer>|", "<|/answer|>",
    ]

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TAG_TOKENS})
    print(f"Added {num_added} special tokens to tokenizer.")

    # Future expansion:
    # Consider adding tokens like "<reasoning type='" and "'>" to support structured attributes
    # without flattening the semantic meaning of the type labels.
    # This allows compositional tag construction while keeping 'deductive', etc., semantically meaningful.

    model.resize_token_embeddings(len(tokenizer),mean_resizing=True)
    print(f"Resized model embeddings to match vocab size: {len(tokenizer)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = [
            "gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth",
    )

    return model, tokenizer

def training_setup( model, tokenizer, training_dataset, max_seq_length, save_steps, batch_size, grad_accum, max_prompt_length, max_steps=-1, num_train_epochs=1, resume_from_checkpoint = False):

        
    training_args = GRPOConfig(
        reward_weights = [1.0,0.4,0.6,2.0],  #["xmlcount_reward_func": 1.0,"soft_format_reward_func": 0.4,"strict_format_reward_func": 0.6,"correctness_reward_func": 2.0,],
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "constant", #"linear",  #"cosine",
        warmup_steps=0,
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum, # Increase to 4 for smoother training
        num_generations = 4, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_seq_length - max_prompt_length,
        num_train_epochs = num_train_epochs, # Set to 1 for a full training run
        max_steps = max_steps,   # -1 for epoch training
        save_steps = save_steps,
        max_grad_norm = 0.1,
        report_to = "none",
        output_dir = "outputs",
        logging_dir= "./runs",
        dataloader_num_workers=16,
        torch_compile=True,
        torch_compile_mode='max-autotune',
        #resume_from_checkpoint=resume_from_checkpoint,
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            correctness_reward_func,
        ],
        args = training_args,
        train_dataset = training_dataset
    )
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with GRPO.")

    # Model and Training Control
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model name")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")

    # Training Hyperparameters
    parser.add_argument( "--max_prompt_length", type=int, default=256, help="Maximum length of system prompt + question combined. Used to calculate completion length.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max training steps (default: -1) set this to exit training early")
    parser.add_argument("--save_steps", type=int, default=20, help="Checkpoint save frequency")
    parser.add_argument("--batch_size", type=int, default=96, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=1536, help="Context length for the model")
    parser.add_argument("--lora_rank", type=int, default=32, help="Higher is smarter, but slower 8,16,32,64...")
    parser.add_argument("--export", type=bool, default=True, help="Export merged BFloat16 when done.")

    args = parser.parse_args()
    open("training.log", "w").close()  # Clears file at start of run
    
    model,tokenizer = load_model_and_tokenizer(
        model=args.model,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
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
        model=model,
        tokenizer=tokenizer,
        training_dataset=dataset,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_prompt_length=args.max_prompt_length,
    )

    from transformers import TrainerCallback

    class LogRewardsOnlyCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "reward" in logs:
                with open("training.log", "a", buffering=1) as f:
                    f.write(f"{logs}\n")

    trainer.add_callback(LogRewardsOnlyCallback())



    trainer.train(resume_from_checkpoint=args.resume)
    
    # needed to make a graceful exit from multi-gpu
    if dist.is_initialized():
        print("Destroying NCCL process group before exit...")
        dist.destroy_process_group()

    print("Done.")


    #save the lora and export as gguf
    #todo: export model needs work
    #model.save_lora("grpo_saved_lora")
    export_model(model=model,tokenizer=tokenizer,export=args.export)


if __name__ == "__main__":
    main()
