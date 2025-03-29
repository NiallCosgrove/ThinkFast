import os
from huggingface_hub import snapshot_download

# Define the local directory for the model download
model_dir = "./model-LLama-4bit/"

# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

try:
    # Download the model from Hugging Face into the specified directory.
    snapshot_download(
        repo_id="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",   #unsloth/Llama-3.2-3B-Instruct",         
        local_dir=model_dir,
        local_dir_use_symlinks=False  # Downloads a full copy rather than using symlinks
    )
    print(f"Model downloaded successfully to '{model_dir}'.")
except Exception as error:
    print(f"An error occurred during download: {error}")
