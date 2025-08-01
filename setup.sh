#!/bin/bash

set -e  # Exit immediately if a command fails

echo "📦 Installing huggingface_hub..."
pip install huggingface_hub

echo "🔐 Logging into Hugging Face (requires manual token entry)..."
huggingface-cli login

if [ ! -d "/root/centaur_model" ] || [ -z "$(ls -A /root/centaur_model)" ]; then
    echo "⬇️ Downloading Centaur-70B to /root/centaur_model..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='marcelbinz/Llama-3.1-Centaur-70B',
    local_dir='/root/centaur_model',
    local_dir_use_symlinks=False
)
"
else
    echo "✅ Model already present in /root/centaur_model. Skipping download."
fi

if [ ! -d "/root/centaur-llm-runpod" ]; then
    echo "📂 Cloning project repo..."
    git clone https://github.com/petekchoo/centaur-llm-runpod.git /root/centaur-llm-runpod
else
    echo "✅ Repo already exists at /root/centaur-llm-runpod, skipping clone."
fi

echo "📦 Installing project dependencies..."
cd /root/centaur-llm-runpod
pip install -r requirements.txt

echo "🚀 Running handler..."
python3 handler.py