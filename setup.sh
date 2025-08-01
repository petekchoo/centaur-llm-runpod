#!/bin/bash

set -e  # Exit immediately if a command fails

echo "📦 Installing huggingface_hub..."
pip install huggingface_hub

echo "🔐 Logging into Hugging Face (requires manual token entry)..."
huggingface-cli login

echo "⬇️ Downloading Centaur-70B to /root/centaur_model..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='NousResearch/Centaur-70B',
    local_dir='/root/centaur_model',
    local_dir_use_symlinks=False
)
"

echo "📂 Cloning project repo..."
cd /root
git clone https://github.com/petekchoo/centaur-llm-runpod.git centaur-llm-runpod

echo "📦 Installing project dependencies..."
cd /root/centaur-llm-runpod
pip install -r requirements.txt

echo "🚀 Running handler..."
python3 handler.py