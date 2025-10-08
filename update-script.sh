#!/bin/bash
# SmolLM3 & SmolVLM2 - Update Script
# Updates all dependencies to their latest compatible versions

set -e

echo "🔄 SmolLM3 & SmolVLM2 - Update Script"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Remove python alias if it exists
unalias python 2>/dev/null || true

echo ""
echo "📦 Current versions:"
python3 -c "
import torch
import transformers
import gradio
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  Gradio: {gradio.__version__}')
"

echo ""
echo "⬆️  Updating pip..."
python3 -m pip install --upgrade pip setuptools wheel

echo ""
echo "🔄 Updating dependencies..."
pip install --upgrade torch torchvision torchaudio
pip install --upgrade transformers gradio
pip install --upgrade pillow sentencepiece protobuf einops

echo ""
echo "📦 New versions:"
python3 -c "
import torch
import transformers
import gradio
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  Gradio: {gradio.__version__}')
"

# Clear model cache option
echo ""
read -p "🗑️  Clear model cache to force re-download? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Clearing model cache..."
    rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B
    rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-2.2B-Instruct
    echo "✅ Cache cleared"
fi

echo ""
echo "✅ Update completed successfully!"
echo ""
echo "Run ./start.sh to launch the application"