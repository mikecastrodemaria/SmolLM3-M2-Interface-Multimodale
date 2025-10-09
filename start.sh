#!/bin/bash
# SmolLM3 & SmolVLM2 - Start Script

cd "$(dirname "$0")"
source venv/bin/activate

# Remove python alias if it exists
unalias python 2>/dev/null || true

echo "🚀 Starting SmolLM3 & SmolVLM2 Interface..."
echo "📍 The interface will be available at: http://localhost:7860"
echo ""

python3 smollm3-gradio-app.py
