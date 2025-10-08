#!/bin/bash
# SmolLM3 & SmolVLM2 - Installation Script for Mac/Linux
# This script will set up everything you need to run the Gradio interface

set -e  # Exit on error

echo "🚀 SmolLM3 & SmolVLM2 - Installation Script"
echo "==========================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    echo "📱 Detected: macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    echo "🐧 Detected: Linux"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

# Check Python version
echo ""
echo "🔍 Checking Python installation..."

# Try different Python commands
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3.10+ is required but not found."
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Found Python: $PYTHON_VERSION ($PYTHON_CMD)"

# Check if Python version is >= 3.10
PYTHON_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    echo "❌ Python 3.10 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

# Remove old virtual environment if it exists
if [ -d "venv" ]; then
    echo ""
    echo "🗑️  Removing old virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch
echo ""
echo "🔥 Installing PyTorch..."
if [ "$OS" == "macOS" ]; then
    # Install PyTorch with MPS support for Mac
    pip install torch torchvision torchaudio
else
    # Linux - detect if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 NVIDIA GPU detected, installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "💻 Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install other dependencies
echo ""
echo "📚 Installing dependencies..."
pip install "transformers>=4.53.0"
pip install "gradio>=4.0.0"
pip install "pillow>=9.0,<11.0"
pip install sentencepiece protobuf einops

# Verify installations
echo ""
echo "✅ Verifying installations..."
python3 -c "
import torch
import transformers
import gradio
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ Gradio: {gradio.__version__}')
if torch.backends.mps.is_available():
    print('✅ Apple Silicon MPS: Available')
elif torch.cuda.is_available():
    print(f'✅ CUDA: Available (GPU: {torch.cuda.get_device_name(0)})')
else:
    print('✅ CPU mode: Ready')
"

# Create start script
echo ""
echo "📝 Creating start script..."
cat > start.sh << 'STARTSCRIPT'
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
STARTSCRIPT

chmod +x start.sh

# Installation complete
echo ""
echo "════════════════════════════════════════"
echo "✅ Installation completed successfully!"
echo "════════════════════════════════════════"
echo ""
echo "📋 Next steps:"
echo "   1. Make sure smollm3-gradio-app.py is in the current directory"
echo "   2. Run: ./start.sh"
echo "   3. Open your browser at: http://localhost:7860"
echo ""
echo "💡 Tips:"
echo "   - First launch will download ~10GB of models (15-20 minutes)"
echo "   - Models are cached, subsequent launches are instant"
echo "   - On Mac with Apple Silicon, MPS acceleration is enabled"
echo ""
echo "🆘 Need help? Check the README.md file"
echo ""