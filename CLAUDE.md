# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Gradio-based web interface for running HuggingFace's SmolLM3 (text generation) and SmolVLM2 (vision analysis) models locally. The application provides a dual-mode interface for both text chat and image analysis, with automatic hardware acceleration detection (CUDA, MPS, or CPU).

## Core Architecture

### Main Application (`smollm3-gradio-app.py`)

**Model Loading Strategy:**
- Models are loaded lazily (on first use of each mode) to save memory
- Device detection happens at startup: CUDA → MPS → CPU
- Uses `AutoModelForCausalLM` for SmolLM3-3B text model
- Uses `AutoModelForImageTextToText` for SmolVLM2-2.2B vision model (NOT `AutoModel` or `AutoModelForCausalLM`)
- Global variables (`text_model`, `vision_model`, etc.) cache loaded models

**Critical Implementation Details:**
1. **SmolVLM2 Loading**: Must use `AutoModelForImageTextToText.from_pretrained()` - this is the correct class for SmolVLM2
2. **dtype Parameter**: Use `dtype=` instead of deprecated `torch_dtype=` throughout
3. **Device Handling**: Different strategies for CUDA (uses `device_map="auto"` with fallback), MPS, and CPU
4. **Response Extraction**: Both `generate_text()` and `analyze_image()` include logic to extract just the assistant's response from the full generation

**Model Configuration:**
- Text Model: `HuggingFaceTB/SmolLM3-3B` (3B parameters, 64K context, multilingual)
- Vision Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (2.2B parameters, SigLIP vision encoder)
- Both models use Apache 2.0 license

### Gradio Interface Structure

**Three-Tab Layout:**
1. **Text Mode** (`💬 Text Mode`):
   - Textbox input with examples
   - Sliders for max_length (50-1024) and temperature (0.1-2.0)
   - Radio button for reasoning mode: `/think` (extended, default) vs `/no_think` (faster)
   - Main response textbox displays the final answer
   - Accordion component below response shows thinking process (when `/think` mode is used)
   - Clear button to reset inputs/outputs (3 outputs: input, answer, thinking)

2. **Chat Mode** (`💬 Chat Mode`):
   - Chatbot component for conversation display
   - Message input with send button
   - Context memory (configurable 1-20 exchanges, default: 10)
   - Higher max length (50-2048 tokens)
   - Sidebar with settings and latest thinking trace
   - Clear conversation button

3. **Vision Mode** (`👁️ Vision Mode`):
   - Image upload component (PIL type)
   - Question textbox with default prompt
   - Max length slider (50-512)
   - Clear button to reset all fields

**Extended Thinking Feature:**
- The `generate_text()` function returns a tuple: `(final_answer, thinking_trace)`
- The `generate_chat_response()` function returns: `(updated_history, "", thinking_trace)`
- When `/think` mode is enabled, functions look for `<think>...</think>` tags in the response
- Thinking content is extracted and displayed separately from the final answer
- In Text Mode: Shown in collapsible Accordion below response
- In Chat Mode: Shown in sidebar "Latest Thinking" textbox
- If no thinking tags are found, only the main answer is shown

**Chat Mode Implementation:**
- `generate_chat_response()` maintains conversation history
- Builds message array with system prompt + conversation history + new message
- Uses sliding window approach: keeps last N exchanges (configurable via `max_history` parameter)
- Returns updated history, empty string (for clearing input), and thinking trace
- History format: List of tuples `[(user_msg, assistant_msg), ...]`

**UI Components Pattern:**
- Components are defined within context managers (`with gr.Row()`, `with gr.Column()`)
- Clear buttons use lambda functions returning tuples: `lambda: ("", "", "")` for Text mode (input, output, thinking), `lambda: (None, "", "")` for Vision mode
- Button click handlers must match the number of outputs returned by the function
- Examples are provided using `gr.Examples()` with predefined inputs

## Common Commands

### Installation & Setup

**Windows:**
```cmd
install-windows.bat    # Sets up venv, installs dependencies, creates start.bat
start.bat             # Activates venv and runs smollm3-gradio-app.py
```

**Mac/Linux:**
```bash
chmod +x install-mac-linux.sh
./install-mac-linux.sh    # Sets up venv, installs dependencies, creates start.sh
./start.sh               # Activates venv and runs smollm3-gradio-app.py
```

**Manual Setup:**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python3 smollm3-gradio-app.py
```

### Running the Application

```bash
# From activated venv
python3 smollm3-gradio-app.py  # or just: python smollm3-gradio-app.py on Windows

# Direct access (if venv is active)
./start.sh  # Mac/Linux
start.bat   # Windows
```

The application launches on `http://127.0.0.1:7860` (localhost only by default).

### Testing Changes

Since this is a single-file application without formal tests, test changes by:
1. Running the application: `python3 smollm3-gradio-app.py`
2. Opening browser to `http://127.0.0.1:7860`
3. Testing both Text Mode and Vision Mode functionality
4. Checking console output for errors or warnings

## Critical Dependencies

**Minimum versions (required):**
- `transformers>=4.53.0` - Required for SmolVLM2 support (older versions cause "Unrecognized configuration class" error)
- `gradio>=4.0.0` - For web interface
- `torch>=2.0.0` - Core ML framework

**Other key dependencies:**
- `pillow>=9.0,<11.0` - Image processing
- `sentencepiece>=0.1.99` - Tokenization
- `protobuf>=3.20,<5.0` - Model serialization
- `einops>=0.7.0` - Tensor operations

**Optional (commented in requirements.txt):**
- `accelerate>=0.20.0` - Enables `device_map="auto"` for multi-GPU
- `bitsandbytes>=0.41.0` - For quantization
- `hf_xet` - Faster model downloads

## Common Issues & Fixes

**"Unrecognized configuration class SmolVLMConfig"**
- Cause: transformers version too old
- Fix: `pip install --upgrade transformers>=4.53.0`

**"SmolVLMModel object has no attribute 'generate'"**
- Cause: Wrong model class being used
- Fix: Ensure code uses `AutoModelForImageTextToText` for SmolVLM2

**"torch_dtype is deprecated"**
- Cause: Old API usage
- Fix: Change `torch_dtype=` to `dtype=` in all `from_pretrained()` calls

## Configuration Options

**Changing Port:**
Edit the `demo.launch()` call at the end of `smollm3-gradio-app.py`:
```python
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,  # Change this
    share=False
)
```

**Enabling Public Sharing:**
Set `share=True` in `demo.launch()` to create a temporary public Gradio URL.

**Changing Models:**
Modify the constants at the top of `smollm3-gradio-app.py`:
```python
TEXT_MODEL = "HuggingFaceTB/SmolLM3-3B"
VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
```

**Memory Optimization:**
- Use quantization (requires `bitsandbytes`)
- Change dtype from `torch.float16` to `torch.float32` (slower but more memory efficient)
- Reduce default `max_length` values in UI components

## Model Storage

Models are cached by HuggingFace Hub in:
- **Mac/Linux**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

First launch downloads ~10GB of models (15-20 minutes). Subsequent launches are instant as models load from cache.

## Platform-Specific Notes

**Windows:**
- Uses `python` command (not `python3`)
- Batch files use `call venv\Scripts\activate.bat`
- Path separators are backslashes

**Mac (Apple Silicon):**
- MPS acceleration enabled automatically via `torch.backends.mps.is_available()`
- Uses `torch.float16` on MPS (good performance with M1/M2/M3)

**Linux:**
- Installation script detects NVIDIA GPU via `nvidia-smi`
- Installs PyTorch with CUDA 12.1 support if GPU detected
- Falls back to CPU-only PyTorch otherwise

## Development Workflow

When making changes:

1. **Edit `smollm3-gradio-app.py`** - This is the only Python source file
2. **Update `requirements.txt`** if adding/changing dependencies
3. **Test locally** by running the application and checking both modes
4. **Update README.md** if changing features, requirements, or usage
5. **Check version compatibility** - transformers ≥4.53.0 is critical

**For installation script changes:**
- Test on the target platform (Windows/Mac/Linux)
- Verify virtual environment creation and activation
- Check that all dependencies install correctly
- Ensure start scripts are created with correct paths

## Git Workflow

This repository uses standard git with `main` as the default branch. Current uncommitted changes:
- Modified: `smollm3-gradio-app.py`
- Untracked: `.claude/` directory

When committing changes, use descriptive messages following the pattern in recent commits:
- "Fix [specific issue]: [brief description]"
- "Add [feature name]"
- "Update [file] with [changes]"
