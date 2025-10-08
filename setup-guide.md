# 📦 Complete Setup Guide

This guide will help you set up the SmolLM3 & SmolVLM2 Gradio Interface from scratch.

## 📁 Project Structure

Your project directory should look like this:

```
smollm-gradio/
├── app.py                    # Main application (already created)
├── install.sh               # Installation script for Mac/Linux
├── install.bat              # Installation script for Windows
├── start.sh                 # Start script for Mac/Linux (created by install.sh)
├── start.bat                # Start script for Windows (created by install.bat)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT License
├── .gitignore              # Git ignore rules
└── venv/                    # Virtual environment (created by install scripts)
```

## 🚀 Quick Setup - Step by Step

### Step 1: Create Project Directory

```bash
# Mac/Linux
mkdir ~/smollm-gradio
cd ~/smollm-gradio

# Windows
mkdir C:\smollm-gradio
cd C:\smollm-gradio
```

### Step 2: Save All Files

Save each file from the artifacts in this conversation:

1. **app.py** - The main application (you already have this)
2. **install.sh** - Copy from "install.sh - Installation Mac/Linux" artifact
3. **install.bat** - Copy from "install.bat - Installation Windows" artifact
4. **requirements.txt** - Copy from "requirements.txt" artifact
5. **README.md** - Copy from "README.md" artifact
6. **.gitignore** - Copy from ".gitignore" artifact
7. **LICENSE** - Copy from "LICENSE" artifact
8. **CONTRIBUTING.md** - Copy from "CONTRIBUTING.md" artifact

### Step 3: Make Scripts Executable (Mac/Linux only)

```bash
chmod +x install.sh
```

### Step 4: Run Installation

**On Mac/Linux:**
```bash
./install.sh
```

**On Windows:**
```cmd
install.bat
```

### Step 5: Start the Application

**On Mac/Linux:**
```bash
./start.sh
```

**On Windows:**
```cmd
start.bat
```

### Step 6: Access the Interface

Open your browser and go to: **http://localhost:7860**

## 🔧 Installation Process Details

### What the install script does:

1. ✅ Checks Python version (requires 3.10+)
2. ✅ Creates a virtual environment
3. ✅ Installs PyTorch (with CUDA/MPS support if available)
4. ✅ Installs all dependencies
5. ✅ Creates a start script
6. ✅ Verifies everything is working

### Installation Time:
- Script execution: 2-5 minutes
- First model download: 15-20 minutes (10 GB)
- **Total first-time setup: ~20-25 minutes**

## 🐙 Setting Up for GitHub

### Step 1: Initialize Git Repository

```bash
cd smollm-gradio
git init
```

### Step 2: Add All Files

```bash
git add .
```

### Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: SmolLM3 & SmolVLM2 Gradio Interface"
```

### Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Name it: `smollm-gradio`
3. Description: "Web interface for SmolLM3 and SmolVLM2 models"
4. **Don't** initialize with README (we already have one)
5. Click "Create repository"

### Step 5: Push to GitHub

```bash
git remote add origin https://github.com/yourusername/smollm-gradio.git
git branch -M main
git push -u origin main
```

## 📝 Customization Options

### Changing the Port

Edit `app.py`, find this line at the bottom:

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,  # ← Change this
    share=False
)
```

### Adding More Models

You can modify the `TEXT_MODEL` and `VISION_MODEL` variables in `app.py`:

```python
TEXT_MODEL = "HuggingFaceTB/SmolLM3-3B"
VISION_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
```

### Creating a Public Link

Change `share=False` to `share=True` in the `demo.launch()` call. Gradio will create a temporary public URL.

## 🎨 Customizing the Interface

### Changing Theme

In `app.py`, modify this line:

```python
with gr.Blocks(title="SmolLM3 & SmolVLM2", theme=gr.themes.Soft()) as demo:
```

Available themes:
- `gr.themes.Soft()` (current)
- `gr.themes.Base()`
- `gr.themes.Glass()`
- `gr.themes.Monochrome()`

### Adding Your Branding

Edit the Markdown header in `app.py`:

```python
gr.Markdown("""
# 🤖 Your Company Name - AI Interface

Your custom description here...
""")
```

## 🔍 Troubleshooting Common Issues

### Issue: "Permission denied" when running install.sh

**Solution:**
```bash
chmod +x install.sh
./install.sh
```

### Issue: Python not found

**Solution:**
- Install Python from https://python.org/downloads/
- Make sure to check "Add to PATH" during installation
- Restart your terminal

### Issue: Git not found

**Solution:**
- Install Git from https://git-scm.com/downloads
- Restart your terminal

### Issue: Models downloading slowly

**Solution:**
- Check internet connection
- Models are ~10 GB total
- Download happens only once
- Subsequent launches are instant

### Issue: Out of memory

**Solution:**
- Close other applications
- Reduce max_length in the UI
- Consider using a machine with more RAM
- Try the quantized versions of models

## 💾 Backing Up Your Setup

### Important directories to backup:

1. **Your code**: The entire project directory
2. **Model cache**: `~/.cache/huggingface/` (Mac/Linux) or `%USERPROFILE%\.cache\huggingface\` (Windows)

### To save disk space:

You can delete the model cache if needed. Models will re-download on next launch:

```bash
# Mac/Linux
rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB--*

# Windows
rmdir /s /q %USERPROFILE%\.cache\huggingface\hub\models--HuggingFaceTB--*
```

## 📊 Performance Tips

### For Faster Loading:
- Use SSD instead of HDD
- Don't use laptop in power-saving mode
- Close background applications

### For Better Generation:
- Use GPU if available
- Increase temperature for creativity
- Decrease temperature for factual responses

### For Lower Memory Usage:
- Reduce max generation length
- Close unused browser tabs
- Use quantized models

## 🆘 Getting Help

If you need help:

1. **Check README.md** - Contains common solutions
2. **Check CONTRIBUTING.md** - For development help
3. **GitHub Issues** - Report bugs
4. **GitHub Discussions** - Ask questions

## 🎉 Next Steps

After setup:

1. ✅ Test both text and vision modes
2. ✅ Try different prompts and images
3. ✅ Explore the settings (temperature, length)
4. ✅ Share with others (using `share=True`)
5. ✅ Contribute improvements (see CONTRIBUTING.md)

## 📚 Additional Resources

- [SmolLM3 Documentation](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [SmolVLM2 Documentation](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [Gradio Documentation](https://gradio.app/docs)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

**You're all set! Enjoy your local AI interface!** 🚀