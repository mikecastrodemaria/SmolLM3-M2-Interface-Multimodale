# Contributing to SmolLM Gradio Interface

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/smollm-gradio.git
   cd smollm-gradio
   ```
3. **Set up the development environment**:
   ```bash
   ./install.sh  # Mac/Linux
   # or
   install.bat  # Windows
   ```

## 🔨 Making Changes

1. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Test your changes** thoroughly:
   - Test on your local machine
   - Try both text and vision modes
   - Check error handling

4. **Commit your changes** with a clear message:
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

## 📝 Code Style Guidelines

### Python Code
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use meaningful variable and function names
- Add docstrings to functions
- Keep functions focused and small
- Add comments for complex logic

### Example:
```python
def generate_text(prompt, max_length=512, temperature=0.7):
    """
    Generate text using SmolLM3 model.
    
    Args:
        prompt (str): The input text prompt
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature (0.1-2.0)
    
    Returns:
        str: Generated text response
    """
    # Your code here
    pass
```

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Exact steps to trigger the bug
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**:
   - OS (Windows 11, macOS 14, Ubuntu 22.04, etc.)
   - Python version
   - GPU (if applicable)
6. **Error messages**: Full error message and traceback
7. **Screenshots**: If relevant

### Bug Report Template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. macOS 14.2]
 - Python Version: [e.g. 3.10.6]
 - GPU: [e.g. Apple M1, NVIDIA RTX 3060]

**Additional context**
Any other context about the problem.
```

## ✨ Feature Requests

We welcome feature suggestions! Please:

1. **Check existing issues** to avoid duplicates
2. **Describe the feature** clearly
3. **Explain the use case**: Why is this useful?
4. **Provide examples**: How would it work?

## 🎯 Areas for Contribution

Here are some areas where contributions are especially welcome:

### High Priority
- [ ] Add support for more models (Llama, Mistral, etc.)
- [ ] Improve error handling and user feedback
- [ ] Add conversation history/memory
- [ ] Implement model switching in UI
- [ ] Add batch processing capabilities

### Medium Priority
- [ ] Add more example prompts
- [ ] Improve documentation
- [ ] Add unit tests
- [ ] Create video tutorials
- [ ] Support for more languages in UI

### Nice to Have
- [ ] Dark mode theme
- [ ] Export conversations
- [ ] Custom model loading
- [ ] Docker support
- [ ] API endpoint

## 🧪 Testing

Before submitting a PR:

1. **Test basic functionality**:
   - Text generation works
   - Vision analysis works
   - Error messages are helpful

2. **Test edge cases**:
   - Empty inputs
   - Very long inputs
   - Invalid images
   - Network issues

3. **Test on multiple platforms** (if possible):
   - Windows, macOS, Linux
   - CPU and GPU modes

## 📤 Submitting Pull Requests

1. **Update documentation** if needed
2. **Add yourself** to CONTRIBUTORS.md (if it exists)
3. **Push your changes** to your fork
4. **Create a Pull Request** with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to related issues (if any)
   - Screenshots/demos (if applicable)

### PR Template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tested on Windows/macOS/Linux
- [ ] Tested with CPU
- [ ] Tested with GPU
- [ ] Tested both modes (text and vision)

## Screenshots (if applicable)
Add screenshots here

## Related Issues
Closes #123
```

## 💬 Communication

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## ❓ Questions?

If you have questions about contributing, feel free to:
- Open a GitHub Discussion
- Comment on an existing issue
- Reach out to maintainers

Thank you for contributing! 🎉