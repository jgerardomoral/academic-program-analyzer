# Contributing to Análisis de Programas de Estudio

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Environment Setup](#development-environment-setup)
4. [Code Style Guidelines](#code-style-guidelines)
5. [Running Tests](#running-tests)
6. [Submitting Changes](#submitting-changes)
7. [Pull Request Process](#pull-request-process)
8. [Issue Templates](#issue-templates)
9. [Project Structure](#project-structure)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in all interactions.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, trolling, or discriminatory comments
- Personal attacks or derogatory language
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

---

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- Python 3.10, 3.11, or 3.12 installed
- Git installed and configured
- A GitHub account
- Basic knowledge of Python and Streamlit

### Finding Issues to Work On

1. **Check the Issues page** for open issues labeled:
   - `good first issue` - Great for beginners
   - `help wanted` - We need community help
   - `bug` - Something isn't working
   - `enhancement` - New feature or improvement

2. **Comment on the issue** to let others know you're working on it

3. **Wait for approval** from maintainers before starting work on major features

---

## Development Environment Setup

### Step 1: Fork and Clone

1. **Fork the repository** on GitHub (click the "Fork" button)

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/analizador_de_programas.git
   cd analizador_de_programas
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/analizador_de_programas.git
   ```

### Step 2: Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 pytest-cov mypy

# Download spaCy model
python -m spacy download es_core_news_sm
```

### Step 4: Install Pre-commit Hooks (Optional but Recommended)

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

### Step 5: Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or changes

---

## Code Style Guidelines

We follow Python best practices and use automated tools to enforce code style.

### Python Style Guide

1. **Follow PEP 8** with some modifications:
   - Maximum line length: 88 characters (Black default)
   - Use double quotes for strings
   - 4 spaces for indentation (no tabs)

2. **Use Black for formatting**:
   ```bash
   black .
   ```

3. **Use Flake8 for linting**:
   ```bash
   flake8 src/ app/ tests/
   ```

### Code Formatting Tools

#### Black Configuration

Create `pyproject.toml`:

```toml
[tool.black]
line-length = 88
target-version = ['py310', 'py311', 'py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | venv
  | _build
  | buck-out
  | build
  | dist
)/
'''
```

#### Flake8 Configuration

Create `.flake8`:

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, E266, E501, W503
exclude =
    .git,
    __pycache__,
    venv,
    .venv,
    build,
    dist,
    *.egg-info
max-complexity = 10
```

### Docstring Style

Use Google-style docstrings:

```python
def analyze_text(text: str, language: str = "es") -> dict:
    """Analyzes text and extracts key information.

    This function processes the input text using NLP techniques
    to extract frequencies, topics, and other linguistic features.

    Args:
        text: The text to analyze.
        language: Language code (default: "es" for Spanish).

    Returns:
        A dictionary containing analysis results with keys:
            - 'frequencies': Word frequency dictionary
            - 'topics': List of identified topics
            - 'entities': Named entities found

    Raises:
        ValueError: If text is empty or language is not supported.

    Example:
        >>> result = analyze_text("Este es un texto de ejemplo")
        >>> print(result['frequencies'])
    """
    if not text:
        raise ValueError("Text cannot be empty")

    # Implementation here
    return result
```

### Type Hints

Use type hints for function parameters and return values:

```python
from typing import List, Dict, Optional, Union
from pathlib import Path

def process_pdf(
    file_path: Path,
    extract_tables: bool = True
) -> Dict[str, Union[str, List[Dict]]]:
    """Process a PDF file and extract content."""
    pass
```

### Import Organization

Organize imports in this order:

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict

# Third-party imports
import pandas as pd
import streamlit as st
import spacy

# Local imports
from src.utils.config import load_config
from src.processing.text_processor import TextProcessor
```

---

## Running Tests

### Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_pdf_processing.py
├── test_text_analysis.py
├── test_topic_modeling.py
└── integration/
    └── test_full_pipeline.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pdf_processing.py

# Run specific test
pytest tests/test_pdf_processing.py::test_extract_text

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_frequency"
```

### Writing Tests

#### Example Test

```python
import pytest
from src.processing.text_processor import TextProcessor

class TestTextProcessor:
    """Test suite for TextProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a TextProcessor instance for testing."""
        return TextProcessor(language="es")

    def test_tokenize_basic(self, processor):
        """Test basic tokenization."""
        text = "Este es un texto de prueba."
        tokens = processor.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert "texto" in tokens

    def test_tokenize_empty_string(self, processor):
        """Test tokenization with empty string."""
        with pytest.raises(ValueError):
            processor.tokenize("")

    @pytest.mark.parametrize("text,expected_length", [
        ("Hola mundo", 2),
        ("Este es un texto más largo", 6),
        ("", 0),
    ])
    def test_tokenize_parametrized(self, processor, text, expected_length):
        """Test tokenization with multiple inputs."""
        tokens = processor.tokenize(text) if text else []
        assert len(tokens) == expected_length
```

#### Test Fixtures

Create reusable fixtures in `tests/conftest.py`:

```python
import pytest
from pathlib import Path
import tempfile

@pytest.fixture
def sample_pdf():
    """Create a sample PDF for testing."""
    # Create temporary PDF
    pdf_path = Path(tempfile.mktemp(suffix=".pdf"))
    # Generate PDF content
    yield pdf_path
    # Cleanup
    if pdf_path.exists():
        pdf_path.unlink()

@pytest.fixture
def sample_text():
    """Provide sample Spanish text for testing."""
    return """
    Este es un texto de ejemplo para análisis.
    Contiene múltiples oraciones y términos técnicos.
    El objetivo es probar el procesamiento de texto.
    """
```

### Test Coverage

Aim for at least 80% test coverage:

```bash
# Generate coverage report
pytest --cov=src --cov-report=term-missing

# Generate HTML report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

---

## Submitting Changes

### Before Submitting

1. **Run all tests**:
   ```bash
   pytest
   ```

2. **Check code style**:
   ```bash
   black --check .
   flake8 src/ app/ tests/
   ```

3. **Update documentation** if needed

4. **Add tests** for new features

### Commit Message Guidelines

Follow the Conventional Commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:

```
feat(pdf): add support for encrypted PDFs

Implemented decryption functionality for password-protected PDFs
using PyPDF2 encryption methods.

Closes #123
```

```
fix(analysis): correct frequency calculation for multi-word terms

Fixed bug where multi-word terms were not being counted correctly
in frequency analysis.

Fixes #456
```

### Keeping Your Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream changes into your branch
git checkout main
git merge upstream/main

# Rebase your feature branch
git checkout feature/your-feature
git rebase main
```

---

## Pull Request Process

### Step 1: Push Your Changes

```bash
git push origin feature/your-feature-name
```

### Step 2: Create Pull Request

1. Go to your fork on GitHub
2. Click "Pull Request" button
3. Select base repository and branch
4. Fill out the PR template (see below)

### Step 3: PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Related Issues
Closes #(issue number)

## Testing
- [ ] All tests pass locally
- [ ] Added new tests for new features
- [ ] Updated existing tests as needed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Code commented where necessary
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added and passing

## Screenshots (if applicable)
Add screenshots to demonstrate changes

## Additional Notes
Any additional information or context
```

### Step 4: Code Review

1. **Wait for review** from maintainers
2. **Address feedback** by making additional commits
3. **Update your PR** as needed
4. **Resolve conflicts** if they arise

### Step 5: Merge

Once approved:
1. Maintainer will merge your PR
2. Your changes will be included in the next release
3. Branch will be deleted (optionally)

---

## Issue Templates

### Bug Report Template

```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
 - OS: [e.g. Windows 10, macOS 13]
 - Python Version: [e.g. 3.11]
 - Browser [e.g. chrome, safari]
 - Version [e.g. 22]

**Additional context**
Add any other context about the problem.
```

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear and concise description of the problem.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request.

**Implementation ideas (optional)**
If you have ideas about how to implement this feature.
```

### Documentation Improvement Template

```markdown
**Section to improve**
Which documentation section needs improvement?

**Current issue**
What is unclear or missing?

**Suggested improvement**
What would make it better?

**Additional context**
Any other relevant information.
```

---

## Project Structure

Understanding the project structure will help you contribute effectively:

```
analizador_de_programas/
├── app/                          # Streamlit application
│   ├── Home.py                   # Main entry point
│   ├── pages/                    # Streamlit pages
│   │   ├── 1_📁_Subir_PDFs.py
│   │   ├── 2_📊_Analisis.py
│   │   └── ...
│   └── utils/                    # App utilities
│       └── session_manager.py
├── src/                          # Core source code
│   ├── processing/               # Data processing modules
│   │   ├── pdf_processor.py
│   │   └── text_processor.py
│   ├── analysis/                 # Analysis modules
│   │   ├── frequency_analyzer.py
│   │   └── topic_modeler.py
│   ├── visualization/            # Visualization modules
│   └── utils/                    # Utilities
│       └── config.py
├── tests/                        # Test suite
│   ├── conftest.py
│   └── test_*.py
├── data/                         # Data directory (gitignored)
├── outputs/                      # Output files (gitignored)
├── docs/                         # Documentation
├── examples/                     # Example files
├── notebooks/                    # Jupyter notebooks
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── DEPLOYMENT.md                 # Deployment guide
├── CONTRIBUTING.md               # This file
└── README.md                     # Project README
```

### Key Directories

- **app/**: Streamlit UI components
- **src/**: Core business logic (should be framework-agnostic)
- **tests/**: All test files
- **docs/**: Additional documentation

---

## Development Workflow

### Typical Workflow

1. **Sync with upstream**
   ```bash
   git checkout main
   git pull upstream main
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/new-feature
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

4. **Run tests**
   ```bash
   pytest
   black .
   flake8 src/ app/
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/new-feature
   # Create PR on GitHub
   ```

### Tips for Success

1. **Keep PRs focused** - One feature or fix per PR
2. **Write descriptive commits** - Help reviewers understand changes
3. **Add tests** - Ensure your code works and prevent regressions
4. **Update docs** - Keep documentation in sync with code
5. **Be responsive** - Address review feedback promptly
6. **Ask questions** - Don't hesitate to ask for clarification

---

## Getting Help

### Resources

- **Documentation**: Check the `docs/` directory
- **Examples**: See `examples/` for usage examples
- **Issues**: Search existing issues for similar problems

### Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: [maintainer email] for private inquiries

---

## Recognition

Contributors will be recognized in:
- Project README
- Release notes
- Contributors page (if applicable)

Thank you for contributing to make this project better!

---

*Last Updated: 2025*
