# GitHub Setup Guide

Complete step-by-step guide to set up your GitHub repository and deploy the **Análisis de Programas de Estudio** Streamlit application.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Git Configuration](#initial-git-configuration)
3. [Initialize Local Repository](#initialize-local-repository)
4. [Create GitHub Repository](#create-github-repository)
5. [Connect Local to GitHub](#connect-local-to-github)
6. [Push Code to GitHub](#push-code-to-github)
7. [Set Up GitHub Actions (Optional)](#set-up-github-actions-optional)
8. [Enable Streamlit Cloud Deployment](#enable-streamlit-cloud-deployment)
9. [Repository Best Practices](#repository-best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Git** - Version control system
   - Download from: https://git-scm.com/downloads
   - Verify installation:
     ```bash
     git --version
     ```

2. **GitHub Account**
   - Sign up at: https://github.com/
   - Free account is sufficient

3. **Text Editor or IDE**
   - VS Code (recommended): https://code.visualstudio.com/
   - Or any editor of your choice

### Recommended Tools

- **GitHub Desktop** (optional): https://desktop.github.com/
- **GitHub CLI** (optional): https://cli.github.com/

---

## Initial Git Configuration

Before using Git, configure your identity. These settings will be used for all your commits.

### Step 1: Set Your Name

```bash
git config --global user.name "Your Name"
```

### Step 2: Set Your Email

```bash
git config --global user.email "your.email@example.com"
```

**Important**: Use the same email associated with your GitHub account.

### Step 3: Set Default Branch Name

```bash
git config --global init.defaultBranch main
```

### Step 4: Verify Configuration

```bash
git config --list
```

You should see:
```
user.name=Your Name
user.email=your.email@example.com
init.defaultbranch=main
```

### Optional: Configure Line Endings

**Windows**:
```bash
git config --global core.autocrlf true
```

**macOS/Linux**:
```bash
git config --global core.autocrlf input
```

---

## Initialize Local Repository

### Step 1: Navigate to Project Directory

```bash
cd "C:\Users\jgera\Documents\PARA System\1_Proyectos_activos\analizador_de_programas"
```

Or use your actual project path.

### Step 2: Initialize Git Repository

```bash
git init
```

You should see:
```
Initialized empty Git repository in .../analizador_de_programas/.git/
```

### Step 3: Create .gitignore File

Create a `.gitignore` file to exclude unnecessary files from version control:

```bash
# Windows
type nul > .gitignore

# macOS/Linux
touch .gitignore
```

Add the following content to `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
.venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Streamlit
.streamlit/secrets.toml

# Data files (keep structure, ignore content)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
outputs/*
!outputs/.gitkeep

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.*.local

# Testing
.pytest_cache/
.coverage
htmlcov/
*.cover

# Logs
*.log
logs/

# OS
Thumbs.db
.DS_Store

# Temporary files
*.tmp
*.temp
~*
```

### Step 4: Create .gitkeep Files for Empty Directories

Keep directory structure but ignore contents:

```bash
# Windows
type nul > data\raw\.gitkeep
type nul > data\processed\.gitkeep
type nul > outputs\.gitkeep

# macOS/Linux
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch outputs/.gitkeep
```

### Step 5: Add Files to Staging Area

```bash
git add .
```

This stages all files (except those in `.gitignore`) for commit.

### Step 6: Create Initial Commit

```bash
git commit -m "Initial commit: Project setup with Streamlit app structure"
```

You should see:
```
[main (root-commit) xxxxxxx] Initial commit: Project setup with Streamlit app structure
 XX files changed, XXX insertions(+)
```

---

## Create GitHub Repository

### Option A: Using GitHub Website

#### Step 1: Log In to GitHub

Navigate to https://github.com/ and log in.

#### Step 2: Create New Repository

1. Click the **+** icon in the top-right corner
2. Select **New repository**

#### Step 3: Configure Repository

Fill in the following:

- **Repository name**: `analizador-de-programas` (or your preferred name)
- **Description**: "Streamlit application for analyzing university curriculum programs using NLP techniques"
- **Visibility**: Choose **Public** or **Private**
- **Initialize repository**:
  - **DO NOT** check "Add a README file"
  - **DO NOT** add .gitignore
  - **DO NOT** choose a license

  (We already have these files locally)

#### Step 4: Create Repository

Click **Create repository**.

#### Step 5: Copy Repository URL

You'll see a setup page. Copy the repository URL:
- **HTTPS**: `https://github.com/username/analizador-de-programas.git`
- **SSH**: `git@github.com:username/analizador-de-programas.git`

### Option B: Using GitHub CLI

If you have GitHub CLI installed:

```bash
# Login to GitHub
gh auth login

# Create repository
gh repo create analizador-de-programas \
  --description "Streamlit application for analyzing university curriculum programs" \
  --public

# Or for private repository
gh repo create analizador-de-programas \
  --description "Streamlit application for analyzing university curriculum programs" \
  --private
```

---

## Connect Local to GitHub

### Step 1: Add Remote Repository

Using HTTPS (recommended for beginners):

```bash
git remote add origin https://github.com/YOUR_USERNAME/analizador-de-programas.git
```

Using SSH (if you have SSH keys set up):

```bash
git remote add origin git@github.com:YOUR_USERNAME/analizador-de-programas.git
```

**Replace `YOUR_USERNAME`** with your actual GitHub username.

### Step 2: Verify Remote

```bash
git remote -v
```

You should see:
```
origin  https://github.com/YOUR_USERNAME/analizador-de-programas.git (fetch)
origin  https://github.com/YOUR_USERNAME/analizador-de-programas.git (push)
```

---

## Push Code to GitHub

### Step 1: Set Upstream Branch

```bash
git branch -M main
```

This renames your default branch to `main` (if it isn't already).

### Step 2: Push to GitHub

```bash
git push -u origin main
```

**First-time authentication**:
- You'll be prompted for credentials
- Use your GitHub username and **Personal Access Token** (not password)

### Step 3: Create Personal Access Token (if needed)

If you don't have a token:

1. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Click **Generate new token** > **Generate new token (classic)**
3. Give it a name: "Git CLI Access"
4. Select scopes:
   - ✅ `repo` (all)
   - ✅ `workflow` (if using GitHub Actions)
5. Click **Generate token**
6. **Copy the token** (you won't see it again!)
7. Use this token as your password when pushing

### Step 4: Verify Push

Go to your GitHub repository in a browser. You should see all your files!

---

## Set Up GitHub Actions (Optional)

Automate testing and deployment using GitHub Actions.

### Step 1: Create Workflows Directory

```bash
# Windows
mkdir .github\workflows

# macOS/Linux
mkdir -p .github/workflows
```

### Step 2: Create CI Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8

    - name: Download spaCy model
      run: |
        python -m spacy download es_core_news_sm

    - name: Lint with flake8
      run: |
        flake8 src/ app/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ app/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

    - name: Check formatting with black
      run: |
        black --check src/ app/

    - name: Run tests with pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
```

### Step 3: Create Deployment Workflow (Optional)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Trigger Streamlit Cloud Deployment
      run: |
        echo "Deployment will be automatic via Streamlit Cloud"
        echo "No manual trigger needed"
```

### Step 4: Commit and Push Workflows

```bash
git add .github/workflows/
git commit -m "ci: add GitHub Actions workflows for testing and deployment"
git push origin main
```

### Step 5: View Actions

1. Go to your GitHub repository
2. Click the **Actions** tab
3. You should see your workflows running!

---

## Enable Streamlit Cloud Deployment

Deploy your app for free on Streamlit Cloud.

### Step 1: Sign Up for Streamlit Cloud

1. Visit: https://share.streamlit.io/
2. Click **Sign up** or **Sign in**
3. Sign in with your GitHub account
4. Authorize Streamlit Cloud to access your repositories

### Step 2: Deploy New App

1. Click **New app** button
2. Select your repository: `YOUR_USERNAME/analizador-de-programas`
3. Select branch: `main`
4. Main file path: `app/Home.py`
5. (Optional) Customize app URL

### Step 3: Configure Advanced Settings

Click **Advanced settings**:

1. **Python version**: Select `3.11` or `3.12`
2. **Secrets**: Add if needed (see below)

#### Adding Secrets

If you need environment variables, add them in TOML format:

```toml
# .streamlit/secrets.toml format
[general]
API_KEY = "your-api-key-here"

[database]
HOST = "localhost"
PORT = 5432
```

### Step 4: Deploy

Click **Deploy!**

Your app will build and deploy. This takes 5-10 minutes.

### Step 5: Access Your App

Once deployed, you'll get a URL like:
```
https://your-username-analizador-de-programas-app-home-abc123.streamlit.app/
```

### Step 6: Configure Custom Domain (Optional)

1. Go to App settings
2. Click **Custom domains**
3. Follow instructions to set up your domain

---

## Repository Best Practices

### Branch Strategy

#### Main Branch

- **main**: Production-ready code
- Protected: Require pull requests
- All tests must pass before merging

#### Development Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Work on feature
git add .
git commit -m "feat: add new feature"

# Push to GitHub
git push origin feature/new-feature

# Create Pull Request on GitHub
# After review and approval, merge to main
```

#### Branch Naming Conventions

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code improvements
- `test/` - Test additions

### Commit Message Guidelines

Follow Conventional Commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Examples**:

```bash
git commit -m "feat(pdf): add support for encrypted PDFs"
git commit -m "fix(analysis): correct frequency calculation"
git commit -m "docs: update README with deployment instructions"
git commit -m "refactor(ui): improve layout of analysis page"
```

### Protect Main Branch

1. Go to repository **Settings**
2. Click **Branches**
3. Add rule for `main`
4. Enable:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass
   - ✅ Require conversation resolution before merging

### Add Repository Topics

1. Go to repository main page
2. Click the gear icon next to "About"
3. Add topics:
   - `streamlit`
   - `nlp`
   - `python`
   - `text-analysis`
   - `education`
   - `curriculum-analysis`

### Create Repository Description

Add to "About" section:
```
Streamlit application for analyzing university curriculum programs using NLP techniques including frequency analysis, topic modeling, and comparative analysis.
```

### Add Repository Features

Enable in Settings:
- ✅ Issues
- ✅ Projects (if using project boards)
- ✅ Wiki (optional)
- ✅ Discussions (for community)

---

## Essential Commands Reference

### Daily Workflow

```bash
# Check status
git status

# Add changes
git add .
# or
git add specific-file.py

# Commit changes
git commit -m "type: description"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main

# View commit history
git log --oneline

# View differences
git diff
```

### Branch Management

```bash
# List branches
git branch

# Create new branch
git branch feature/new-feature

# Switch to branch
git checkout feature/new-feature

# Create and switch in one command
git checkout -b feature/new-feature

# Delete local branch
git branch -d feature/new-feature

# Delete remote branch
git push origin --delete feature/new-feature
```

### Syncing with Remote

```bash
# Fetch changes (doesn't merge)
git fetch origin

# Pull and merge changes
git pull origin main

# View remote info
git remote show origin

# Update remote URL
git remote set-url origin <new-url>
```

### Undoing Changes

```bash
# Discard local changes
git checkout -- file.py

# Unstage files
git reset HEAD file.py

# Amend last commit
git commit --amend -m "new message"

# Revert a commit
git revert <commit-hash>

# Reset to specific commit (careful!)
git reset --hard <commit-hash>
```

---

## Troubleshooting

### Issue 1: Authentication Failed

**Error**: `fatal: Authentication failed`

**Solution**:

1. Use Personal Access Token instead of password
2. Or set up SSH keys:

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings > SSH and GPG keys > New SSH key
```

### Issue 2: Repository Already Exists

**Error**: `fatal: remote origin already exists`

**Solution**:

```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin <repository-url>
```

### Issue 3: Merge Conflicts

**Error**: Merge conflict when pulling

**Solution**:

```bash
# Pull with rebase
git pull --rebase origin main

# Or manually resolve conflicts
# 1. Open conflicting files
# 2. Look for <<<<<<< and >>>>>>>
# 3. Edit to resolve
# 4. Remove conflict markers
# 5. Add and commit

git add .
git commit -m "fix: resolve merge conflicts"
git push origin main
```

### Issue 4: Large Files

**Error**: File too large to push

**Solution**:

1. Add to `.gitignore`:
```bash
echo "large-file.pdf" >> .gitignore
```

2. Remove from Git but keep locally:
```bash
git rm --cached large-file.pdf
git commit -m "chore: remove large file from git"
```

3. Or use Git LFS (Large File Storage):
```bash
git lfs install
git lfs track "*.pdf"
git add .gitattributes
git commit -m "chore: track PDFs with Git LFS"
```

### Issue 5: Wrong Branch

**Error**: Committed to wrong branch

**Solution**:

```bash
# Move commit to correct branch
git log  # Note commit hash
git checkout correct-branch
git cherry-pick <commit-hash>

# Remove from wrong branch
git checkout wrong-branch
git reset --hard HEAD~1
```

### Issue 6: Streamlit Cloud Build Fails

**Error**: Build fails on Streamlit Cloud

**Solution**:

1. Check `requirements.txt` compatibility
2. Ensure Python version matches local
3. Check logs in Streamlit Cloud dashboard
4. Verify all imports work
5. Test locally with same Python version

### Issue 7: Push Rejected

**Error**: `! [rejected] main -> main (fetch first)`

**Solution**:

```bash
# Pull changes first
git pull origin main

# Resolve any conflicts
# Then push
git push origin main
```

---

## Next Steps

After setup:

1. ✅ **Read CONTRIBUTING.md** for contribution guidelines
2. ✅ **Review DEPLOYMENT.md** for deployment options
3. ✅ **Set up development environment** with virtual environment
4. ✅ **Enable GitHub Actions** for automated testing
5. ✅ **Deploy to Streamlit Cloud** for public access
6. ✅ **Invite collaborators** if working in a team
7. ✅ **Create project board** for task management

---

## Additional Resources

### Git Resources
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Pro Git Book](https://git-scm.com/book/en/v2)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

### GitHub Resources
- [GitHub Skills](https://skills.github.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub CLI Manual](https://cli.github.com/manual/)

### Streamlit Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Forum](https://discuss.streamlit.io/)

### Best Practices
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)

---

## Quick Start Checklist

Use this checklist to ensure you've completed all setup steps:

- [ ] Git installed and configured
- [ ] GitHub account created
- [ ] Local repository initialized
- [ ] `.gitignore` file created
- [ ] Initial commit made
- [ ] GitHub repository created
- [ ] Local repository connected to GitHub
- [ ] Code pushed to GitHub
- [ ] Repository description and topics added
- [ ] Branch protection rules set (optional)
- [ ] GitHub Actions workflows added (optional)
- [ ] Streamlit Cloud deployment configured (optional)
- [ ] Collaborators invited (if applicable)
- [ ] README.md reviewed and updated
- [ ] LICENSE file verified

---

**Congratulations!** Your GitHub repository is now set up and ready for collaboration and deployment.

For questions or issues, please refer to the [Troubleshooting](#troubleshooting) section or open an issue in the repository.

---

*Last Updated: 2025*
