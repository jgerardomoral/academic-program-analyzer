# Deployment Guide

Complete guide for deploying the **Análisis de Programas de Estudio** Streamlit application to various platforms.

---

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Heroku Deployment](#heroku-deployment)
5. [Railway Deployment](#railway-deployment)
6. [Environment Variables](#environment-variables)
7. [Troubleshooting](#troubleshooting)

---

## Local Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12 (Python 3.13 has limited package support)
- pip (Python package manager)
- Git

### Step-by-Step Instructions

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd analizador_de_programas
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download spaCy Language Model**
   ```bash
   python -m spacy download es_core_news_sm
   ```

5. **Run the Application**
   ```bash
   streamlit run app/Home.py
   ```

6. **Access the Application**

   Open your browser and navigate to: `http://localhost:8501`

---

## Docker Deployment

Docker provides a consistent environment for your application across different platforms.

### Prerequisites

- Docker Desktop installed
- Docker Hub account (optional, for pushing images)

### Step 1: Create Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download es_core_news_sm

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Create .dockerignore

Create a `.dockerignore` file:

```
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.git/
.gitignore
.env
*.md
notebooks/
tests/
.pytest_cache/
*.log
outputs/
data/raw/
```

### Step 3: Build Docker Image

```bash
docker build -t analizador-programas:latest .
```

### Step 4: Run Docker Container

```bash
# Basic run
docker run -p 8501:8501 analizador-programas:latest

# Run with volume mounting (for persistent data)
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  analizador-programas:latest

# Run in detached mode
docker run -d -p 8501:8501 --name analizador-app analizador-programas:latest
```

### Step 5: Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

Run with Docker Compose:

```bash
docker-compose up -d
```

---

## Streamlit Cloud Deployment

Streamlit Cloud offers free hosting for Streamlit applications.

### Prerequisites

- GitHub account
- Streamlit Cloud account (sign up at https://share.streamlit.io/)

### Step-by-Step Instructions

#### Step 1: Prepare Your Repository

1. **Ensure all files are committed to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Create a `packages.txt` file** (for system dependencies)

   If your app needs system packages:
   ```
   build-essential
   ```

3. **Verify `requirements.txt`** is up to date

#### Step 2: Deploy to Streamlit Cloud

1. **Log in to Streamlit Cloud**

   Visit: https://share.streamlit.io/

2. **Click "New app"**

3. **Configure Deployment**
   - Repository: Select your GitHub repository
   - Branch: `main` (or your default branch)
   - Main file path: `app/Home.py`
   - App URL: Choose a custom URL (optional)

4. **Advanced Settings (Optional)**
   - Python version: Select `3.11`
   - Add secrets in "Advanced settings" if needed

5. **Click "Deploy"**

   Your app will be built and deployed. This may take 5-10 minutes.

#### Step 3: Configure Secrets (if needed)

1. Go to App settings > Secrets
2. Add your secrets in TOML format:
   ```toml
   [secrets]
   API_KEY = "your-api-key"
   ```

#### Screenshot Placeholders

```
[Screenshot 1: Streamlit Cloud Dashboard - New App Button]
[Screenshot 2: Repository Selection Screen]
[Screenshot 3: Deployment Configuration Screen]
[Screenshot 4: Advanced Settings with Secrets]
[Screenshot 5: Successful Deployment Screen]
```

---

## Heroku Deployment

Heroku is a cloud platform that supports Python applications.

### Prerequisites

- Heroku account (sign up at https://heroku.com/)
- Heroku CLI installed

### Step 1: Install Heroku CLI

```bash
# Windows (using installer from heroku.com/cli)
# macOS
brew tap heroku/brew && brew install heroku

# Ubuntu/Debian
curl https://cli-assets.heroku.com/install.sh | sh
```

### Step 2: Create Heroku Configuration Files

#### Create `setup.sh`

```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

#### Create `Procfile`

```
web: sh setup.sh && streamlit run app/Home.py
```

### Step 3: Login to Heroku

```bash
heroku login
```

### Step 4: Create Heroku App

```bash
heroku create analizador-programas

# Or with a specific region
heroku create analizador-programas --region eu
```

### Step 5: Configure Buildpacks

```bash
heroku buildpacks:set heroku/python
```

### Step 6: Add spaCy Model Download

Add to your `requirements.txt` or create a `runtime.txt`:

```
python-3.11.7
```

Create a `bin/post_compile` script:

```bash
#!/bin/bash
python -m spacy download es_core_news_sm
```

Make it executable:
```bash
chmod +x bin/post_compile
```

### Step 7: Deploy

```bash
git add .
git commit -m "Configure for Heroku deployment"
git push heroku main
```

### Step 8: Scale Dynos

```bash
heroku ps:scale web=1
```

### Step 9: Open Application

```bash
heroku open
```

### Step 10: View Logs (if issues occur)

```bash
heroku logs --tail
```

---

## Railway Deployment

Railway offers a modern deployment platform with simple configuration.

### Prerequisites

- Railway account (sign up at https://railway.app/)
- GitHub repository connected

### Step-by-Step Instructions

#### Step 1: Sign Up and Connect GitHub

1. Visit https://railway.app/
2. Sign up using your GitHub account
3. Authorize Railway to access your repositories

#### Step 2: Create New Project

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your repository

#### Step 3: Configure Deployment

Railway will automatically detect your Python app.

1. **Add Environment Variables** (if needed)
   - Go to Variables tab
   - Add any required environment variables

2. **Configure Start Command**

   In Settings > Deploy:
   ```bash
   streamlit run app/Home.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Set Python Version**

   Create `runtime.txt`:
   ```
   python-3.11.7
   ```

#### Step 4: Add Build Command

In Settings > Build:

```bash
pip install -r requirements.txt && python -m spacy download es_core_news_sm
```

#### Step 5: Deploy

1. Railway will automatically deploy on every push to main
2. Your app will be available at a generated Railway URL
3. You can add a custom domain in Settings

#### Step 6: Monitor Deployment

1. View logs in the Deployments tab
2. Check metrics in the Observability tab

---

## Environment Variables

### Required Environment Variables

The application uses the following environment variables:

```bash
# Optional: Custom configuration path
CONFIG_PATH=./config.yaml

# Optional: Data directories
DATA_DIR=./data
OUTPUT_DIR=./outputs

# Optional: Logging level
LOG_LEVEL=INFO

# Optional: Streamlit specific
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
```

### Setting Environment Variables

#### Local Development (.env file)

Create a `.env` file in the project root:

```bash
CONFIG_PATH=./config.yaml
LOG_LEVEL=DEBUG
```

#### Docker

```bash
docker run -p 8501:8501 \
  -e CONFIG_PATH=./config.yaml \
  -e LOG_LEVEL=INFO \
  analizador-programas:latest
```

#### Streamlit Cloud

Add in App Settings > Secrets (TOML format)

#### Heroku

```bash
heroku config:set LOG_LEVEL=INFO
heroku config:set CONFIG_PATH=./config.yaml
```

#### Railway

Add in the Variables tab through the Railway dashboard

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: spaCy Model Not Found

**Error**: `OSError: [E050] Can't find model 'es_core_news_sm'`

**Solution**:
```bash
python -m spacy download es_core_news_sm
```

For Docker, ensure this line is in your Dockerfile:
```dockerfile
RUN python -m spacy download es_core_news_sm
```

#### Issue 2: Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Kill process on port 8501 (Windows)
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Kill process on port 8501 (macOS/Linux)
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run app/Home.py --server.port=8502
```

#### Issue 3: Memory Issues on Cloud Platforms

**Error**: Application crashes due to memory limits

**Solution**:
- Optimize your code to use less memory
- Use pagination for large datasets
- Clear Streamlit cache regularly
- Upgrade to a paid tier with more memory

For Streamlit:
```python
@st.cache_data(max_entries=10)
def cached_function():
    pass
```

#### Issue 4: Python Version Compatibility

**Error**: Package installation failures

**Solution**:
- Use Python 3.10, 3.11, or 3.12
- Check `PYTHON_313_NOTES.md` for Python 3.13 specific issues
- Specify Python version in `runtime.txt`

#### Issue 5: File Upload Issues

**Error**: File upload fails or files disappear

**Solution**:
- Check file size limits (Streamlit Cloud: 200MB default)
- Use session state to persist uploads
- Implement proper file handling:

```python
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    # Process immediately or save to disk
```

#### Issue 6: Streamlit Cloud Build Failures

**Error**: Build fails during deployment

**Solution**:
1. Check `requirements.txt` for incompatible packages
2. Ensure Python version compatibility
3. Review build logs in Streamlit Cloud dashboard
4. Add `packages.txt` for system dependencies
5. Use `--no-cache-dir` in pip install

#### Issue 7: Heroku Slug Size Too Large

**Error**: `Compiled slug size: XGB is too large (max is 500MB)`

**Solution**:
1. Use `.slugignore` to exclude unnecessary files:
   ```
   *.md
   tests/
   notebooks/
   .git/
   ```
2. Remove unused dependencies
3. Use smaller machine learning models

#### Issue 8: Railway Build Timeout

**Error**: Build exceeds time limit

**Solution**:
1. Optimize build process
2. Use Docker for faster builds
3. Cache dependencies
4. Split large installation steps

#### Issue 9: CORS Issues

**Error**: Cross-Origin Resource Sharing errors

**Solution**:

Create `.streamlit/config.toml`:
```toml
[server]
enableCORS = false
enableXsrfProtection = false
```

#### Issue 10: Secrets Not Loading

**Error**: Environment variables or secrets not accessible

**Solution**:

For Streamlit Cloud, use:
```python
import streamlit as st

# Access secrets
api_key = st.secrets["API_KEY"]
```

For other platforms, use python-dotenv:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")
```

### Getting Help

If you encounter issues not covered here:

1. **Check Logs**
   - Streamlit Cloud: View in dashboard
   - Heroku: `heroku logs --tail`
   - Railway: View in Deployments tab
   - Docker: `docker logs <container-id>`

2. **Community Resources**
   - Streamlit Forum: https://discuss.streamlit.io/
   - Stack Overflow: Tag with `streamlit`
   - GitHub Issues: Check repository issues

3. **Contact Support**
   - Open an issue in the repository
   - Contact platform support (Heroku, Railway, etc.)

---

## Best Practices

### Performance Optimization

1. **Use Caching**
   ```python
   @st.cache_data
   def expensive_computation():
       pass
   ```

2. **Lazy Loading**
   - Load large models only when needed
   - Use session state to avoid recomputation

3. **Optimize Dependencies**
   - Only include necessary packages in requirements.txt
   - Use lightweight alternatives when possible

### Security

1. **Never commit secrets**
   - Use `.gitignore` for `.env` files
   - Use platform-specific secret management

2. **Input Validation**
   - Validate file uploads
   - Sanitize user inputs

3. **HTTPS**
   - Use HTTPS in production
   - Most platforms provide this by default

### Monitoring

1. **Set up logging**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **Track errors**
   - Use error tracking services (Sentry, etc.)
   - Monitor application metrics

3. **Regular updates**
   - Keep dependencies updated
   - Monitor security advisories

---

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Railway Documentation](https://docs.railway.app/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)

---

*Last Updated: 2025*
