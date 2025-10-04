# Deployment Checklist

Use this checklist to ensure all necessary files are in place before deploying your application.

## Pre-Deployment Files Verification

### Required Files (Created)
- [x] `requirements-prod.txt` - Production dependencies with pinned versions
- [x] `Dockerfile` - Multi-stage Docker build configuration
- [x] `docker-compose.yml` - Local Docker development setup
- [x] `.dockerignore` - Files to exclude from Docker builds
- [x] `Procfile` - Heroku/Railway process definition
- [x] `runtime.txt` - Python version specification
- [x] `setup.sh` - NLP models download script for Heroku
- [x] `.slugignore` - Heroku slug size optimization
- [x] `.streamlit/config.toml` - Streamlit server configuration
- [x] `.env.example` - Environment variables template

### Existing Application Files
- [x] `app/Home.py` - Main Streamlit application
- [x] `src/` - Source code directory
- [x] `config.yaml` - Application configuration
- [x] `requirements.txt` - Development dependencies

## Docker Deployment Checklist

### Building the Image
```bash
# 1. Navigate to project directory
cd "c:\Users\jgera\Documents\PARA System\1_Proyectos_activos\analizador_de_programas"

# 2. Build the Docker image
docker build -t academic-analyzer:latest .

# 3. Verify the image was created
docker images | grep academic-analyzer
```

### Running with Docker
```bash
# 1. Run the container
docker run -d -p 8501:8501 --name academic-analyzer academic-analyzer:latest

# 2. Check container status
docker ps

# 3. View logs
docker logs academic-analyzer

# 4. Access the application
# Open browser to: http://localhost:8501
```

### Running with Docker Compose
```bash
# 1. Start services
docker-compose up -d

# 2. Check status
docker-compose ps

# 3. View logs
docker-compose logs -f

# 4. Stop services
docker-compose down
```

## Heroku Deployment Checklist

### Prerequisites
- [ ] Heroku CLI installed
- [ ] Heroku account created
- [ ] Git repository initialized

### Deployment Steps
```bash
# 1. Login to Heroku
heroku login

# 2. Create Heroku app
heroku create your-app-name

# 3. Set Python buildpack
heroku buildpacks:set heroku/python

# 4. Make setup.sh executable (Git Bash/Linux/Mac)
chmod +x setup.sh
git add setup.sh
git commit -m "Make setup.sh executable"

# 5. Deploy to Heroku
git push heroku main

# 6. Scale web dyno
heroku ps:scale web=1

# 7. Open application
heroku open

# 8. Monitor logs
heroku logs --tail
```

### Post-Deployment Verification
- [ ] Application loads successfully
- [ ] File upload functionality works
- [ ] NLP models loaded correctly
- [ ] No memory errors in logs

## Railway Deployment Checklist

### Prerequisites
- [ ] Railway account created
- [ ] GitHub repository connected

### Deployment Steps
1. [ ] Go to Railway dashboard
2. [ ] Click "New Project"
3. [ ] Select "Deploy from GitHub repo"
4. [ ] Choose your repository
5. [ ] Railway auto-detects Dockerfile
6. [ ] Add environment variables if needed
7. [ ] Deploy automatically triggers
8. [ ] Access provided Railway URL

### Configuration
- [ ] Verify Python version in runtime.txt
- [ ] Check build logs for errors
- [ ] Test application functionality
- [ ] Set up custom domain (optional)

## Streamlit Cloud Deployment Checklist

### Prerequisites
- [ ] Streamlit Cloud account
- [ ] GitHub repository public or accessible

### Deployment Steps
1. [ ] Login to Streamlit Cloud
2. [ ] Click "New app"
3. [ ] Select repository
4. [ ] Set main file path: `app/Home.py`
5. [ ] Select Python version: 3.11
6. [ ] Click "Deploy"
7. [ ] Wait for build to complete

### Post-Deployment
- [ ] Test all features
- [ ] Verify file uploads work
- [ ] Check resource usage
- [ ] Set up custom domain (optional)

## Environment Variables Setup

### Docker
```bash
# Add to docker-compose.yml or use -e flags
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
PYTHONUNBUFFERED=1
```

### Heroku
```bash
heroku config:set STREAMLIT_SERVER_HEADLESS=true
heroku config:set PYTHONUNBUFFERED=1
```

### Railway
Add in Variables tab:
- STREAMLIT_SERVER_HEADLESS=true
- PYTHONUNBUFFERED=1

## Testing Checklist

After deployment, verify:
- [ ] Homepage loads correctly
- [ ] PDF upload functionality works
- [ ] Analysis features work
- [ ] Visualizations render properly
- [ ] Session state persists
- [ ] No console errors
- [ ] Acceptable response times
- [ ] Health check endpoint responds

## Troubleshooting Common Issues

### Issue: spaCy model not found
**Solution**: Ensure setup.sh runs or Dockerfile downloads model
```bash
python -m spacy download es_core_news_sm
```

### Issue: Out of memory
**Solution**:
- Upgrade to larger instance
- Optimize code for memory usage
- Process files in smaller chunks

### Issue: Port binding error
**Solution**: Check PORT environment variable is set correctly
```bash
# Heroku sets $PORT automatically
streamlit run app/Home.py --server.port=$PORT
```

### Issue: Build timeout
**Solution**:
- Use multi-stage Docker build (already implemented)
- Cache dependencies
- Use smaller spaCy model (es_core_news_sm)

## Maintenance Checklist

### Regular Tasks
- [ ] Monitor application logs weekly
- [ ] Update dependencies monthly
- [ ] Check security advisories
- [ ] Backup data regularly
- [ ] Test new deployments in staging first
- [ ] Monitor resource usage (CPU, memory, disk)

### Performance Optimization
- [ ] Enable Streamlit caching
- [ ] Optimize large data processing
- [ ] Monitor response times
- [ ] Review and optimize queries
- [ ] Clean up old data/outputs

## Quick Reference Commands

### Docker
```bash
# Build
docker build -t academic-analyzer .

# Run
docker run -p 8501:8501 academic-analyzer

# Stop
docker stop academic-analyzer

# Remove
docker rm academic-analyzer

# Logs
docker logs -f academic-analyzer
```

### Docker Compose
```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Rebuild
docker-compose up --build

# Logs
docker-compose logs -f
```

### Heroku
```bash
# Deploy
git push heroku main

# Logs
heroku logs --tail

# Restart
heroku restart

# Scale
heroku ps:scale web=1

# Config
heroku config:set KEY=value
```

## Security Checklist

Before deploying to production:
- [ ] Remove all hardcoded secrets
- [ ] Use environment variables for sensitive data
- [ ] Enable XSRF protection (already enabled)
- [ ] Set appropriate CORS settings
- [ ] Limit file upload sizes (already set to 50MB)
- [ ] Use HTTPS (provided by platforms)
- [ ] Review .gitignore to exclude secrets
- [ ] Validate all user inputs
- [ ] Keep dependencies updated
- [ ] Enable logging for security events

## Notes

- **Python Version**: Using Python 3.11.10 for maximum compatibility
- **Production Requirements**: Stripped of dev dependencies (pytest, black, flake8, reportlab)
- **Docker Strategy**: Multi-stage build for optimal image size
- **Platforms Supported**: Docker, Heroku, Railway, Streamlit Cloud
- **Health Checks**: Implemented at /_stcore/health endpoint
- **Port Configuration**: 8501 (Streamlit default)
