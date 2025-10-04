#!/bin/bash
# Setup script for Heroku deployment
# Downloads required NLP models and data

echo "Setting up environment for Academic Program Analyzer..."

# Download spaCy Spanish model
echo "Downloading spaCy Spanish model..."
python -m spacy download es_core_news_sm

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p outputs

echo "Setup complete!"
