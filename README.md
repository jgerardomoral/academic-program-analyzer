# Academic Program Analyzer

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Streamlit-based application for analyzing academic study programs (programas de estudio). Extract insights from PDF curricula using NLP techniques including frequency analysis, topic modeling, and automated skills mapping.

![Academic Program Analyzer](docs/screenshots/demo.png)

[Live Demo](#) | [Documentation](docs/) | [Report Issue](../../issues)

---

## Features

- **PDF Text Extraction**: Robust extraction from academic PDFs using pdfplumber with table support
- **Frequency Analysis**: TF-IDF scoring and n-gram analysis to identify key terms and concepts
- **Topic Modeling**: Automatic topic discovery using LDA and NMF algorithms
- **Skills Mapping**: Map curriculum content to educational skills taxonomy
- **Program Comparison**: Side-by-side comparison of multiple study programs
- **Interactive Visualizations**: Beautiful charts powered by Plotly, word clouds, and heatmaps
- **Export Capabilities**: Generate reports in Excel, CSV, and JSON formats
- **Multi-page Interface**: Intuitive workflow from upload to analysis to comparison

---

## Tech Stack

### Core Technologies
- **Python 3.11+** - Main programming language
- **Streamlit 1.30+** - Web application framework
- **spaCy (es_core_news_lg)** - Spanish NLP processing and lemmatization
- **scikit-learn** - TF-IDF vectorization, LDA/NMF topic modeling
- **Plotly** - Interactive data visualizations

### Additional Libraries
- **PDF Processing**: pdfplumber, PyPDF2, tabula-py
- **Data Processing**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, wordcloud
- **UI Components**: streamlit-aggrid, streamlit-option-menu
- **Utilities**: PyYAML, python-dotenv, openpyxl

### NLP Capabilities
- Spanish language processing (es_core_news_lg model)
- Custom academic stopwords filtering
- POS tagging (NOUN, VERB, ADJ)
- Lemmatization and tokenization
- TF-IDF and n-gram extraction
- Topic modeling (LDA/NMF)

---

## Installation

### Prerequisites
- Python 3.11 or higher (3.13 compatible with minor limitations - see [PYTHON_313_NOTES.md](PYTHON_313_NOTES.md))
- pip package manager
- 500MB+ free disk space (for spaCy model)

### Step-by-step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/analizador-de-programas.git
   cd analizador-de-programas
   ```

2. **Create virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Spanish spaCy model**
   ```bash
   python -m spacy download es_core_news_lg
   ```

5. **Run the application**
   ```bash
   streamlit run app/Home.py
   ```

6. **Expected output**
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.1.x:8501
   ```

---

## Usage

### Workflow

The application follows a four-step workflow:

#### 1. Upload PDFs (📁 Subir PDFs)
- Upload one or more academic program PDFs
- Automatic text extraction and preprocessing
- Documents are stored in `data/raw/`

#### 2. Frequency Analysis (📊 Análisis de Frecuencias)
- View top terms by TF-IDF score
- Explore n-grams (1-3 words)
- Interactive frequency charts and word clouds
- Export results to Excel/CSV

#### 3. Topics & Skills (🎯 Topics y Habilidades)
- Automatic topic discovery (LDA/NMF)
- Configure number of topics (3-20)
- Map content to skills taxonomy
- Visualize skill profiles per document

#### 4. Comparison (📈 Comparativa)
- Compare multiple programs side-by-side
- Heatmaps showing term overlap
- Skill distribution radar charts
- Export comparative reports

### Quick Start Example

```python
# After launching the app:
# 1. Navigate to "📁 Subir PDFs"
# 2. Upload your curriculum PDF files
# 3. Click "Procesar PDFs"
# 4. Explore the analysis pages
```

---

## Project Structure

```
analizador-de-programas/
│
├── app/                          # Streamlit frontend
│   ├── Home.py                  # Main entry point
│   ├── pages/                   # Multi-page app sections
│   │   ├── 1_📁_Subir_PDFs.py
│   │   ├── 2_📊_Analisis_Frecuencias.py
│   │   ├── 3_🎯_Topics_Habilidades.py
│   │   └── 4_📈_Comparativa.py
│   ├── components/              # Reusable UI components
│   └── utils/                   # Session state, caching, exports
│       ├── session_manager.py
│       ├── cache_manager.py
│       └── export.py
│
├── src/                         # Backend processing modules
│   ├── extraction/             # PDF text extraction and preprocessing
│   │   ├── pdf_extractor.py
│   │   └── preprocessor.py
│   ├── analysis/               # NLP analysis
│   │   ├── frequency.py        # TF-IDF and n-grams
│   │   ├── topics.py           # LDA/NMF topic modeling
│   │   └── skills_mapper.py    # Skills taxonomy mapping
│   ├── visualization/          # Charts and reports
│   │   ├── charts.py
│   │   ├── wordcloud.py
│   │   └── reports.py
│   └── utils/                  # Config, schemas, utilities
│       ├── config.py
│       ├── schemas.py
│       └── file_io.py
│
├── data/                        # Data storage
│   ├── raw/                    # Original PDFs
│   ├── processed/              # Extracted texts (JSON/pickle)
│   ├── cache/                  # Analysis cache
│   └── taxonomia/              # Skills taxonomy and custom stopwords
│
├── tests/                       # Pytest test suite
├── notebooks/                   # Jupyter notebooks for exploration
├── outputs/                     # Generated reports and exports
│
├── config.yaml                  # Application configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Key Directories

- **`app/`**: Streamlit UI with multi-page navigation
- **`src/`**: Modular backend processing (extraction, analysis, visualization)
- **`data/`**: Organized data storage with clear separation of raw/processed
- **`tests/`**: Comprehensive test suite with fixtures

---

## Configuration

The application is configured via `config.yaml`:

### Main Configuration Sections

```yaml
# NLP Configuration
nlp:
  spacy_model: "es_core_news_lg"
  min_word_length: 3
  pos_tags_keep: ["NOUN", "VERB", "ADJ"]
  lemmatize: true

# Frequency Analysis
frequency:
  top_n_terms: 50
  ngram_range: [1, 3]
  tfidf_max_features: 500
  min_df: 2
  max_df: 0.8

# Topic Modeling
topics:
  default_n_topics: 10
  min_topics: 3
  max_topics: 20
  lda_iterations: 100

# Skills Mapping
skills:
  min_confidence: 0.3
  weight_tfidf: 0.6
  weight_frequency: 0.4
```

### Customization

1. **Custom Stopwords**: Add domain-specific terms to `data/taxonomia/stopwords_custom.txt`
2. **Skills Taxonomy**: Edit `data/taxonomia/habilidades.json` to define your skills structure
3. **Visualization**: Modify color schemes and chart settings in `config.yaml` under `visualization`
4. **UI Settings**: Adjust upload limits, cache TTL, and theme in `config.yaml`

---

## Deployment

### Local Deployment

Already covered in [Installation](#installation) section.

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download es_core_news_lg

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t academic-analyzer .
docker run -p 8501:8501 academic-analyzer
```

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set entry point: `app/Home.py`
5. Add `packages.txt`:
   ```
   python3-dev
   ```
6. Deploy!

### Heroku/Railway Deployment

1. Add `Procfile`:
   ```
   web: streamlit run app/Home.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Add `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. Deploy to Heroku:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_extraction.py

# Verbose output
pytest -v
```

### Code Formatting

```bash
# Format code with Black
black src/ app/ tests/

# Lint with flake8
flake8 src/ app/ tests/

# Type checking (optional)
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Write/update tests
5. Ensure tests pass (`pytest`)
6. Format code (`black .`)
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Guidelines

- All modules use strict dataclasses (see `src/utils/schemas.py`)
- Write unit tests for new functionality
- Use pytest fixtures from `tests/conftest.py`
- Follow the existing code structure
- Update documentation when adding features

---

## Architecture Highlights

### Data Contracts (Schemas)
All modules communicate using strict dataclasses defined in `src/utils/schemas.py`:
- **ExtractedText**: Raw PDF extraction output
- **ProcessedText**: Cleaned and tokenized text
- **FrequencyAnalysis**: TF-IDF and n-gram results
- **TopicModelResult**: Topic modeling output
- **DocumentSkillProfile**: Skills mapping results

### NLP Pipeline
1. **Extraction** → 2. **Preprocessing** → 3. **Analysis** → 4. **Visualization**

### Session State Management
Centralized via `app/utils/session_manager.py`:
- Uploaded PDFs tracking
- Processed texts caching
- Analysis results persistence
- Cross-page state sharing

---

## Known Issues & Limitations

### Python 3.13 Compatibility
- ✅ All core functionality works on Python 3.13
- ⚠️ `gensim` disabled (uses scikit-learn LDA/NMF instead)
- ⚠️ `streamlit-extras` disabled (minor UI limitations)
- 📖 See [PYTHON_313_NOTES.md](PYTHON_313_NOTES.md) for details

### Recommended Production Setup
- Python 3.11 or 3.12 for full ecosystem compatibility
- Python 3.13 for latest features (with minor limitations)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Academic Program Analyzer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## Credits & Acknowledgments

### Libraries & Frameworks
- [Streamlit](https://streamlit.io/) - Web application framework
- [spaCy](https://spacy.io/) - Industrial-strength NLP
- [scikit-learn](https://scikit-learn.org/) - Machine learning toolkit
- [Plotly](https://plotly.com/) - Interactive visualizations
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF text extraction

### Resources
- Spanish spaCy model: `es_core_news_lg`
- Topic modeling techniques: LDA and NMF
- TF-IDF implementation from scikit-learn

---

## Support & Contact

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Documentation**: [docs/](docs/)

---

## Changelog

### v1.0.0 (Current)
- Initial release
- PDF extraction and preprocessing
- Frequency analysis (TF-IDF, n-grams)
- Topic modeling (LDA/NMF)
- Skills mapping
- Multi-program comparison
- Interactive visualizations
- Export functionality

---

**Built with ❤️ using Python, Streamlit, spaCy, and scikit-learn**
