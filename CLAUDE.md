# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Streamlit-based application for analyzing academic study programs** (programas de estudio). It extracts text from PDF documents, performs NLP analysis (frequency analysis, topic modeling, skills mapping), and provides interactive visualizations.

**Core functionality:**
- Extract and preprocess text from academic PDFs
- Analyze term frequencies using TF-IDF and n-grams
- Discover topics using LDA/NMF topic modeling
- Map keywords to skills taxonomy
- Compare multiple study programs
- Generate interactive visualizations and reports

## Project Structure

The project follows a modular architecture with clear separation between backend processing and frontend UI:

```
src/                      # Backend processing modules
├── extraction/          # PDF text extraction and preprocessing
├── analysis/           # NLP analysis (frequency, topics, skills mapping)
├── visualization/      # Plotly charts, wordclouds, reports
└── utils/              # Config management, file I/O, schemas

app/                     # Streamlit frontend
├── Home.py            # Main entry point
├── pages/             # Multi-page app sections
├── components/        # Reusable UI components
└── utils/             # Session state, caching, exports

data/                   # Data storage
├── raw/               # Original PDFs
├── processed/         # Extracted texts (JSON/pickle)
├── cache/             # Analysis cache
└── taxonomia/         # Skills taxonomy and custom stopwords

tests/                  # Pytest test suite
notebooks/             # Jupyter notebooks for exploration
outputs/               # Generated reports and exports
```

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download Spanish spaCy model
python -m spacy download es_core_news_lg
```

**Python 3.13 Note**: Some packages (gensim, streamlit-extras) have compatibility issues. They are temporarily disabled in requirements.txt. All core functionality works - see [PYTHON_313_NOTES.md](PYTHON_313_NOTES.md).

### Running the Application
```bash
# Run Streamlit app (entry point: app/Home.py)
streamlit run app/Home.py
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_extraction.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format code with black
black src/ app/ tests/

# Lint with flake8
flake8 src/ app/ tests/
```

## Key Architecture Patterns

### Data Contracts (Schemas)
All modules use strict dataclasses defined in `src/utils/schemas.py` to ensure consistent data flow:

- **ExtractedText**: Raw PDF extraction output
- **ProcessedText**: Cleaned and tokenized text
- **FrequencyAnalysis**: TF-IDF and n-gram results
- **TopicModelResult**: Topic modeling output
- **DocumentSkillProfile**: Skills mapping results

These schemas define the **contracts between modules** - any module consuming data expects these exact formats.

### NLP Pipeline
The processing pipeline follows these stages:

1. **Extraction** (`src/extraction/pdf_extractor.py`): Uses pdfplumber to extract text and tables from PDFs
2. **Preprocessing** (`src/extraction/preprocessor.py`): Cleans text, tokenizes, lemmatizes using spaCy
3. **Analysis**:
   - `src/analysis/frequency.py`: TF-IDF, n-grams
   - `src/analysis/topics.py`: LDA/NMF topic modeling
   - `src/analysis/skills_mapper.py`: Maps keywords to skills taxonomy
4. **Visualization** (`src/visualization/`): Generates interactive charts and reports

### Streamlit State Management
The app uses `app/utils/session_manager.py` to centralize all session state:

- `uploaded_pdfs`: List of uploaded PDF files
- `processed_texts`: Dictionary of processed texts by filename
- `frequency_analyses`: Cached frequency analysis results
- `topic_model`: Current topic modeling results
- `skill_profiles`: Skill mapping results per document

**Important**: Always use `init_session_state()` at the start of each page to ensure state is initialized.

### Configuration
All configuration is centralized in `config.yaml`:

- **Paths**: Data directories, output locations
- **NLP settings**: spaCy model, stopwords, POS tags to keep
- **Analysis parameters**: TF-IDF settings, topic count ranges, skill mapping thresholds
- **UI settings**: Theme, upload limits, cache TTL

Load config using: `from src.utils.config import load_config`

## Development Workflow

### Parallel Development Strategy
The plan emphasizes modular development where multiple components can be built in parallel:

1. **Always create/update schemas first** (`src/utils/schemas.py`) to establish contracts
2. **Use test fixtures** (`tests/conftest.py`) for mock data during development
3. **Write unit tests** alongside implementation to validate contracts
4. **Independent module development**: Each module has defined inputs/outputs

### Testing with Mock Data
Use pytest fixtures from `tests/conftest.py`:

- `mock_extracted_text`: Simulated PDF extraction
- `mock_processed_text`: Simulated preprocessing output
- `mock_frequency_df`: Sample frequency analysis
- `mock_taxonomia`: Skills taxonomy for testing

This allows developing and testing modules without requiring actual PDFs.

## Critical Dependencies

**NLP Processing:**
- spaCy (Spanish model: `es_core_news_lg`)
- scikit-learn (TF-IDF, vectorization)
- gensim (LDA topic modeling)

**PDF Processing:**
- pdfplumber (primary extractor)
- PyPDF2 (fallback/metadata)

**Visualization:**
- plotly (interactive charts)
- wordcloud (word clouds)
- streamlit (UI framework)

**Data:**
- pandas (data manipulation)
- numpy (numerical operations)

## Important Notes

- **Language**: All text processing is configured for Spanish (`es_core_news_lg`)
- **Custom stopwords**: Academic terms are filtered via `data/taxonomia/stopwords_custom.txt` and `config.yaml`
- **Caching**: Streamlit caching is managed via `app/utils/cache_manager.py` - use provided decorators
- **Entry point**: The Streamlit app starts at `app/Home.py` (not `main.py` or `app.py`)
- **Multi-page app**: Streamlit pages are in `app/pages/` with numeric prefixes for ordering

## Common Tasks

### Adding a new analysis module
1. Define output schema in `src/utils/schemas.py`
2. Create module in appropriate `src/` subdirectory
3. Add test fixtures in `tests/conftest.py`
4. Write unit tests
5. Integrate into Streamlit page

### Adding a new Streamlit page
1. Create `app/pages/N_PageName.py` (N sets order)
2. Import and call `init_session_state()` at top
3. Use components from `app/components/` for consistent UI
4. Add caching for expensive operations using `cache_manager.py`

### Modifying NLP configuration
Edit `config.yaml`:
- `nlp` section: spaCy settings, stopwords
- `frequency` section: TF-IDF parameters
- `topics` section: Topic modeling parameters
- `skills` section: Skills mapping thresholds
