"""
Fixtures de pytest para datos mock.
Permite desarrollo paralelo sin PDFs reales.
"""
import pytest
from datetime import datetime
from src.utils.schemas import ExtractedText, ProcessedText, FrequencyAnalysis
import pandas as pd


@pytest.fixture
def mock_extracted_text():
    """PDF extraído simulado"""
    return ExtractedText(
        filename="Matematicas_I.pdf",
        raw_text="""
        MATEMÁTICAS I

        Objetivos:
        - Desarrollar pensamiento lógico y analítico
        - Resolver problemas mediante algoritmos
        - Aplicar técnicas de cálculo diferencial

        Contenidos:
        1. Álgebra lineal
        2. Cálculo diferencial
        3. Métodos numéricos

        Evaluación:
        - Exámenes parciales (40%)
        - Proyecto final (30%)
        - Tareas (30%)
        """,
        metadata={
            'programa': 'Ingeniería en Computación',
            'facultad': 'Ciencias',
            'año': '2024',
            'semestre': '1'
        },
        page_count=5,
        extraction_date=datetime.now(),
        has_tables=False
    )


@pytest.fixture
def mock_processed_text():
    """Texto procesado simulado"""
    return ProcessedText(
        filename="Matematicas_I.pdf",
        clean_text="matemáticas desarrollar pensamiento lógico analítico resolver problemas algoritmos",
        tokens=['matemáticas', 'desarrollar', 'pensamiento', 'lógico', 'analítico',
                'resolver', 'problemas', 'algoritmos', 'aplicar', 'cálculo'],
        lemmas=['matemática', 'desarrollar', 'pensamiento', 'lógico', 'analítico',
                'resolver', 'problema', 'algoritmo', 'aplicar', 'cálculo'],
        pos_tags=['NOUN', 'VERB', 'NOUN', 'ADJ', 'ADJ',
                  'VERB', 'NOUN', 'NOUN', 'VERB', 'NOUN'],
        entities=[],
        metadata={
            'programa': 'Ingeniería en Computación',
            'año': '2024'
        },
        processing_date=datetime.now()
    )


@pytest.fixture
def mock_frequency_df():
    """DataFrame de frecuencias simulado"""
    return pd.DataFrame({
        'term': ['algoritmo', 'problema', 'cálculo', 'lógico', 'analítico'],
        'frequency': [15, 12, 10, 8, 7],
        'tfidf': [0.85, 0.72, 0.68, 0.55, 0.51]
    })


@pytest.fixture
def mock_taxonomia():
    """Taxonomía de habilidades simulada"""
    return {
        "pensamiento_critico": {
            "name": "Pensamiento Crítico",
            "keywords": ["análisis", "evaluar", "criticar", "argumentar", "razonar"],
            "synonyms": ["analítico", "crítico", "evaluativo"],
            "weight": 1.0,
            "category": "cognitiva"
        },
        "programacion": {
            "name": "Programación",
            "keywords": ["código", "programar", "algoritmo", "software", "desarrollo"],
            "synonyms": ["codificar", "implementar", "desarrollar"],
            "weight": 1.0,
            "category": "tecnica"
        },
        "resolucion_problemas": {
            "name": "Resolución de Problemas",
            "keywords": ["resolver", "problema", "solución", "optimizar"],
            "synonyms": ["solucionar", "abordar"],
            "weight": 1.0,
            "category": "cognitiva"
        },
        "matematicas": {
            "name": "Matemáticas",
            "keywords": ["cálculo", "álgebra", "geometría", "estadística", "matemática"],
            "synonyms": ["numérico", "cuantitativo"],
            "weight": 0.9,
            "category": "tecnica"
        }
    }


@pytest.fixture
def mock_frequency_analysis():
    """FrequencyAnalysis object simulado para tests"""
    term_df = pd.DataFrame({
        'term': ['algoritmo', 'problema', 'cálculo', 'lógico', 'analítico',
                 'resolver', 'matemática', 'programación'],
        'frequency': [15, 12, 10, 8, 7, 6, 5, 4],
        'tfidf': [0.85, 0.72, 0.68, 0.55, 0.51, 0.48, 0.45, 0.42]
    })

    ngrams = {
        1: pd.DataFrame({
            'ngram': ['algoritmo', 'problema', 'cálculo'],
            'frequency': [15, 12, 10]
        }),
        2: pd.DataFrame({
            'ngram': ['algoritmo problema', 'resolver problema', 'pensamiento lógico'],
            'frequency': [5, 4, 3]
        }),
        3: pd.DataFrame({
            'ngram': ['resolver problema algoritmo', 'pensamiento lógico analítico'],
            'frequency': [2, 2]
        })
    }

    return FrequencyAnalysis(
        document_id="Matematicas_I.pdf",
        term_frequencies=term_df,
        ngrams=ngrams,
        top_terms=['algoritmo', 'problema', 'cálculo', 'lógico', 'analítico'],
        vocabulary_size=8,
        analysis_date=datetime.now()
    )
