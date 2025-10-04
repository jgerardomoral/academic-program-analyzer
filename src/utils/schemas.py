"""
Schemas de datos para garantizar consistencia entre módulos.
Estos son los contratos que todos los módulos deben respetar.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd


@dataclass
class ExtractedText:
    """Texto extraído de un PDF"""
    filename: str
    raw_text: str
    metadata: Dict[str, str]
    page_count: int
    extraction_date: datetime
    has_tables: bool = False
    tables: List[pd.DataFrame] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'filename': self.filename,
            'raw_text': self.raw_text,
            'metadata': self.metadata,
            'page_count': self.page_count,
            'extraction_date': self.extraction_date.isoformat(),
            'has_tables': self.has_tables,
            'table_count': len(self.tables)
        }


@dataclass
class ProcessedText:
    """Texto procesado y limpio"""
    filename: str
    clean_text: str
    tokens: List[str]
    lemmas: List[str]
    pos_tags: List[str]
    entities: List[Dict[str, str]]
    metadata: Dict[str, str]
    processing_date: datetime

    def to_dict(self) -> dict:
        return {
            'filename': self.filename,
            'clean_text': self.clean_text,
            'token_count': len(self.tokens),
            'unique_tokens': len(set(self.tokens)),
            'metadata': self.metadata,
            'processing_date': self.processing_date.isoformat()
        }


@dataclass
class FrequencyAnalysis:
    """Resultado de análisis de frecuencias"""
    document_id: str
    term_frequencies: pd.DataFrame  # columns: [term, frequency, tfidf]
    ngrams: Dict[int, pd.DataFrame]  # {1: unigrams_df, 2: bigrams_df, 3: trigrams_df}
    top_terms: List[str]
    vocabulary_size: int
    analysis_date: datetime


@dataclass
class Topic:
    """Representación de un topic"""
    topic_id: int
    keywords: List[str]
    weights: List[float]
    label: str  # Etiqueta manual
    coherence_score: float
    documents: List[str]  # IDs de documentos donde aparece


@dataclass
class TopicModelResult:
    """Resultado de topic modeling"""
    model_type: str  # 'LDA' o 'NMF'
    n_topics: int
    topics: List[Topic]
    document_topic_matrix: pd.DataFrame
    coherence_score: float
    perplexity: Optional[float] = None


@dataclass
class Skill:
    """Definición de una habilidad"""
    skill_id: str
    name: str
    keywords: List[str]
    synonyms: List[str]
    weight: float
    category: str


@dataclass
class SkillScore:
    """Score de una habilidad en un documento"""
    skill_id: str
    skill_name: str
    score: float  # 0-1
    confidence: float  # 0-1
    matched_terms: List[str]
    context_snippets: List[str]


@dataclass
class DocumentSkillProfile:
    """Perfil de habilidades de un documento"""
    document_id: str
    skill_scores: List[SkillScore]
    top_skills: List[str]
    skill_coverage: float  # % del texto mapeado a habilidades
    analysis_date: datetime

    def to_dataframe(self) -> pd.DataFrame:
        """Convierte a DataFrame para visualización"""
        return pd.DataFrame([
            {
                'skill': ss.skill_name,
                'score': ss.score,
                'confidence': ss.confidence,
                'matched_terms': ', '.join(ss.matched_terms[:3])
            }
            for ss in self.skill_scores
        ])
