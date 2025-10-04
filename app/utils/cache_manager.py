"""
Gestión de caché para funciones pesadas utilizando decoradores de Streamlit.
Proporciona caching para operaciones costosas como extracción de PDF, procesamiento de texto y análisis.
"""
import streamlit as st
from functools import wraps
from typing import Dict, Any, List, Callable
import hashlib
import pickle


@st.cache_data(ttl=3600, show_spinner="Extrayendo texto...")
def cached_pdf_extraction(pdf_bytes: bytes, filename: str):
    """Cache de extracción de PDFs"""
    from src.extraction.pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    # pdf_bytes viene de uploaded_file.read()
    import io
    pdf_file = io.BytesIO(pdf_bytes)

    return extractor.extract(pdf_file, metadata={'filename': filename})


@st.cache_data(ttl=3600, show_spinner="Procesando texto...")
def cached_text_processing(extracted_dict: dict):
    """Cache de procesamiento de texto"""
    from src.extraction.preprocessor import TextPreprocessor
    from src.utils.schemas import ExtractedText
    from datetime import datetime

    # Reconstruir ExtractedText desde dict
    extracted = ExtractedText(
        filename=extracted_dict['filename'],
        raw_text=extracted_dict['raw_text'],
        metadata=extracted_dict['metadata'],
        page_count=extracted_dict['page_count'],
        extraction_date=datetime.fromisoformat(extracted_dict['extraction_date']),
        has_tables=extracted_dict.get('has_tables', False),
        tables=[]
    )

    preprocessor = TextPreprocessor()
    return preprocessor.process(extracted)


@st.cache_resource
def load_spacy_model():
    """Carga modelo spaCy una sola vez"""
    import spacy
    return spacy.load("es_core_news_lg")


@st.cache_data(ttl=7200, show_spinner="Analizando frecuencias...")
def cached_frequency_analysis(processed_dict: dict):
    """Cache de análisis de frecuencias"""
    from src.analysis.frequency import FrequencyAnalyzer
    from src.utils.schemas import ProcessedText
    from datetime import datetime

    # Reconstruir ProcessedText
    processed = ProcessedText(
        filename=processed_dict['filename'],
        clean_text=processed_dict['clean_text'],
        tokens=processed_dict['tokens'],
        lemmas=processed_dict['lemmas'],
        pos_tags=processed_dict['pos_tags'],
        entities=processed_dict['entities'],
        metadata=processed_dict['metadata'],
        processing_date=datetime.fromisoformat(processed_dict['processing_date'])
    )

    analyzer = FrequencyAnalyzer()
    return analyzer.analyze_single(processed)


@st.cache_data(ttl=7200, show_spinner="Generando modelo de topics...")
def cached_topic_modeling(processed_dicts: list, n_topics: int, method: str):
    """Cache de topic modeling"""
    from src.analysis.topics import TopicModeler
    from src.utils.schemas import ProcessedText
    from datetime import datetime

    # Reconstruir ProcessedTexts
    documents = []
    for pd in processed_dicts:
        doc = ProcessedText(
            filename=pd['filename'],
            clean_text=pd['clean_text'],
            tokens=pd['tokens'],
            lemmas=pd['lemmas'],
            pos_tags=pd['pos_tags'],
            entities=pd['entities'],
            metadata=pd['metadata'],
            processing_date=datetime.fromisoformat(pd['processing_date'])
        )
        documents.append(doc)

    modeler = TopicModeler()
    return modeler.fit(documents, n_topics=n_topics, method=method)


@st.cache_data(ttl=7200, show_spinner="Analizando habilidades...")
def cached_skills_analysis(processed_dict: dict):
    """Cache de análisis de habilidades"""
    from src.analysis.skills_mapper import SkillsMapper
    from src.utils.schemas import ProcessedText
    from datetime import datetime

    # Reconstruir ProcessedText
    processed = ProcessedText(
        filename=processed_dict['filename'],
        clean_text=processed_dict['clean_text'],
        tokens=processed_dict['tokens'],
        lemmas=processed_dict['lemmas'],
        pos_tags=processed_dict['pos_tags'],
        entities=processed_dict['entities'],
        metadata=processed_dict['metadata'],
        processing_date=datetime.fromisoformat(processed_dict['processing_date'])
    )

    mapper = SkillsMapper()
    return mapper.map_skills(processed)


# Decoradores personalizados para aplicar a funciones del usuario
def cache_extraction(func: Callable) -> Callable:
    """
    Decorador para cachear funciones de extracción de PDF.
    Usa st.cache_data con TTL de 1 hora.

    Args:
        func: Función a cachear

    Returns:
        Función decorada con caché
    """
    return st.cache_data(ttl=3600, show_spinner="Extrayendo datos...")(func)


def cache_preprocessing(func: Callable) -> Callable:
    """
    Decorador para cachear funciones de preprocesamiento de texto.
    Usa st.cache_data con TTL de 1 hora.

    Args:
        func: Función a cachear

    Returns:
        Función decorada con caché
    """
    return st.cache_data(ttl=3600, show_spinner="Procesando texto...")(func)


def cache_analysis(func: Callable) -> Callable:
    """
    Decorador para cachear funciones de análisis.
    Usa st.cache_data con TTL de 2 horas.

    Args:
        func: Función a cachear

    Returns:
        Función decorada con caché
    """
    return st.cache_data(ttl=7200, show_spinner="Analizando...")(func)


def clear_cache() -> None:
    """
    Limpia todas las cachés de Streamlit.
    Incluye tanto st.cache_data como st.cache_resource.
    """
    st.cache_data.clear()
    st.cache_resource.clear()


def clear_cache_data() -> None:
    """
    Limpia solo la caché de datos (st.cache_data).
    """
    st.cache_data.clear()


def clear_cache_resource() -> None:
    """
    Limpia solo la caché de recursos (st.cache_resource).
    """
    st.cache_resource.clear()


def get_cache_stats() -> Dict[str, Any]:
    """
    Retorna estadísticas sobre el uso de caché.

    Returns:
        Diccionario con estadísticas de caché (actualmente limitado por la API de Streamlit)
    """
    # Nota: Streamlit no proporciona una API directa para obtener estadísticas de caché
    # Esta es una implementación básica que puede expandirse en el futuro
    import sys

    stats = {
        'cache_available': True,
        'cache_data_enabled': hasattr(st, 'cache_data'),
        'cache_resource_enabled': hasattr(st, 'cache_resource'),
        'python_version': sys.version,
        'streamlit_version': st.__version__ if hasattr(st, '__version__') else 'Unknown'
    }

    return stats


def invalidate_cached_extraction(filename: str) -> None:
    """
    Invalida la caché de extracción para un archivo específico.
    Nota: Requiere limpiar toda la caché de extracción debido a limitaciones de Streamlit.

    Args:
        filename: Nombre del archivo cuya caché se debe invalidar
    """
    # Streamlit no permite invalidación selectiva, así que limpiamos toda la caché
    cached_pdf_extraction.clear()


def invalidate_cached_processing(filename: str) -> None:
    """
    Invalida la caché de procesamiento para un archivo específico.
    Nota: Requiere limpiar toda la caché de procesamiento debido a limitaciones de Streamlit.

    Args:
        filename: Nombre del archivo cuya caché se debe invalidar
    """
    # Streamlit no permite invalidación selectiva, así que limpiamos toda la caché
    cached_text_processing.clear()


def invalidate_cached_analysis(filename: str) -> None:
    """
    Invalida la caché de análisis para un archivo específico.
    Nota: Requiere limpiar toda la caché de análisis debido a limitaciones de Streamlit.

    Args:
        filename: Nombre del archivo cuya caché se debe invalidar
    """
    # Streamlit no permite invalidación selectiva, así que limpiamos toda la caché
    cached_frequency_analysis.clear()
    cached_skills_analysis.clear()
