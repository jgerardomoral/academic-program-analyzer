"""
Gestión centralizada de st.session_state para la aplicación Streamlit.
Proporciona funciones para inicializar, obtener, establecer y limpiar variables de sesión.
"""
import streamlit as st
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

# Agregar src al path si no está presente
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils.config import load_config


def init_session_state():
    """
    Inicializa todas las variables de sesión necesarias para la aplicación.
    Esta función debe llamarse al inicio de cada página de Streamlit.

    Variables inicializadas:
    - config: Configuración de la aplicación
    - uploaded_pdfs: Diccionario de archivos PDF cargados {filename: bytes}
    - extracted_texts: Diccionario de textos extraídos {filename: ExtractedText}
    - processed_texts: Diccionario de textos procesados {filename: ProcessedText}
    - frequency_analyses: Diccionario de análisis de frecuencia {filename: FrequencyAnalysis}
    - topic_model: Modelo de tópicos (TopicModelResult o None)
    - skill_profiles: Diccionario de perfiles de habilidades {filename: DocumentSkillProfile}
    - taxonomia: Taxonomía de habilidades cargada
    - current_page: Página actual de la aplicación
    - current_view: Vista actual (para navegación)
    - selected_documents: Lista de documentos seleccionados para análisis
    - processing_status: Estado de procesamiento {filename: status_msg}
    """

    # Configuración
    if 'config' not in st.session_state:
        try:
            st.session_state.config = load_config()
        except Exception as e:
            # Fallback a configuración por defecto si no se puede cargar
            st.session_state.config = {
                'app_name': 'Análisis de Programas de Estudio',
                'max_upload_size': 10,  # MB
                'supported_formats': ['pdf'],
                'default_n_topics': 5
            }

    # PDFs y procesamiento
    if 'uploaded_pdfs' not in st.session_state:
        st.session_state.uploaded_pdfs = {}  # {filename: file_bytes}

    if 'extracted_texts' not in st.session_state:
        st.session_state.extracted_texts = {}  # {filename: ExtractedText}

    if 'processed_texts' not in st.session_state:
        st.session_state.processed_texts = {}  # {filename: ProcessedText}

    # Análisis
    if 'frequency_analyses' not in st.session_state:
        st.session_state.frequency_analyses = {}  # {filename: FrequencyAnalysis}

    if 'topic_model' not in st.session_state:
        st.session_state.topic_model = None  # TopicModelResult

    if 'skill_profiles' not in st.session_state:
        st.session_state.skill_profiles = {}  # {filename: DocumentSkillProfile}

    # Taxonomía de habilidades
    if 'taxonomia' not in st.session_state:
        try:
            from src.analysis.skills_mapper import SkillsMapper
            mapper = SkillsMapper()
            st.session_state.taxonomia = mapper.taxonomia
        except Exception as e:
            # Taxonomía vacía si no se puede cargar
            st.session_state.taxonomia = {}

    # Estado de la UI
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'

    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'home'

    if 'selected_documents' not in st.session_state:
        st.session_state.selected_documents = []

    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}  # {filename: status_msg}


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Obtiene un valor del estado de sesión de forma segura.

    Args:
        key: Clave del valor a obtener
        default: Valor por defecto si la clave no existe

    Returns:
        Valor almacenado o valor por defecto
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """
    Establece un valor en el estado de sesión de forma segura.

    Args:
        key: Clave donde almacenar el valor
        value: Valor a almacenar
    """
    st.session_state[key] = value


def clear_session_key(key: str) -> None:
    """
    Elimina una clave específica del estado de sesión.

    Args:
        key: Clave a eliminar
    """
    if key in st.session_state:
        del st.session_state[key]


def clear_session_state() -> None:
    """
    Reinicia todas las variables de sesión a su estado inicial.
    Útil para limpiar completamente la aplicación.
    """
    # Limpiar PDFs y datos procesados
    st.session_state.uploaded_pdfs = {}
    st.session_state.extracted_texts = {}
    st.session_state.processed_texts = {}

    # Limpiar análisis
    st.session_state.frequency_analyses = {}
    st.session_state.topic_model = None
    st.session_state.skill_profiles = {}

    # Limpiar estado de UI
    st.session_state.selected_documents = []
    st.session_state.processing_status = {}
    st.session_state.current_view = 'home'


def reset_analysis() -> None:
    """
    Reinicia solo los resultados de análisis, manteniendo los PDFs cargados.
    Útil cuando se quiere reanalizar documentos con diferentes parámetros.
    """
    st.session_state.frequency_analyses = {}
    st.session_state.topic_model = None
    st.session_state.skill_profiles = {}
    st.session_state.processing_status = {}


def reset_uploads() -> None:
    """
    Reinicia solo los PDFs cargados y todos los análisis derivados.
    """
    st.session_state.uploaded_pdfs = {}
    st.session_state.extracted_texts = {}
    st.session_state.processed_texts = {}
    reset_analysis()


def get_uploaded_count() -> int:
    """
    Retorna el número de PDFs cargados.

    Returns:
        Cantidad de PDFs en sesión
    """
    return len(st.session_state.get('uploaded_pdfs', {}))


def get_processed_count() -> int:
    """
    Retorna el número de documentos procesados.

    Returns:
        Cantidad de documentos procesados
    """
    return len(st.session_state.get('processed_texts', {}))


def get_analyzed_count() -> int:
    """
    Retorna el número de análisis de frecuencia completados.

    Returns:
        Cantidad de análisis realizados
    """
    return len(st.session_state.get('frequency_analyses', {}))


def has_topic_model() -> bool:
    """
    Verifica si existe un modelo de tópicos generado.

    Returns:
        True si hay modelo de tópicos, False en caso contrario
    """
    return st.session_state.get('topic_model') is not None


def get_document_list() -> List[str]:
    """
    Retorna la lista de nombres de documentos cargados.

    Returns:
        Lista de nombres de archivos
    """
    return list(st.session_state.get('uploaded_pdfs', {}).keys())


def get_processed_document_list() -> List[str]:
    """
    Retorna la lista de nombres de documentos procesados.

    Returns:
        Lista de nombres de archivos procesados
    """
    return list(st.session_state.get('processed_texts', {}).keys())


def is_document_processed(filename: str) -> bool:
    """
    Verifica si un documento específico ha sido procesado.

    Args:
        filename: Nombre del archivo a verificar

    Returns:
        True si está procesado, False en caso contrario
    """
    return filename in st.session_state.get('processed_texts', {})


def is_document_analyzed(filename: str) -> bool:
    """
    Verifica si un documento específico ha sido analizado.

    Args:
        filename: Nombre del archivo a verificar

    Returns:
        True si está analizado, False en caso contrario
    """
    return filename in st.session_state.get('frequency_analyses', {})


def get_session_stats() -> Dict[str, Any]:
    """
    Retorna estadísticas generales de la sesión actual.

    Returns:
        Diccionario con estadísticas de sesión
    """
    return {
        'uploaded_pdfs': get_uploaded_count(),
        'processed_texts': get_processed_count(),
        'frequency_analyses': get_analyzed_count(),
        'skill_profiles': len(st.session_state.get('skill_profiles', {})),
        'has_topic_model': has_topic_model(),
        'selected_documents': len(st.session_state.get('selected_documents', [])),
        'current_page': st.session_state.get('current_page', 'Unknown')
    }


def update_processing_status(filename: str, status: str) -> None:
    """
    Actualiza el estado de procesamiento de un documento.

    Args:
        filename: Nombre del archivo
        status: Mensaje de estado
    """
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    st.session_state.processing_status[filename] = status


def get_processing_status(filename: str) -> Optional[str]:
    """
    Obtiene el estado de procesamiento de un documento.

    Args:
        filename: Nombre del archivo

    Returns:
        Mensaje de estado o None
    """
    return st.session_state.get('processing_status', {}).get(filename)


def clear_processing_status(filename: str) -> None:
    """
    Limpia el estado de procesamiento de un documento específico.

    Args:
        filename: Nombre del archivo
    """
    if 'processing_status' in st.session_state and filename in st.session_state.processing_status:
        del st.session_state.processing_status[filename]


# Aliases for backward compatibility
get_state = get_session_value
set_state = set_session_value
clear_state = clear_session_key
