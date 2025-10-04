"""
Carga y gestión de configuración desde config.yaml
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración. Si es None, busca 'config.yaml'
                    en el directorio raíz del proyecto.

    Returns:
        Diccionario con toda la configuración cargada.

    Raises:
        FileNotFoundError: Si el archivo de configuración no existe y no se puede crear uno por defecto.

    Example:
        >>> config = load_config()
        >>> nlp_model = config['nlp']['spacy_model']
        >>> max_topics = config['topics']['max_topics']
    """
    # Si no se proporciona una ruta, usar la predeterminada
    if config_path is None:
        # Buscar el directorio raíz del proyecto (donde está config.yaml)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    # Verificar si el archivo existe
    if not config_path.exists():
        logger.warning(f"Archivo de configuración no encontrado: {config_path}")
        logger.info("Retornando configuración por defecto")
        return _get_default_config()

    # Cargar el archivo YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info(f"Configuración cargada exitosamente desde: {config_path}")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error al parsear el archivo YAML: {e}")
        logger.info("Retornando configuración por defecto")
        return _get_default_config()

    except Exception as e:
        logger.error(f"Error inesperado al cargar configuración: {e}")
        logger.info("Retornando configuración por defecto")
        return _get_default_config()


def _get_default_config() -> Dict[str, Any]:
    """
    Retorna una configuración por defecto en caso de que no exista config.yaml
    o haya errores al cargarlo.

    Returns:
        Diccionario con la configuración por defecto.
    """
    return {
        'app': {
            'name': 'Análisis de Programas de Estudio',
            'version': '1.0.0',
            'debug': False
        },
        'paths': {
            'raw_pdfs': 'data/raw',
            'processed': 'data/processed',
            'cache': 'data/cache',
            'taxonomia': 'data/taxonomia/habilidades.json',
            'stopwords': 'data/taxonomia/stopwords_custom.txt',
            'outputs': 'outputs'
        },
        'nlp': {
            'spacy_model': 'es_core_news_lg',
            'min_word_length': 3,
            'max_word_length': 30,
            'remove_numbers': True,
            'lemmatize': True,
            'pos_tags_keep': ['NOUN', 'VERB', 'ADJ']
        },
        'stopwords_custom': [
            'universidad',
            'asignatura',
            'créditos',
            'ects',
            'semestre',
            'curso',
            'requisito',
            'página'
        ],
        'frequency': {
            'top_n_terms': 50,
            'ngram_range': [1, 3],
            'tfidf_max_features': 500,
            'min_df': 2,
            'max_df': 0.8
        },
        'topics': {
            'default_n_topics': 10,
            'min_topics': 3,
            'max_topics': 20,
            'lda_iterations': 100,
            'lda_passes': 10,
            'coherence_threshold': 0.4
        },
        'skills': {
            'min_confidence': 0.3,
            'weight_tfidf': 0.6,
            'weight_frequency': 0.4
        },
        'ui': {
            'theme': 'light',
            'max_upload_size_mb': 50,
            'results_per_page': 20,
            'cache_ttl_hours': 24
        },
        'export': {
            'default_format': 'xlsx',
            'include_charts': True,
            'chart_dpi': 300
        }
    }


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Obtiene un valor de configuración usando una ruta de claves separada por puntos.

    Args:
        config: Diccionario de configuración
        key_path: Ruta a la clave (ej: 'nlp.spacy_model')
        default: Valor por defecto si la clave no existe

    Returns:
        El valor de configuración o el valor por defecto

    Example:
        >>> config = load_config()
        >>> model = get_config_value(config, 'nlp.spacy_model', 'es_core_news_sm')
    """
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        logger.warning(f"Clave de configuración no encontrada: {key_path}. Usando valor por defecto: {default}")
        return default
