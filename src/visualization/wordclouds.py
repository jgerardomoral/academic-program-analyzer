"""
Generación de WordClouds para visualización de términos frecuentes.

Este módulo proporciona funcionalidad para crear nubes de palabras a partir
de texto o DataFrames de frecuencias, configurables y listas para Streamlit.

CONTRATO:
- Input: Texto plano, DataFrame de frecuencias, o diccionarios
- Output: matplotlib Figure objects compatibles con Streamlit
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.figure
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Union
import logging

from src.utils.config import load_config, get_config_value

logger = logging.getLogger(__name__)


class WordCloudGenerator:
    """
    Generador de WordClouds personalizables.

    Esta clase crea nubes de palabras con estilos configurables,
    soportando múltiples métodos de entrada y opciones de personalización.

    Attributes:
        config: Diccionario de configuración cargado desde config.yaml
        width: Ancho de la imagen en píxeles
        height: Alto de la imagen en píxeles
        background_color: Color de fondo
        colormap: Mapa de colores de matplotlib
        relative_scaling: Escalado relativo del tamaño de fuente
        min_font_size: Tamaño mínimo de fuente
        max_words: Número máximo de palabras a mostrar
        prefer_horizontal: Proporción de palabras horizontales (0-1)

    Example:
        >>> generator = WordCloudGenerator()
        >>> fig = generator.generate_from_text("análisis datos programación")
        >>> # En Streamlit:
        >>> st.pyplot(fig)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el generador de wordclouds.

        Args:
            config_path: Ruta opcional al archivo de configuración.
                        Si es None, usa la configuración por defecto.
        """
        # Cargar configuración
        self.config = load_config(config_path)

        # Extraer parámetros de wordcloud
        wc_config = self.config.get('visualization', {}).get('wordcloud', {})

        self.width = wc_config.get('width', 800)
        self.height = wc_config.get('height', 400)
        self.background_color = wc_config.get('background_color', 'white')
        self.colormap = wc_config.get('colormap', 'viridis')
        self.relative_scaling = wc_config.get('relative_scaling', 0.5)
        self.min_font_size = wc_config.get('min_font_size', 10)
        self.max_words = wc_config.get('max_words', 200)
        self.prefer_horizontal = wc_config.get('prefer_horizontal', 0.7)

        logger.info("WordCloudGenerator inicializado con configuración")

    def generate_from_text(self,
                          text: str,
                          mask_image: Optional[np.ndarray] = None,
                          custom_colormap: Optional[str] = None,
                          custom_background: Optional[str] = None) -> matplotlib.figure.Figure:
        """
        Genera una wordcloud desde texto plano.

        Este método toma un texto y crea una nube de palabras basada en
        la frecuencia de aparición de cada término.

        Args:
            text: Texto del cual generar la wordcloud
            mask_image: Array numpy opcional para definir forma personalizada
            custom_colormap: Colormap personalizado (sobrescribe config)
            custom_background: Color de fondo personalizado (sobrescribe config)

        Returns:
            matplotlib Figure object listo para mostrar en Streamlit

        Raises:
            ValueError: Si el texto está vacío

        Example:
            >>> text = "python machine learning datos análisis"
            >>> fig = generator.generate_from_text(text)
            >>> st.pyplot(fig)
        """
        if not text or not text.strip():
            raise ValueError("El texto no puede estar vacío")

        colormap = custom_colormap or self.colormap
        background = custom_background or self.background_color

        # Crear wordcloud
        wordcloud = WordCloud(
            width=self.width,
            height=self.height,
            background_color=background,
            colormap=colormap,
            mask=mask_image,
            relative_scaling=self.relative_scaling,
            min_font_size=self.min_font_size,
            max_words=self.max_words,
            prefer_horizontal=self.prefer_horizontal,
            collocations=False  # Evitar duplicados de bigramas
        )

        wordcloud.generate(text)

        # Crear figura matplotlib
        fig = self._wordcloud_to_figure(wordcloud)

        logger.info(f"WordCloud generada desde texto ({len(text)} caracteres)")
        return fig

    def generate_from_frequencies(self,
                                 frequencies: Union[pd.DataFrame, Dict[str, float]],
                                 mask_image: Optional[np.ndarray] = None,
                                 custom_colormap: Optional[str] = None,
                                 custom_background: Optional[str] = None) -> matplotlib.figure.Figure:
        """
        Genera una wordcloud desde un DataFrame de frecuencias o diccionario.

        Este método es útil cuando ya tienes calculadas las frecuencias
        o scores TF-IDF de términos.

        Args:
            frequencies: DataFrame con columnas ['term', 'frequency'] o ['term', 'tfidf'],
                        o diccionario {término: frecuencia}
            mask_image: Array numpy opcional para definir forma personalizada
            custom_colormap: Colormap personalizado (sobrescribe config)
            custom_background: Color de fondo personalizado (sobrescribe config)

        Returns:
            matplotlib Figure object listo para mostrar en Streamlit

        Raises:
            ValueError: Si el DataFrame no tiene las columnas requeridas

        Example:
            >>> df = pd.DataFrame({
            ...     'term': ['python', 'datos', 'análisis'],
            ...     'tfidf': [0.9, 0.7, 0.6]
            ... })
            >>> fig = generator.generate_from_frequencies(df)
            >>> st.pyplot(fig)
        """
        # Convertir DataFrame a diccionario si es necesario
        if isinstance(frequencies, pd.DataFrame):
            freq_dict = self._dataframe_to_dict(frequencies)
        elif isinstance(frequencies, dict):
            freq_dict = frequencies
        else:
            raise TypeError("frequencies debe ser DataFrame o diccionario")

        if not freq_dict:
            raise ValueError("El diccionario de frecuencias está vacío")

        colormap = custom_colormap or self.colormap
        background = custom_background or self.background_color

        # Crear wordcloud
        wordcloud = WordCloud(
            width=self.width,
            height=self.height,
            background_color=background,
            colormap=colormap,
            mask=mask_image,
            relative_scaling=self.relative_scaling,
            min_font_size=self.min_font_size,
            max_words=self.max_words,
            prefer_horizontal=self.prefer_horizontal
        )

        wordcloud.generate_from_frequencies(freq_dict)

        # Crear figura matplotlib
        fig = self._wordcloud_to_figure(wordcloud)

        logger.info(f"WordCloud generada desde frecuencias ({len(freq_dict)} términos)")
        return fig

    def save_wordcloud(self,
                      wordcloud_figure: matplotlib.figure.Figure,
                      path: Union[str, Path],
                      dpi: Optional[int] = None) -> None:
        """
        Guarda una wordcloud en archivo.

        Args:
            wordcloud_figure: Figura de matplotlib a guardar
            path: Ruta donde guardar el archivo (PNG, PDF, SVG, etc.)
            dpi: DPI para la imagen. Si es None, usa el de config

        Example:
            >>> fig = generator.generate_from_text("python datos")
            >>> generator.save_wordcloud(fig, "outputs/wordcloud.png")
        """
        if dpi is None:
            dpi = get_config_value(self.config, 'export.chart_dpi', 300)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        wordcloud_figure.savefig(
            path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor=self.background_color
        )

        logger.info(f"WordCloud guardada en: {path}")

    def create_comparative_wordclouds(self,
                                    texts_dict: Dict[str, str],
                                    layout: str = 'horizontal') -> matplotlib.figure.Figure:
        """
        Crea múltiples wordclouds lado a lado para comparación.

        Args:
            texts_dict: Diccionario {título: texto} de hasta 4 documentos
            layout: 'horizontal' o 'grid' (2x2)

        Returns:
            Figura con múltiples subplots

        Example:
            >>> texts = {
            ...     'Programa A': "python datos análisis",
            ...     'Programa B': "java desarrollo software"
            ... }
            >>> fig = generator.create_comparative_wordclouds(texts)
            >>> st.pyplot(fig)
        """
        n_docs = len(texts_dict)

        if n_docs > 4:
            logger.warning("Solo se mostrarán los primeros 4 documentos")
            texts_dict = dict(list(texts_dict.items())[:4])
            n_docs = 4

        # Configurar layout
        if layout == 'horizontal' or n_docs <= 2:
            nrows, ncols = 1, n_docs
            figsize = (self.width / 100 * n_docs, self.height / 100)
        else:
            nrows = 2
            ncols = 2 if n_docs > 2 else 1
            figsize = (self.width / 100 * 2, self.height / 100 * 2)

        # Crear figura
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if n_docs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if nrows > 1 else axes

        # Generar wordclouds
        for idx, (title, text) in enumerate(texts_dict.items()):
            wordcloud = WordCloud(
                width=self.width // ncols,
                height=self.height // nrows,
                background_color=self.background_color,
                colormap=self.colormap,
                relative_scaling=self.relative_scaling,
                min_font_size=self.min_font_size,
                max_words=self.max_words,
                prefer_horizontal=self.prefer_horizontal,
                collocations=False
            ).generate(text)

            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].axis('off')

        # Ocultar ejes sobrantes
        for idx in range(len(texts_dict), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        logger.info(f"WordClouds comparativas generadas para {n_docs} documentos")
        return fig

    # ==================== MÉTODOS PRIVADOS ====================

    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Convierte DataFrame a diccionario de frecuencias.

        Busca columnas 'term' y alguna de: 'tfidf', 'frequency', 'score'
        """
        if 'term' not in df.columns:
            raise ValueError("DataFrame debe tener columna 'term'")

        # Buscar columna de valores
        value_col = None
        for col in ['tfidf', 'frequency', 'score', 'weight']:
            if col in df.columns:
                value_col = col
                break

        if value_col is None:
            raise ValueError("DataFrame debe tener columna 'tfidf', 'frequency', 'score' o 'weight'")

        # Convertir a diccionario
        freq_dict = dict(zip(df['term'], df[value_col]))

        # Filtrar valores negativos o cero
        freq_dict = {k: v for k, v in freq_dict.items() if v > 0}

        return freq_dict

    def _wordcloud_to_figure(self, wordcloud: WordCloud) -> matplotlib.figure.Figure:
        """
        Convierte un objeto WordCloud en una figura matplotlib.
        """
        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)

        return fig

    def load_mask_image(self, mask_path: Union[str, Path]) -> np.ndarray:
        """
        Carga una imagen para usar como máscara de forma.

        Args:
            mask_path: Ruta a la imagen (PNG, JPG, etc.)

        Returns:
            Array numpy para usar como máscara

        Example:
            >>> mask = generator.load_mask_image("masks/circle.png")
            >>> fig = generator.generate_from_text(text, mask_image=mask)
        """
        mask_path = Path(mask_path)

        if not mask_path.exists():
            raise FileNotFoundError(f"Imagen de máscara no encontrada: {mask_path}")

        image = Image.open(mask_path)
        mask_array = np.array(image)

        logger.info(f"Máscara cargada desde: {mask_path}")
        return mask_array


# ==================== FUNCIONES DE CONVENIENCIA ====================

def create_wordcloud_from_text(text: str,
                              config_path: Optional[str] = None,
                              **kwargs) -> matplotlib.figure.Figure:
    """
    Función de conveniencia para generar wordcloud desde texto.

    Args:
        text: Texto para generar wordcloud
        config_path: Ruta opcional a configuración
        **kwargs: Parámetros adicionales para WordCloudGenerator

    Returns:
        Figura matplotlib

    Example:
        >>> fig = create_wordcloud_from_text("python datos análisis")
        >>> st.pyplot(fig)
    """
    generator = WordCloudGenerator(config_path)
    return generator.generate_from_text(text, **kwargs)


def create_wordcloud_from_dataframe(df: pd.DataFrame,
                                   config_path: Optional[str] = None,
                                   **kwargs) -> matplotlib.figure.Figure:
    """
    Función de conveniencia para generar wordcloud desde DataFrame.

    Args:
        df: DataFrame con columnas ['term', 'frequency'] o ['term', 'tfidf']
        config_path: Ruta opcional a configuración
        **kwargs: Parámetros adicionales para WordCloudGenerator

    Returns:
        Figura matplotlib

    Example:
        >>> df = pd.DataFrame({'term': ['python', 'datos'], 'tfidf': [0.9, 0.7]})
        >>> fig = create_wordcloud_from_dataframe(df)
        >>> st.pyplot(fig)
    """
    generator = WordCloudGenerator(config_path)
    return generator.generate_from_frequencies(df, **kwargs)
