"""
Módulo de visualización para el análisis de programas de estudio.

Este módulo proporciona herramientas para crear visualizaciones interactivas
y nubes de palabras para Streamlit.

Componentes:
- wordclouds: Generación de nubes de palabras
- plotly_charts: Gráficos interactivos con Plotly
"""

from src.visualization.wordclouds import (
    WordCloudGenerator,
    create_wordcloud_from_text,
    create_wordcloud_from_dataframe
)

from src.visualization.plotly_charts import (
    PlotlyChartsConfig,
    PlotlyCharts,
    create_frequency_bar_chart,
    create_tfidf_scatter,
    create_topic_heatmap,
    create_topic_distribution_stacked,
    create_skills_radar_chart,
    create_comparison_chart,
    create_skills_heatmap,
    create_ngrams_comparison,
    create_cooccurrence_network,
    create_skills_bar_comparison
)

__all__ = [
    # WordCloud classes and functions
    'WordCloudGenerator',
    'create_wordcloud_from_text',
    'create_wordcloud_from_dataframe',

    # Plotly classes and functions
    'PlotlyChartsConfig',
    'PlotlyCharts',
    'create_frequency_bar_chart',
    'create_tfidf_scatter',
    'create_topic_heatmap',
    'create_topic_distribution_stacked',
    'create_skills_radar_chart',
    'create_comparison_chart',
    'create_skills_heatmap',
    'create_ngrams_comparison',
    'create_cooccurrence_network',
    'create_skills_bar_comparison'
]
