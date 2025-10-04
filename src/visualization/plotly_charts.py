"""
Gráficos interactivos con Plotly para Streamlit.

Este módulo proporciona funciones para crear visualizaciones interactivas
de análisis de texto, habilidades y topics usando Plotly.

CONTRATO:
- Input: DataFrames de análisis, listas de schemas
- Output: plotly.graph_objects.Figure (directamente renderizables en Streamlit)
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
import logging

from src.utils.config import load_config, get_config_value
from src.utils.schemas import SkillScore, DocumentSkillProfile

logger = logging.getLogger(__name__)


class PlotlyChartsConfig:
    """
    Configuración centralizada para gráficos Plotly.

    Esta clase mantiene todos los parámetros de estilo y configuración
    para garantizar consistencia visual en todos los gráficos.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa configuración de gráficos.

        Args:
            config_path: Ruta opcional al archivo de configuración
        """
        self.config = load_config(config_path)
        plotly_config = self.config.get('visualization', {}).get('plotly', {})

        self.theme = plotly_config.get('theme', 'plotly_white')
        self.color_scale = plotly_config.get('color_scale', 'Viridis')
        self.default_height = plotly_config.get('default_height', 500)
        self.bar_chart_height = plotly_config.get('bar_chart_height', 400)
        self.heatmap_row_height = plotly_config.get('heatmap_row_height', 30)
        self.radar_height = plotly_config.get('radar_height', 500)
        self.network_height = plotly_config.get('network_height', 600)

        colors = plotly_config.get('colors', {})
        self.colors = [
            colors.get('primary', '#636EFA'),
            colors.get('secondary', '#EF553B'),
            colors.get('tertiary', '#00CC96'),
            colors.get('quaternary', '#AB63FA')
        ]

        heatmap_cfg = plotly_config.get('heatmap', {})
        self.heatmap_colorscale = heatmap_cfg.get('colorscale', 'RdYlGn')
        self.heatmap_threshold = heatmap_cfg.get('show_values_threshold', 0.1)


# ==================== GRÁFICOS DE FRECUENCIAS ====================

def create_frequency_bar_chart(df: pd.DataFrame,
                               title: str = "Términos Más Frecuentes",
                               top_n: int = 20,
                               value_column: str = 'tfidf',
                               config_path: Optional[str] = None) -> go.Figure:
    """
    Crea un gráfico de barras horizontales de términos frecuentes.

    Args:
        df: DataFrame con columnas ['term', 'frequency', 'tfidf']
        title: Título del gráfico
        top_n: Número de términos a mostrar
        value_column: Columna para ordenar ('tfidf', 'frequency')
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly interactiva

    Example:
        >>> df = pd.DataFrame({
        ...     'term': ['python', 'datos'],
        ...     'frequency': [10, 8],
        ...     'tfidf': [0.9, 0.7]
        ... })
        >>> fig = create_frequency_bar_chart(df)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    # Validar columnas
    required_cols = ['term', value_column]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame debe tener columnas: {required_cols}")

    # Tomar top N términos
    df_top = df.nlargest(top_n, value_column)

    # Crear gráfico
    fig = px.bar(
        df_top,
        x=value_column,
        y='term',
        orientation='h',
        title=title,
        labels={
            value_column: 'TF-IDF Score' if value_column == 'tfidf' else 'Frecuencia',
            'term': 'Término'
        },
        color=value_column,
        color_continuous_scale=cfg.color_scale,
        hover_data=['frequency'] if 'frequency' in df.columns else None
    )

    # Actualizar layout
    chart_height = max(cfg.bar_chart_height, top_n * 25)
    fig.update_layout(
        height=chart_height,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        template=cfg.theme,
        hovermode='closest'
    )

    logger.info(f"Gráfico de barras creado: {title}")
    return fig


def create_tfidf_scatter(df: pd.DataFrame,
                        title: str = "Análisis TF-IDF de Términos",
                        config_path: Optional[str] = None) -> go.Figure:
    """
    Crea un gráfico de dispersión de TF-IDF vs Frecuencia.

    Este gráfico ayuda a identificar términos importantes (alto TF-IDF)
    que no son necesariamente los más frecuentes.

    Args:
        df: DataFrame con columnas ['term', 'frequency', 'tfidf']
        title: Título del gráfico
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly interactiva

    Example:
        >>> fig = create_tfidf_scatter(frequency_df)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    # Validar columnas
    required_cols = ['term', 'frequency', 'tfidf']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame debe tener columnas: {required_cols}")

    # Crear scatter plot
    fig = px.scatter(
        df,
        x='frequency',
        y='tfidf',
        text='term',
        title=title,
        labels={
            'frequency': 'Frecuencia',
            'tfidf': 'TF-IDF Score',
            'term': 'Término'
        },
        color='tfidf',
        color_continuous_scale=cfg.color_scale,
        size='frequency',
        size_max=20
    )

    # Ajustar posición del texto
    fig.update_traces(
        textposition='top center',
        textfont_size=8
    )

    fig.update_layout(
        height=cfg.default_height,
        template=cfg.theme,
        hovermode='closest'
    )

    logger.info(f"Gráfico de dispersión TF-IDF creado")
    return fig


# ==================== GRÁFICOS DE TOPICS ====================

def create_topic_heatmap(doc_topic_matrix: pd.DataFrame,
                        title: str = "Distribución de Topics por Documento",
                        topic_labels: Optional[List[str]] = None,
                        config_path: Optional[str] = None) -> go.Figure:
    """
    Crea un heatmap de distribución de topics en documentos.

    Args:
        doc_topic_matrix: DataFrame con documentos en filas, topics en columnas
        title: Título del gráfico
        topic_labels: Etiquetas opcionales para topics
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly interactiva

    Example:
        >>> # doc_topic_matrix tiene shape (n_docs, n_topics)
        >>> fig = create_topic_heatmap(doc_topic_matrix)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    # Aplicar etiquetas si se proporcionan
    df_display = doc_topic_matrix.copy()
    if topic_labels and len(topic_labels) == len(df_display.columns):
        df_display.columns = topic_labels

    # Crear heatmap
    fig = px.imshow(
        df_display,
        labels=dict(x="Topics", y="Documentos", color="Proporción"),
        x=df_display.columns,
        y=df_display.index,
        color_continuous_scale=cfg.heatmap_colorscale,
        aspect="auto",
        title=title
    )

    # Calcular altura dinámica
    height = max(400, len(df_display) * cfg.heatmap_row_height)

    fig.update_layout(
        height=height,
        template=cfg.theme,
        xaxis={'tickangle': -45}
    )

    # Agregar anotaciones con valores
    annotations = []
    for i, row_name in enumerate(df_display.index):
        for j, col_name in enumerate(df_display.columns):
            val = df_display.loc[row_name, col_name]
            if val > cfg.heatmap_threshold:
                annotations.append(
                    dict(
                        x=col_name,
                        y=row_name,
                        text=f"{val:.2f}",
                        showarrow=False,
                        font=dict(
                            color='white' if val > 0.5 else 'black',
                            size=10
                        )
                    )
                )

    fig.update_layout(annotations=annotations)

    logger.info(f"Heatmap de topics creado: {doc_topic_matrix.shape}")
    return fig


def create_topic_distribution_stacked(doc_topic_df: pd.DataFrame,
                                     title: str = "Distribución de Topics",
                                     topic_labels: Optional[List[str]] = None,
                                     config_path: Optional[str] = None) -> go.Figure:
    """
    Crea un gráfico de barras apiladas de distribución de topics.

    Args:
        doc_topic_df: DataFrame con docs en filas, topics en columnas
        title: Título del gráfico
        topic_labels: Etiquetas opcionales para topics
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly

    Example:
        >>> fig = create_topic_distribution_stacked(doc_topic_df)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    # Aplicar etiquetas
    df_display = doc_topic_df.copy()
    if topic_labels and len(topic_labels) == len(df_display.columns):
        df_display.columns = topic_labels

    fig = go.Figure()

    for col in df_display.columns:
        fig.add_trace(go.Bar(
            x=df_display.index,
            y=df_display[col],
            name=str(col)
        ))

    fig.update_layout(
        barmode='stack',
        title=title,
        xaxis_title="Documento",
        yaxis_title="Proporción",
        template=cfg.theme,
        height=cfg.default_height,
        xaxis={'tickangle': -45}
    )

    logger.info(f"Gráfico de barras apiladas creado")
    return fig


# ==================== GRÁFICOS DE HABILIDADES ====================

def create_skills_radar_chart(skill_scores: List[SkillScore],
                              title: str = "Perfil de Habilidades",
                              top_n: int = 10,
                              config_path: Optional[str] = None) -> go.Figure:
    """
    Crea un gráfico de radar de habilidades de un documento.

    Args:
        skill_scores: Lista de objetos SkillScore
        title: Título del gráfico
        top_n: Número de habilidades a mostrar
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly

    Example:
        >>> # Desde un DocumentSkillProfile
        >>> profile = skills_mapper.map_skills(processed_text)
        >>> fig = create_skills_radar_chart(profile.skill_scores)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    # Ordenar por score y tomar top N
    sorted_skills = sorted(skill_scores, key=lambda x: x.score, reverse=True)[:top_n]

    # Extraer datos
    categories = [skill.skill_name for skill in sorted_skills]
    scores = [skill.score for skill in sorted_skills]

    # Cerrar el polígono
    categories = categories + [categories[0]]
    scores = scores + [scores[0]]

    # Crear figura
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Habilidades',
        line_color=cfg.colors[0]
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0]
            )
        ),
        showlegend=False,
        title=title,
        template=cfg.theme,
        height=cfg.radar_height
    )

    logger.info(f"Gráfico de radar creado con {top_n} habilidades")
    return fig


def create_comparison_chart(profiles: List[DocumentSkillProfile],
                           title: str = "Comparación de Perfiles de Habilidades",
                           top_n_skills: int = 10,
                           config_path: Optional[str] = None) -> go.Figure:
    """
    Crea un gráfico de radar comparando múltiples documentos.

    Args:
        profiles: Lista de DocumentSkillProfile (máximo 4)
        title: Título del gráfico
        top_n_skills: Número de habilidades a comparar
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly

    Example:
        >>> profiles = [profile_a, profile_b, profile_c]
        >>> fig = create_comparison_chart(profiles)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    if len(profiles) > 4:
        logger.warning("Solo se compararán los primeros 4 perfiles")
        profiles = profiles[:4]

    # Obtener todas las habilidades únicas de los top skills
    all_skills = set()
    for profile in profiles:
        all_skills.update(profile.top_skills[:top_n_skills])

    # Si hay muchas skills, tomar solo las más frecuentes entre documentos
    if len(all_skills) > top_n_skills:
        skill_counts = {}
        for profile in profiles:
            for skill in profile.top_skills[:top_n_skills]:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        all_skills = sorted(skill_counts.keys(), key=skill_counts.get, reverse=True)[:top_n_skills]
    else:
        all_skills = sorted(all_skills)

    # Crear figura
    fig = go.Figure()

    for idx, profile in enumerate(profiles):
        # Crear diccionario de scores para este perfil
        skill_dict = {ss.skill_name: ss.score for ss in profile.skill_scores}

        # Obtener scores para las skills seleccionadas
        scores = [skill_dict.get(skill, 0.0) for skill in all_skills]

        # Cerrar el polígono
        categories = list(all_skills) + [list(all_skills)[0]]
        scores = scores + [scores[0]]

        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name=profile.document_id,
            line_color=cfg.colors[idx % len(cfg.colors)]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0]
            )
        ),
        showlegend=True,
        title=title,
        template=cfg.theme,
        height=cfg.radar_height
    )

    logger.info(f"Gráfico comparativo creado con {len(profiles)} perfiles")
    return fig


def create_skills_heatmap(profiles: List[DocumentSkillProfile],
                         title: str = "Matriz de Habilidades por Documento",
                         config_path: Optional[str] = None) -> go.Figure:
    """
    Crea un heatmap de habilidades × documentos.

    Args:
        profiles: Lista de DocumentSkillProfile
        title: Título del gráfico
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly

    Example:
        >>> fig = create_skills_heatmap([profile_a, profile_b])
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    # Construir matriz de skills × documentos
    all_skills = set()
    for profile in profiles:
        all_skills.update([ss.skill_name for ss in profile.skill_scores])

    all_skills = sorted(all_skills)

    # Crear DataFrame
    data = []
    for profile in profiles:
        skill_dict = {ss.skill_name: ss.score for ss in profile.skill_scores}
        row = [skill_dict.get(skill, 0.0) for skill in all_skills]
        data.append(row)

    df_matrix = pd.DataFrame(
        data,
        index=[p.document_id for p in profiles],
        columns=all_skills
    )

    # Crear heatmap
    fig = px.imshow(
        df_matrix,
        labels=dict(x="Habilidades", y="Documentos", color="Score"),
        x=df_matrix.columns,
        y=df_matrix.index,
        color_continuous_scale=cfg.heatmap_colorscale,
        aspect="auto",
        title=title
    )

    height = max(400, len(df_matrix) * cfg.heatmap_row_height)

    fig.update_layout(
        height=height,
        template=cfg.theme,
        xaxis={'tickangle': -45}
    )

    # Agregar anotaciones
    annotations = []
    for i, row in enumerate(df_matrix.index):
        for j, col in enumerate(df_matrix.columns):
            val = df_matrix.loc[row, col]
            if val > cfg.heatmap_threshold:
                annotations.append(
                    dict(
                        x=col,
                        y=row,
                        text=f"{val:.2f}",
                        showarrow=False,
                        font=dict(color='white' if val > 0.5 else 'black')
                    )
                )

    fig.update_layout(annotations=annotations)

    logger.info(f"Heatmap de habilidades creado: {df_matrix.shape}")
    return fig


# ==================== GRÁFICOS DE N-GRAMS Y CO-OCURRENCIAS ====================

def create_ngrams_comparison(ngrams_dict: Dict[str, pd.DataFrame],
                            n: int = 2,
                            top: int = 15,
                            title: Optional[str] = None,
                            config_path: Optional[str] = None) -> go.Figure:
    """
    Compara n-grams entre múltiples documentos.

    Args:
        ngrams_dict: Diccionario {doc_id: DataFrame de n-grams}
                    DataFrame debe tener columnas ['ngram', 'frequency']
        n: Tipo de n-gram (1, 2, 3)
        top: Número de n-grams a mostrar por documento
        title: Título del gráfico
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly

    Example:
        >>> ngrams = {
        ...     'Programa A': ngrams_df_a,
        ...     'Programa B': ngrams_df_b
        ... }
        >>> fig = create_ngrams_comparison(ngrams, n=2, top=10)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    if title is None:
        title = f"Top {top} {n}-grams por Documento"

    fig = go.Figure()

    for doc_id, df in ngrams_dict.items():
        # Filtrar por longitud de n-gram
        if 'ngram' in df.columns:
            df_n = df[df['ngram'].str.split().str.len() == n]
        else:
            df_n = df

        df_top = df_n.nlargest(top, 'frequency')

        fig.add_trace(go.Bar(
            x=df_top['ngram'] if 'ngram' in df_top.columns else df_top.index,
            y=df_top['frequency'],
            name=doc_id
        ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{n}-gram",
        yaxis_title="Frecuencia",
        barmode='group',
        template=cfg.theme,
        height=cfg.default_height,
        xaxis={'tickangle': -45}
    )

    logger.info(f"Gráfico de comparación de {n}-grams creado")
    return fig


def create_cooccurrence_network(df_cooccur: pd.DataFrame,
                                top_n: int = 30,
                                title: str = "Red de Co-ocurrencias",
                                config_path: Optional[str] = None) -> go.Figure:
    """
    Crea un gráfico de red de co-ocurrencias de términos.

    Args:
        df_cooccur: DataFrame con columnas ['term1', 'term2', 'frequency']
        top_n: Número de pares a mostrar
        title: Título del gráfico
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly

    Example:
        >>> # df_cooccur generado por análisis de co-ocurrencias
        >>> fig = create_cooccurrence_network(df_cooccur, top_n=20)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    cfg = PlotlyChartsConfig(config_path)

    # Validar columnas
    required_cols = ['term1', 'term2', 'frequency']
    if not all(col in df_cooccur.columns for col in required_cols):
        raise ValueError(f"DataFrame debe tener columnas: {required_cols}")

    df_top = df_cooccur.nlargest(top_n, 'frequency')

    # Crear nodos únicos
    nodes = list(set(df_top['term1'].tolist() + df_top['term2'].tolist()))
    node_indices = {node: idx for idx, node in enumerate(nodes)}

    # Posiciones circulares (simplificado)
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    node_x = np.cos(angles).tolist()
    node_y = np.sin(angles).tolist()

    # Crear edges
    edge_x = []
    edge_y = []

    for _, row in df_top.iterrows():
        idx1 = node_indices[row['term1']]
        idx2 = node_indices[row['term2']]

        edge_x.extend([node_x[idx1], node_x[idx2], None])
        edge_y.extend([node_y[idx1], node_y[idx2], None])

    # Crear figura
    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(color='lightgray', width=1),
        hoverinfo='none',
        showlegend=False
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=20,
            color=cfg.colors[0],
            line=dict(color='darkblue', width=2)
        ),
        text=nodes,
        textposition='top center',
        hoverinfo='text',
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        showlegend=False,
        template=cfg.theme,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=cfg.network_height
    )

    logger.info(f"Red de co-ocurrencias creada con {len(nodes)} nodos")
    return fig


# ==================== FUNCIONES ADICIONALES ====================

def create_skills_bar_comparison(profiles: List[DocumentSkillProfile],
                                 skill_name: str,
                                 title: Optional[str] = None,
                                 config_path: Optional[str] = None) -> go.Figure:
    """
    Compara el score de una habilidad específica entre documentos.

    Args:
        profiles: Lista de DocumentSkillProfile
        skill_name: Nombre de la habilidad a comparar
        title: Título del gráfico
        config_path: Ruta opcional a configuración

    Returns:
        Figura Plotly

    Example:
        >>> fig = create_skills_bar_comparison(profiles, "Machine Learning")
        >>> st.plotly_chart(fig)
    """
    cfg = PlotlyChartsConfig(config_path)

    if title is None:
        title = f"Comparación: {skill_name}"

    # Extraer scores
    doc_ids = []
    scores = []
    confidences = []

    for profile in profiles:
        doc_ids.append(profile.document_id)
        # Buscar el skill
        skill_score = next(
            (ss for ss in profile.skill_scores if ss.skill_name == skill_name),
            None
        )
        if skill_score:
            scores.append(skill_score.score)
            confidences.append(skill_score.confidence)
        else:
            scores.append(0.0)
            confidences.append(0.0)

    # Crear gráfico de barras
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=doc_ids,
        y=scores,
        name='Score',
        marker_color=cfg.colors[0],
        text=[f"{s:.2f}" for s in scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Documento",
        yaxis_title="Score",
        template=cfg.theme,
        height=cfg.bar_chart_height,
        yaxis=dict(range=[0, 1.0])
    )

    logger.info(f"Gráfico de comparación de habilidad '{skill_name}' creado")
    return fig


# ==================== WRAPPER CLASS ====================

class PlotlyCharts:
    """
    Clase wrapper para facilitar el uso de las funciones de gráficos en Streamlit.
    Proporciona una interfaz orientada a objetos para crear visualizaciones.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la clase de gráficos Plotly.

        Args:
            config_path: Ruta opcional al archivo de configuración
        """
        self.config_path = config_path

    def plot_top_terms(self, df: pd.DataFrame, top_n: int = 20,
                      title: str = "Términos Más Frecuentes") -> go.Figure:
        """Crea gráfico de barras de términos frecuentes"""
        return create_frequency_bar_chart(df, title, top_n, config_path=self.config_path)

    def plot_tfidf_scatter(self, df: pd.DataFrame,
                          title: str = "Análisis TF-IDF") -> go.Figure:
        """Crea gráfico de dispersión TF-IDF"""
        return create_tfidf_scatter(df, title, config_path=self.config_path)

    def plot_topic_heatmap(self, doc_topic_matrix: pd.DataFrame,
                          title: str = "Distribución de Topics",
                          topic_labels: Optional[List[str]] = None) -> go.Figure:
        """Crea heatmap de distribución de topics"""
        return create_topic_heatmap(doc_topic_matrix, title, topic_labels, config_path=self.config_path)

    def plot_topic_distribution(self, doc_topic_df: pd.DataFrame,
                               topic_labels: Optional[List[str]] = None,
                               title: str = "Distribución de Topics") -> go.Figure:
        """Crea gráfico de barras apiladas de topics"""
        return create_topic_distribution_stacked(doc_topic_df, title, topic_labels, config_path=self.config_path)

    def plot_skills_radar(self, skills_matrix: pd.DataFrame,
                         programs: List[str],
                         title: str = "Perfil de Habilidades") -> go.Figure:
        """
        Crea gráfico de radar comparando habilidades de múltiples programas.

        Args:
            skills_matrix: DataFrame con programas en filas, habilidades en columnas
            programs: Lista de nombres de programas
            title: Título del gráfico

        Returns:
            Figura Plotly
        """
        cfg = PlotlyChartsConfig(self.config_path)

        # Crear figura
        fig = go.Figure()

        # Cerrar el polígono
        categories = list(skills_matrix.columns) + [list(skills_matrix.columns)[0]]

        for idx, program in enumerate(programs):
            if program in skills_matrix.index:
                scores = skills_matrix.loc[program].tolist() + [skills_matrix.loc[program].iloc[0]]

                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=categories,
                    fill='toself',
                    name=program,
                    line_color=cfg.colors[idx % len(cfg.colors)]
                ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.0]
                )
            ),
            showlegend=True,
            title=title,
            template=cfg.theme,
            height=cfg.radar_height
        )

        return fig

    def plot_skills_heatmap(self, skills_matrix: pd.DataFrame,
                           title: str = "Matriz de Habilidades") -> go.Figure:
        """
        Crea heatmap de habilidades.

        Args:
            skills_matrix: DataFrame con datos en formato matriz
            title: Título del gráfico

        Returns:
            Figura Plotly
        """
        cfg = PlotlyChartsConfig(self.config_path)

        # Crear heatmap
        fig = px.imshow(
            skills_matrix,
            labels=dict(color="Score"),
            color_continuous_scale=cfg.heatmap_colorscale,
            aspect="auto",
            title=title
        )

        height = max(400, len(skills_matrix) * cfg.heatmap_row_height)

        fig.update_layout(
            height=height,
            template=cfg.theme,
            xaxis={'tickangle': -45}
        )

        # Agregar anotaciones
        annotations = []
        for i, row in enumerate(skills_matrix.index):
            for j, col in enumerate(skills_matrix.columns):
                val = skills_matrix.loc[row, col]
                if val > cfg.heatmap_threshold:
                    annotations.append(
                        dict(
                            x=j,
                            y=i,
                            text=f"{val:.2f}",
                            showarrow=False,
                            font=dict(color='white' if val > 0.5 else 'black', size=10)
                        )
                    )

        fig.update_layout(annotations=annotations)

        return fig

    def plot_ngrams_comparison(self, ngrams_dict: Dict[str, pd.DataFrame],
                              n: int = 2, top: int = 15) -> go.Figure:
        """Crea gráfico de comparación de n-grams"""
        return create_ngrams_comparison(ngrams_dict, n, top, config_path=self.config_path)
