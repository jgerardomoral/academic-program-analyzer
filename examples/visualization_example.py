"""
Ejemplo de uso de los módulos de visualización en Streamlit.

Este script demuestra cómo usar WordCloudGenerator y las funciones
de plotly_charts en una aplicación Streamlit.

Para ejecutar:
    streamlit run examples/visualization_example.py
"""
import streamlit as st
import pandas as pd
from datetime import datetime

# Importar módulos de visualización
from src.visualization import (
    WordCloudGenerator,
    create_wordcloud_from_text,
    create_frequency_bar_chart,
    create_tfidf_scatter,
    create_topic_heatmap,
    create_skills_radar_chart,
    create_comparison_chart
)

# Importar schemas para ejemplos
from src.utils.schemas import SkillScore, DocumentSkillProfile


def main():
    st.set_page_config(
        page_title="Visualizaciones - Análisis de Programas",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Ejemplos de Visualización")
    st.markdown("---")

    # Sidebar para selección de visualización
    viz_type = st.sidebar.selectbox(
        "Selecciona tipo de visualización:",
        [
            "WordClouds",
            "Gráficos de Frecuencia",
            "Gráficos de Topics",
            "Gráficos de Habilidades"
        ]
    )

    if viz_type == "WordClouds":
        show_wordcloud_examples()

    elif viz_type == "Gráficos de Frecuencia":
        show_frequency_charts()

    elif viz_type == "Gráficos de Topics":
        show_topic_charts()

    elif viz_type == "Gráficos de Habilidades":
        show_skills_charts()


def show_wordcloud_examples():
    """Muestra ejemplos de WordClouds"""
    st.header("🔤 WordClouds")

    # Ejemplo 1: WordCloud desde texto
    st.subheader("1. WordCloud desde texto")

    sample_text = """
    python programación desarrollo software datos análisis
    machine learning inteligencia artificial deep learning
    base de datos sql nosql cloud computing aws azure
    desarrollo web frontend backend javascript react
    """

    with st.expander("Ver texto de ejemplo"):
        st.text(sample_text)

    # Generar wordcloud
    generator = WordCloudGenerator()
    fig = generator.generate_from_text(sample_text)

    st.pyplot(fig)

    # Ejemplo 2: WordCloud desde DataFrame
    st.subheader("2. WordCloud desde DataFrame de frecuencias")

    df_freq = pd.DataFrame({
        'term': ['python', 'datos', 'análisis', 'machine', 'learning',
                 'desarrollo', 'software', 'web', 'cloud', 'database'],
        'tfidf': [0.95, 0.87, 0.82, 0.78, 0.75, 0.70, 0.68, 0.65, 0.60, 0.55],
        'frequency': [25, 20, 18, 15, 14, 12, 11, 10, 9, 8]
    })

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(df_freq, use_container_width=True)

    with col2:
        fig2 = generator.generate_from_frequencies(df_freq)
        st.pyplot(fig2)

    # Ejemplo 3: WordCloud con colormap personalizado
    st.subheader("3. WordCloud con estilos personalizados")

    colormap = st.selectbox(
        "Selecciona colormap:",
        ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Reds']
    )

    fig3 = generator.generate_from_text(
        sample_text,
        custom_colormap=colormap
    )
    st.pyplot(fig3)


def show_frequency_charts():
    """Muestra ejemplos de gráficos de frecuencia"""
    st.header("📊 Gráficos de Frecuencia")

    # Crear datos de ejemplo
    df_freq = pd.DataFrame({
        'term': [f'término_{i}' for i in range(1, 21)],
        'frequency': [25, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4, 3, 3, 2],
        'tfidf': [0.95, 0.87, 0.82, 0.78, 0.75, 0.70, 0.68, 0.65, 0.60, 0.55,
                 0.50, 0.45, 0.42, 0.38, 0.35, 0.30, 0.28, 0.25, 0.22, 0.20]
    })

    # Ejemplo 1: Gráfico de barras
    st.subheader("1. Gráfico de barras de términos frecuentes")

    top_n = st.slider("Número de términos:", 5, 20, 15)

    fig1 = create_frequency_bar_chart(
        df_freq,
        title=f"Top {top_n} Términos Más Frecuentes",
        top_n=top_n
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Ejemplo 2: Scatter plot TF-IDF
    st.subheader("2. Dispersión TF-IDF vs Frecuencia")

    fig2 = create_tfidf_scatter(df_freq)
    st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 Este gráfico ayuda a identificar términos importantes (alto TF-IDF) que no son necesariamente los más frecuentes.")


def show_topic_charts():
    """Muestra ejemplos de gráficos de topics"""
    st.header("🏷️ Gráficos de Topics")

    # Crear matriz documento-topic de ejemplo
    doc_topic_matrix = pd.DataFrame(
        {
            'Topic 1': [0.6, 0.1, 0.2, 0.3],
            'Topic 2': [0.2, 0.7, 0.1, 0.2],
            'Topic 3': [0.1, 0.1, 0.5, 0.3],
            'Topic 4': [0.1, 0.1, 0.2, 0.2]
        },
        index=['Programa A', 'Programa B', 'Programa C', 'Programa D']
    )

    # Ejemplo 1: Heatmap
    st.subheader("1. Heatmap de distribución de topics")

    topic_labels = ['Programación', 'Datos & IA', 'Desarrollo Web', 'Cloud & DevOps']

    fig1 = create_topic_heatmap(
        doc_topic_matrix,
        title="Distribución de Topics por Programa",
        topic_labels=topic_labels
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Ejemplo 2: Barras apiladas
    st.subheader("2. Barras apiladas de topics")

    from src.visualization.plotly_charts import create_topic_distribution_stacked

    fig2 = create_topic_distribution_stacked(
        doc_topic_matrix,
        topic_labels=topic_labels
    )
    st.plotly_chart(fig2, use_container_width=True)


def show_skills_charts():
    """Muestra ejemplos de gráficos de habilidades"""
    st.header("🎯 Gráficos de Habilidades")

    # Crear datos de ejemplo
    skill_scores_a = [
        SkillScore('s1', 'Python', 0.9, 0.95, ['python', 'programming'], []),
        SkillScore('s2', 'Machine Learning', 0.8, 0.85, ['ml', 'sklearn'], []),
        SkillScore('s3', 'Data Analysis', 0.85, 0.90, ['pandas', 'numpy'], []),
        SkillScore('s4', 'Web Development', 0.6, 0.70, ['html', 'css'], []),
        SkillScore('s5', 'Databases', 0.7, 0.75, ['sql', 'mongodb'], []),
        SkillScore('s6', 'Cloud Computing', 0.5, 0.60, ['aws', 'azure'], []),
        SkillScore('s7', 'DevOps', 0.4, 0.50, ['docker', 'ci/cd'], []),
    ]

    skill_scores_b = [
        SkillScore('s1', 'Python', 0.7, 0.80, ['python'], []),
        SkillScore('s2', 'Machine Learning', 0.6, 0.65, ['ml'], []),
        SkillScore('s3', 'Data Analysis', 0.65, 0.70, ['data'], []),
        SkillScore('s4', 'Web Development', 0.9, 0.95, ['react', 'node'], []),
        SkillScore('s5', 'Databases', 0.8, 0.85, ['sql', 'postgres'], []),
        SkillScore('s6', 'Cloud Computing', 0.7, 0.75, ['aws'], []),
        SkillScore('s7', 'DevOps', 0.75, 0.80, ['kubernetes', 'docker'], []),
    ]

    profile_a = DocumentSkillProfile(
        'Programa A',
        skill_scores_a,
        ['Python', 'Data Analysis', 'Machine Learning'],
        0.82,
        datetime.now()
    )

    profile_b = DocumentSkillProfile(
        'Programa B',
        skill_scores_b,
        ['Web Development', 'Databases', 'DevOps'],
        0.78,
        datetime.now()
    )

    # Ejemplo 1: Radar chart individual
    st.subheader("1. Radar Chart - Perfil Individual")

    fig1 = create_skills_radar_chart(
        skill_scores_a,
        title="Perfil de Habilidades - Programa A",
        top_n=7
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Ejemplo 2: Radar comparativo
    st.subheader("2. Radar Chart Comparativo")

    fig2 = create_comparison_chart(
        [profile_a, profile_b],
        title="Comparación de Perfiles de Habilidades",
        top_n_skills=7
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Ejemplo 3: Heatmap de habilidades
    st.subheader("3. Heatmap de Habilidades")

    from src.visualization.plotly_charts import create_skills_heatmap

    fig3 = create_skills_heatmap([profile_a, profile_b])
    st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
