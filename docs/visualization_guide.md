# Guía de Visualización

Esta guía explica cómo usar los módulos de visualización (`wordclouds.py` y `plotly_charts.py`) en tu aplicación Streamlit.

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [WordClouds](#wordclouds)
3. [Gráficos Plotly](#gráficos-plotly)
4. [Configuración](#configuración)
5. [Ejemplos en Streamlit](#ejemplos-en-streamlit)

---

## Instalación

Asegúrate de tener instaladas las dependencias necesarias:

```bash
pip install wordcloud matplotlib plotly pandas numpy pillow
```

---

## WordClouds

### Uso Básico

```python
from src.visualization import WordCloudGenerator

# Inicializar generador
generator = WordCloudGenerator()

# Generar desde texto
fig = generator.generate_from_text("python datos análisis machine learning")

# Mostrar en Streamlit
st.pyplot(fig)
```

### Generar desde DataFrame

```python
import pandas as pd

# DataFrame con frecuencias
df = pd.DataFrame({
    'term': ['python', 'datos', 'análisis'],
    'tfidf': [0.9, 0.7, 0.6],
    'frequency': [25, 20, 15]
})

# Generar wordcloud
fig = generator.generate_from_frequencies(df)
st.pyplot(fig)
```

### Personalización

```python
# Con colormap personalizado
fig = generator.generate_from_text(
    text,
    custom_colormap='plasma',
    custom_background='black'
)

# Con máscara de forma
mask = generator.load_mask_image("path/to/mask.png")
fig = generator.generate_from_text(text, mask_image=mask)
```

### WordClouds Comparativos

```python
texts = {
    'Programa A': "python machine learning datos",
    'Programa B': "java desarrollo software web"
}

fig = generator.create_comparative_wordclouds(texts, layout='horizontal')
st.pyplot(fig)
```

### Guardar WordCloud

```python
fig = generator.generate_from_text(text)
generator.save_wordcloud(fig, "outputs/wordcloud.png", dpi=300)
```

---

## Gráficos Plotly

### 1. Gráficos de Frecuencia

#### Gráfico de Barras

```python
from src.visualization import create_frequency_bar_chart

# DataFrame requerido: ['term', 'frequency', 'tfidf']
df_freq = pd.DataFrame({
    'term': ['python', 'datos', 'análisis'],
    'frequency': [25, 20, 15],
    'tfidf': [0.9, 0.7, 0.6]
})

fig = create_frequency_bar_chart(
    df_freq,
    title="Top Términos",
    top_n=20,
    value_column='tfidf'
)

st.plotly_chart(fig, use_container_width=True)
```

#### Scatter Plot TF-IDF

```python
from src.visualization import create_tfidf_scatter

fig = create_tfidf_scatter(df_freq)
st.plotly_chart(fig, use_container_width=True)
```

### 2. Gráficos de Topics

#### Heatmap de Topics

```python
from src.visualization import create_topic_heatmap

# DataFrame: documentos × topics
doc_topic_matrix = pd.DataFrame(
    [[0.6, 0.2, 0.1, 0.1],
     [0.1, 0.7, 0.1, 0.1]],
    index=['Doc A', 'Doc B'],
    columns=['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4']
)

fig = create_topic_heatmap(
    doc_topic_matrix,
    topic_labels=['Programación', 'Datos', 'Web', 'Cloud']
)

st.plotly_chart(fig, use_container_width=True)
```

#### Barras Apiladas de Topics

```python
from src.visualization import create_topic_distribution_stacked

fig = create_topic_distribution_stacked(
    doc_topic_matrix,
    topic_labels=['Programación', 'Datos', 'Web', 'Cloud']
)

st.plotly_chart(fig, use_container_width=True)
```

### 3. Gráficos de Habilidades

#### Radar Chart Individual

```python
from src.visualization import create_skills_radar_chart

# Desde un DocumentSkillProfile
profile = skills_mapper.map_skills(processed_text)

fig = create_skills_radar_chart(
    profile.skill_scores,
    title="Perfil de Habilidades",
    top_n=10
)

st.plotly_chart(fig, use_container_width=True)
```

#### Radar Chart Comparativo

```python
from src.visualization import create_comparison_chart

# Comparar múltiples perfiles (máximo 4)
profiles = [profile_a, profile_b, profile_c]

fig = create_comparison_chart(
    profiles,
    title="Comparación de Programas",
    top_n_skills=10
)

st.plotly_chart(fig, use_container_width=True)
```

#### Heatmap de Habilidades

```python
from src.visualization import create_skills_heatmap

fig = create_skills_heatmap(
    profiles,
    title="Matriz de Habilidades"
)

st.plotly_chart(fig, use_container_width=True)
```

#### Comparación de Habilidad Específica

```python
from src.visualization import create_skills_bar_comparison

fig = create_skills_bar_comparison(
    profiles,
    skill_name="Machine Learning",
    title="Comparación: Machine Learning"
)

st.plotly_chart(fig, use_container_width=True)
```

### 4. Gráficos de N-grams

#### Comparación de N-grams

```python
from src.visualization import create_ngrams_comparison

# Diccionario de DataFrames de n-grams
ngrams_dict = {
    'Programa A': ngrams_df_a,  # Columnas: ['ngram', 'frequency']
    'Programa B': ngrams_df_b
}

fig = create_ngrams_comparison(
    ngrams_dict,
    n=2,  # bigramas
    top=15
)

st.plotly_chart(fig, use_container_width=True)
```

### 5. Red de Co-ocurrencias

```python
from src.visualization import create_cooccurrence_network

# DataFrame con columnas: ['term1', 'term2', 'frequency']
df_cooccur = pd.DataFrame({
    'term1': ['python', 'python', 'machine'],
    'term2': ['datos', 'machine', 'learning'],
    'frequency': [10, 8, 12]
})

fig = create_cooccurrence_network(
    df_cooccur,
    top_n=30
)

st.plotly_chart(fig, use_container_width=True)
```

---

## Configuración

### Archivo config.yaml

Todas las visualizaciones usan configuración centralizada en `config.yaml`:

```yaml
visualization:
  # WordCloud settings
  wordcloud:
    width: 800
    height: 400
    background_color: "white"
    colormap: "viridis"
    relative_scaling: 0.5
    min_font_size: 10
    max_words: 200
    prefer_horizontal: 0.7

  # Plotly chart settings
  plotly:
    theme: "plotly_white"
    color_scale: "Viridis"
    default_height: 500
    bar_chart_height: 400
    heatmap_row_height: 30
    radar_height: 500
    network_height: 600
    colors:
      primary: "#636EFA"
      secondary: "#EF553B"
      tertiary: "#00CC96"
      quaternary: "#AB63FA"
    heatmap:
      colorscale: "RdYlGn"
      show_values_threshold: 0.1
```

### Uso Programático de Config

```python
from src.visualization.wordclouds import WordCloudGenerator

# Usar configuración personalizada
generator = WordCloudGenerator(config_path="mi_config.yaml")

# O usar configuración por defecto
generator = WordCloudGenerator()
```

---

## Ejemplos en Streamlit

### Ejemplo Completo: Página de Análisis

```python
import streamlit as st
from src.visualization import (
    create_wordcloud_from_dataframe,
    create_frequency_bar_chart,
    create_skills_radar_chart
)

def show_analysis_page(analysis_result):
    """Muestra página completa de análisis"""

    st.title("📊 Análisis de Programa")

    # Tabs para organizar visualizaciones
    tab1, tab2, tab3 = st.tabs(["Frecuencias", "Habilidades", "Topics"])

    with tab1:
        st.header("Análisis de Frecuencias")

        col1, col2 = st.columns(2)

        with col1:
            # WordCloud
            st.subheader("Nube de Palabras")
            fig_wc = create_wordcloud_from_dataframe(
                analysis_result.term_frequencies
            )
            st.pyplot(fig_wc)

        with col2:
            # Gráfico de barras
            st.subheader("Top Términos")
            fig_bar = create_frequency_bar_chart(
                analysis_result.term_frequencies,
                top_n=15
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.header("Perfil de Habilidades")

        # Radar chart
        fig_radar = create_skills_radar_chart(
            analysis_result.skill_profile.skill_scores,
            top_n=10
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Tabla de habilidades
        st.subheader("Detalle de Habilidades")
        st.dataframe(
            analysis_result.skill_profile.to_dataframe(),
            use_container_width=True
        )

    with tab3:
        st.header("Modelado de Topics")

        # Heatmap de topics
        if hasattr(analysis_result, 'topic_model'):
            fig_topics = create_topic_heatmap(
                analysis_result.topic_model.document_topic_matrix
            )
            st.plotly_chart(fig_topics, use_container_width=True)
```

### Ejemplo: Comparación de Múltiples Documentos

```python
import streamlit as st
from src.visualization import create_comparison_chart

def compare_programs(profiles):
    """Compara múltiples programas"""

    st.title("⚖️ Comparación de Programas")

    # Selector de programas
    selected = st.multiselect(
        "Selecciona programas a comparar (máx. 4):",
        [p.document_id for p in profiles],
        default=[p.document_id for p in profiles[:2]]
    )

    # Filtrar perfiles seleccionados
    selected_profiles = [
        p for p in profiles
        if p.document_id in selected
    ]

    if len(selected_profiles) >= 2:
        # Radar comparativo
        fig = create_comparison_chart(
            selected_profiles,
            title=f"Comparación: {', '.join(selected)}"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Comparación por habilidad específica
        st.subheader("Comparación por Habilidad")

        all_skills = set()
        for p in selected_profiles:
            all_skills.update(p.top_skills)

        skill = st.selectbox("Selecciona habilidad:", sorted(all_skills))

        from src.visualization import create_skills_bar_comparison
        fig_bar = create_skills_bar_comparison(
            selected_profiles,
            skill_name=skill
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ Selecciona al menos 2 programas para comparar")
```

### Ejemplo: Exportar Visualizaciones

```python
import streamlit as st
from src.visualization import WordCloudGenerator

def export_visualizations(analysis_result):
    """Exporta visualizaciones a archivos"""

    st.sidebar.header("Exportar Visualizaciones")

    if st.sidebar.button("💾 Exportar WordCloud"):
        generator = WordCloudGenerator()
        fig = generator.generate_from_frequencies(
            analysis_result.term_frequencies
        )

        output_path = f"outputs/{analysis_result.document_id}_wordcloud.png"
        generator.save_wordcloud(fig, output_path, dpi=300)

        st.sidebar.success(f"✅ Guardado en: {output_path}")

    if st.sidebar.button("💾 Exportar Gráficos"):
        # Los gráficos Plotly se pueden exportar con:
        fig.write_image("outputs/chart.png")  # Requiere kaleido
        fig.write_html("outputs/chart.html")  # HTML interactivo

        st.sidebar.success("✅ Gráficos exportados")
```

---

## Referencia Rápida

### Formatos de DataFrames Requeridos

| Función | Columnas Requeridas | Descripción |
|---------|---------------------|-------------|
| `create_frequency_bar_chart` | `['term', 'frequency', 'tfidf']` | Análisis de frecuencias |
| `create_tfidf_scatter` | `['term', 'frequency', 'tfidf']` | Dispersión TF-IDF |
| `create_topic_heatmap` | Docs × Topics | Matriz de distribución |
| `create_ngrams_comparison` | `['ngram', 'frequency']` | N-gramas |
| `create_cooccurrence_network` | `['term1', 'term2', 'frequency']` | Co-ocurrencias |

### Tipos de Entrada para WordClouds

| Método | Entrada | Ejemplo |
|--------|---------|---------|
| `generate_from_text` | String | `"python datos análisis"` |
| `generate_from_frequencies` | DataFrame o Dict | `{'python': 0.9, 'datos': 0.7}` |

### Colormaps Disponibles

- **Secuenciales**: `viridis`, `plasma`, `inferno`, `magma`, `cividis`
- **Divergentes**: `RdYlGn`, `RdBu`, `coolwarm`
- **Categóricos**: `Blues`, `Reds`, `Greens`, `Purples`

---

## Solución de Problemas

### Error: "DataFrame debe tener columna 'term'"

**Solución**: Asegúrate de que tu DataFrame tenga las columnas requeridas:

```python
# Renombrar columnas si es necesario
df = df.rename(columns={'palabra': 'term', 'score': 'tfidf'})
```

### WordCloud en blanco o vacía

**Solución**: Verifica que el texto no esté vacío y contenga palabras válidas:

```python
if text.strip():
    fig = generator.generate_from_text(text)
else:
    st.warning("El texto está vacío")
```

### Gráficos Plotly no se muestran

**Solución**: Usa siempre `st.plotly_chart()` para Plotly (no `st.pyplot()`):

```python
# ✅ Correcto
st.plotly_chart(fig, use_container_width=True)

# ❌ Incorrecto
st.pyplot(fig)  # Solo para matplotlib
```

---

## Mejores Prácticas

1. **Usa `use_container_width=True`** en `st.plotly_chart()` para gráficos responsivos

2. **Organiza con tabs y columnas** para mejorar la experiencia de usuario:
   ```python
   tab1, tab2, tab3 = st.tabs(["Frecuencias", "Skills", "Topics"])
   ```

3. **Cachea las visualizaciones** si son costosas:
   ```python
   @st.cache_data
   def generate_wordcloud(df):
       return create_wordcloud_from_dataframe(df)
   ```

4. **Proporciona controles interactivos**:
   ```python
   top_n = st.slider("Términos a mostrar:", 5, 50, 20)
   colormap = st.selectbox("Color:", ['viridis', 'plasma'])
   ```

5. **Maneja errores gracefully**:
   ```python
   try:
       fig = create_frequency_bar_chart(df)
       st.plotly_chart(fig)
   except ValueError as e:
       st.error(f"Error al generar gráfico: {e}")
   ```

---

## Recursos Adicionales

- **Documentación Plotly**: https://plotly.com/python/
- **Documentación WordCloud**: https://amueller.github.io/word_cloud/
- **Streamlit Docs**: https://docs.streamlit.io/

---

**Última actualización**: 2025-10-03
