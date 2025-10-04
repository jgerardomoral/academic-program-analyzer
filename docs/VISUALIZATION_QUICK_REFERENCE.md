# Visualización - Referencia Rápida

## 🎯 Importaciones

```python
# WordClouds
from src.visualization import WordCloudGenerator, create_wordcloud_from_dataframe

# Gráficos Plotly
from src.visualization import (
    create_frequency_bar_chart,
    create_skills_radar_chart,
    create_comparison_chart,
    create_topic_heatmap
)
```

## 📊 Uso Rápido

### WordCloud

```python
# Desde texto
generator = WordCloudGenerator()
fig = generator.generate_from_text("python datos análisis")
st.pyplot(fig)

# Desde DataFrame
fig = create_wordcloud_from_dataframe(df_freq)
st.pyplot(fig)
```

### Gráficos de Frecuencia

```python
# Barras
fig = create_frequency_bar_chart(df, top_n=20)
st.plotly_chart(fig, use_container_width=True)

# Scatter TF-IDF
from src.visualization import create_tfidf_scatter
fig = create_tfidf_scatter(df)
st.plotly_chart(fig, use_container_width=True)
```

### Gráficos de Habilidades

```python
# Radar individual
fig = create_skills_radar_chart(profile.skill_scores, top_n=10)
st.plotly_chart(fig, use_container_width=True)

# Comparación (2-4 perfiles)
fig = create_comparison_chart([profile_a, profile_b])
st.plotly_chart(fig, use_container_width=True)
```

### Gráficos de Topics

```python
# Heatmap
fig = create_topic_heatmap(doc_topic_matrix)
st.plotly_chart(fig, use_container_width=True)
```

## 📋 Formatos de DataFrames

```python
# Frecuencias
df_freq = pd.DataFrame({
    'term': ['python', 'datos'],
    'frequency': [25, 20],
    'tfidf': [0.9, 0.7]
})

# Topics (docs × topics)
doc_topic_matrix = pd.DataFrame(
    [[0.6, 0.2, 0.1, 0.1]],
    index=['Doc A'],
    columns=['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4']
)

# N-grams
df_ngrams = pd.DataFrame({
    'ngram': ['machine learning', 'data science'],
    'frequency': [10, 8]
})
```

## ⚙️ Configuración (config.yaml)

```yaml
visualization:
  wordcloud:
    width: 800
    height: 400
    colormap: "viridis"

  plotly:
    theme: "plotly_white"
    color_scale: "Viridis"
    default_height: 500
```

## 🎨 Personalización

```python
# WordCloud con colormap
fig = generator.generate_from_text(
    text,
    custom_colormap='plasma',
    custom_background='black'
)

# Gráfico con config custom
fig = create_frequency_bar_chart(
    df,
    config_path="mi_config.yaml"
)
```

## 💾 Exportar

```python
# WordCloud a archivo
generator.save_wordcloud(fig, "output.png", dpi=300)

# Plotly
fig.write_html("chart.html")
fig.write_image("chart.png")  # Requiere kaleido
```

## 🔗 Links

- Guía completa: `docs/visualization_guide.md`
- Ejemplos: `examples/visualization_example.py`
- Reporte: `VISUALIZATION_IMPLEMENTATION_REPORT.md`
