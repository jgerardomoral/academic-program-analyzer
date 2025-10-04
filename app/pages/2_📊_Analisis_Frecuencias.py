"""
Página de análisis de frecuencias y términos.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import io

# Setup path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

# Imports
from utils.session_manager import init_session_state
from utils.cache_manager import cached_frequency_analysis
from src.visualization.plotly_charts import (
    create_frequency_bar_chart,
    create_tfidf_scatter,
    create_cooccurrence_network
)
from src.visualization.wordclouds import WordCloudGenerator
from src.analysis.frequency import FrequencyAnalyzer

# Inicializar sesión
init_session_state()

st.title("📊 Análisis de Frecuencias")

st.markdown("""
Analiza la frecuencia de términos, n-gramas y co-ocurrencias en tus documentos.
Los resultados incluyen scores TF-IDF, nubes de palabras y visualizaciones interactivas.
""")

# Verificar que hay documentos procesados
if not st.session_state.processed_texts:
    st.warning("⚠️ No hay documentos procesados. Ve a **📁 Subir PDFs** primero.")
    st.stop()

# ==================== SIDEBAR: CONFIGURACIÓN ====================
with st.sidebar:
    st.header("🎛️ Configuración")

    # Selección de documentos
    available_docs = list(st.session_state.processed_texts.keys())
    selected_docs = st.multiselect(
        "Documentos a analizar",
        options=available_docs,
        default=available_docs[:min(3, len(available_docs))],
        help="Selecciona uno o más documentos para analizar"
    )

    if not selected_docs:
        st.error("⚠️ Debes seleccionar al menos un documento")
        st.stop()

    st.markdown("---")

    # Parámetros de visualización
    st.subheader("📈 Visualización")

    top_n = st.slider(
        "Top N términos",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Número de términos principales a mostrar"
    )

    min_freq = st.number_input(
        "Frecuencia mínima",
        min_value=1,
        max_value=50,
        value=2,
        step=1,
        help="Frecuencia mínima para incluir términos en tablas"
    )

    st.markdown("---")

    # Análisis comparativo
    compare_mode = st.checkbox(
        "Modo comparación",
        value=len(selected_docs) > 1,
        help="Compara términos entre documentos",
        disabled=len(selected_docs) == 1
    )

    # Botón de reanalizar
    if st.button("🔄 Reanalizar", use_container_width=True):
        for doc in selected_docs:
            if doc in st.session_state.frequency_analyses:
                del st.session_state.frequency_analyses[doc]
        st.rerun()

# ==================== EJECUTAR ANÁLISIS ====================
st.markdown("---")
st.subheader("🔄 Procesando Análisis...")

progress_bar = st.progress(0)
status_text = st.empty()
analyses = {}

for idx, doc_name in enumerate(selected_docs):
    status_text.text(f"Analizando {doc_name}... ({idx+1}/{len(selected_docs)})")

    processed = st.session_state.processed_texts[doc_name]

    # Usar caché para análisis
    processed_dict = {
        'filename': processed.filename,
        'clean_text': processed.clean_text,
        'tokens': processed.tokens,
        'lemmas': processed.lemmas,
        'pos_tags': processed.pos_tags,
        'entities': processed.entities,
        'metadata': processed.metadata,
        'processing_date': processed.processing_date.isoformat()
    }

    analysis = cached_frequency_analysis(processed_dict)
    analyses[doc_name] = analysis

    # Guardar en session state
    st.session_state.frequency_analyses[doc_name] = analysis

    progress_bar.progress((idx + 1) / len(selected_docs))

progress_bar.empty()
status_text.empty()
st.success(f"✅ {len(selected_docs)} documento(s) analizado(s) correctamente")

# ==================== TABS PRINCIPALES ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Top Términos",
    "📈 TF-IDF",
    "🔤 N-gramas",
    "☁️ Nubes de Palabras"
])

# ==================== TAB 1: TOP TÉRMINOS ====================
with tab1:
    st.subheader("Términos Más Frecuentes")

    if compare_mode and len(selected_docs) > 1:
        st.info("📊 Mostrando comparación entre documentos")

        # Gráficos lado a lado
        cols = st.columns(min(len(selected_docs), 2))

        for idx, (doc_name, analysis) in enumerate(analyses.items()):
            with cols[idx % 2]:
                st.markdown(f"**{doc_name}**")

                # Crear gráfico de barras
                try:
                    fig = create_frequency_bar_chart(
                        analysis.term_frequencies,
                        title=f"Top {top_n} Términos",
                        top_n=top_n,
                        value_column='tfidf'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generando gráfico: {str(e)}")

                # Métricas
                col1, col2 = st.columns(2)
                col1.metric("Vocabulario", analysis.vocabulary_size)
                col2.metric("Top términos", len(analysis.top_terms))
    else:
        # Un solo documento
        doc_name = selected_docs[0]
        analysis = analyses[doc_name]

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Vocabulario", analysis.vocabulary_size)
        col2.metric("Términos únicos", len(analysis.term_frequencies))
        col3.metric("Unigrams", len(analysis.ngrams.get(1, [])))
        col4.metric("Bigrams", len(analysis.ngrams.get(2, [])))

        st.markdown("---")

        # Gráfico de barras principal
        try:
            fig = create_frequency_bar_chart(
                analysis.term_frequencies,
                title=f"Top {top_n} Términos - {doc_name}",
                top_n=top_n,
                value_column='tfidf'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generando gráfico: {str(e)}")

        # Tabla de términos
        st.markdown("### 📋 Tabla de Frecuencias")

        # Filtrar por frecuencia mínima
        df_filtered = analysis.term_frequencies[
            analysis.term_frequencies['frequency'] >= min_freq
        ].head(50)

        st.dataframe(
            df_filtered,
            use_container_width=True,
            height=400
        )

        # Botón de descarga
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"frecuencias_{doc_name}.csv",
            mime="text/csv"
        )

# ==================== TAB 2: TF-IDF SCATTER ====================
with tab2:
    st.subheader("Análisis TF-IDF: Frecuencia vs. Importancia")

    st.info("""
    **TF-IDF (Term Frequency - Inverse Document Frequency)** mide la importancia de un término
    en un documento en relación con el corpus completo. Valores altos indican términos
    distintivos del documento.
    """)

    # Selector de documento si hay múltiples
    if len(selected_docs) > 1:
        selected_doc_tfidf = st.selectbox(
            "Selecciona documento para análisis TF-IDF",
            options=selected_docs,
            key="tfidf_selector"
        )
    else:
        selected_doc_tfidf = selected_docs[0]

    analysis = analyses[selected_doc_tfidf]

    # Gráfico de dispersión
    try:
        fig = create_tfidf_scatter(
            analysis.term_frequencies.head(100),  # Limitar a top 100 para legibilidad
            title=f"TF-IDF vs Frecuencia - {selected_doc_tfidf}"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generando gráfico: {str(e)}")

    # Estadísticas
    st.markdown("### 📊 Estadísticas TF-IDF")

    col1, col2, col3 = st.columns(3)
    col1.metric("TF-IDF promedio", f"{analysis.term_frequencies['tfidf'].mean():.4f}")
    col2.metric("TF-IDF máximo", f"{analysis.term_frequencies['tfidf'].max():.4f}")
    col3.metric("TF-IDF mínimo", f"{analysis.term_frequencies['tfidf'].min():.4f}")

# ==================== TAB 3: N-GRAMAS ====================
with tab3:
    st.subheader("Análisis de N-gramas")

    st.info("""
    **N-gramas** son secuencias de N palabras consecutivas. Ayudan a identificar
    frases y conceptos compuestos que aparecen frecuentemente en el texto.
    """)

    # Selector de tipo de n-grama
    ngram_type = st.radio(
        "Tipo de N-grama",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Unigramas (palabras simples)",
            2: "Bigramas (pares de palabras)",
            3: "Trigramas (triples de palabras)"
        }[x],
        horizontal=True
    )

    # Selector de documento si hay múltiples
    if len(selected_docs) > 1:
        selected_doc_ngram = st.selectbox(
            "Selecciona documento",
            options=selected_docs,
            key="ngram_selector"
        )
    else:
        selected_doc_ngram = selected_docs[0]

    analysis = analyses[selected_doc_ngram]

    # Mostrar n-gramas
    if ngram_type in analysis.ngrams:
        df_ngrams = analysis.ngrams[ngram_type].head(top_n)

        if not df_ngrams.empty:
            # Gráfico de barras
            col1, col2 = st.columns([2, 1])

            with col1:
                try:
                    # Preparar DataFrame para gráfico (necesita columna 'term' y 'tfidf' o 'frequency')
                    df_chart = df_ngrams.copy()
                    df_chart = df_chart.rename(columns={'ngram': 'term'})

                    # Si no tiene tfidf, usar frequency
                    if 'tfidf' not in df_chart.columns:
                        df_chart['tfidf'] = df_chart['frequency']

                    fig = create_frequency_bar_chart(
                        df_chart,
                        title=f"Top {top_n} {ngram_type}-gramas",
                        top_n=top_n,
                        value_column='frequency'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generando gráfico: {str(e)}")

            with col2:
                st.metric(f"Total {ngram_type}-gramas", len(analysis.ngrams[ngram_type]))
                st.metric("Mostrando", len(df_ngrams))

            # Tabla
            st.markdown(f"### 📋 Tabla de {ngram_type}-gramas")
            st.dataframe(df_ngrams, use_container_width=True, height=400)

            # Descarga
            csv_ngrams = df_ngrams.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Descargar {ngram_type}-gramas CSV",
                data=csv_ngrams,
                file_name=f"{ngram_type}grams_{selected_doc_ngram}.csv",
                mime="text/csv"
            )
        else:
            st.warning(f"No se encontraron {ngram_type}-gramas en este documento")
    else:
        st.warning(f"No hay datos de {ngram_type}-gramas disponibles")

# ==================== TAB 4: NUBES DE PALABRAS ====================
with tab4:
    st.subheader("Nubes de Palabras")

    st.info("""
    Las **nubes de palabras** visualizan la frecuencia de términos mediante el tamaño de la fuente.
    Términos más grandes aparecen con mayor frecuencia en el documento.
    """)

    # Configuración de wordcloud
    col1, col2 = st.columns(2)

    with col1:
        colormap = st.selectbox(
            "Esquema de colores",
            options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Greens', 'Reds'],
            index=0
        )

    with col2:
        bg_color = st.selectbox(
            "Color de fondo",
            options=['white', 'black'],
            index=0
        )

    st.markdown("---")

    # Generar wordclouds
    if compare_mode and len(selected_docs) > 1:
        st.markdown("### Comparación de Nubes de Palabras")

        # Máximo 4 documentos en grid 2x2
        docs_to_show = selected_docs[:4]
        cols = st.columns(2)

        for idx, doc_name in enumerate(docs_to_show):
            processed = st.session_state.processed_texts[doc_name]

            with cols[idx % 2]:
                st.markdown(f"**{doc_name}**")

                with st.spinner("Generando nube de palabras..."):
                    try:
                        wordcloud_gen = WordCloudGenerator()
                        wc_fig = wordcloud_gen.generate_from_text(
                            processed.clean_text,
                            custom_colormap=colormap,
                            custom_background=bg_color
                        )
                        st.pyplot(wc_fig)
                    except Exception as e:
                        st.error(f"Error generando nube: {str(e)}")
    else:
        # Un solo documento
        doc_name = selected_docs[0]
        processed = st.session_state.processed_texts[doc_name]
        analysis = analyses[doc_name]

        tab_wc1, tab_wc2 = st.tabs(["Desde Texto", "Desde TF-IDF"])

        with tab_wc1:
            st.markdown("#### Nube de palabras desde texto completo")

            with st.spinner("Generando nube de palabras desde texto..."):
                try:
                    wordcloud_gen = WordCloudGenerator()
                    wc_fig = wordcloud_gen.generate_from_text(
                        processed.clean_text,
                        custom_colormap=colormap,
                        custom_background=bg_color
                    )
                    st.pyplot(wc_fig)

                    # Botón de descarga
                    buf = io.BytesIO()
                    wc_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)

                    st.download_button(
                        label="💾 Descargar PNG",
                        data=buf,
                        file_name=f"wordcloud_text_{doc_name}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error generando nube desde texto: {str(e)}")

        with tab_wc2:
            st.markdown("#### Nube de palabras ponderada por TF-IDF")

            with st.spinner("Generando nube de palabras desde TF-IDF..."):
                try:
                    wordcloud_gen = WordCloudGenerator()

                    # Usar top términos por TF-IDF
                    df_top_tfidf = analysis.term_frequencies.nlargest(200, 'tfidf')

                    wc_fig = wordcloud_gen.generate_from_frequencies(
                        df_top_tfidf,
                        custom_colormap=colormap,
                        custom_background=bg_color
                    )
                    st.pyplot(wc_fig)

                    # Botón de descarga
                    buf = io.BytesIO()
                    wc_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                    buf.seek(0)

                    st.download_button(
                        label="💾 Descargar PNG",
                        data=buf,
                        file_name=f"wordcloud_tfidf_{doc_name}.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error generando nube desde TF-IDF: {str(e)}")

# ==================== SECCIÓN DE CO-OCURRENCIAS ====================
st.markdown("---")
st.subheader("🔗 Análisis de Co-ocurrencias")

with st.expander("ℹ️ ¿Qué son las co-ocurrencias?", expanded=False):
    st.markdown("""
    Las **co-ocurrencias** muestran qué términos aparecen frecuentemente cerca uno del otro
    en el texto. Esto ayuda a identificar:
    - Conceptos relacionados
    - Frases compuestas
    - Patrones temáticos

    La "ventana" define la distancia máxima entre palabras para considerarlas relacionadas.
    """)

# Selector de documento para co-ocurrencias
if len(selected_docs) > 1:
    selected_doc_cooccur = st.selectbox(
        "Selecciona documento para análisis de co-ocurrencias",
        options=selected_docs,
        key="cooccur_selector"
    )
else:
    selected_doc_cooccur = selected_docs[0]

processed = st.session_state.processed_texts[selected_doc_cooccur]

# Parámetros
col1, col2 = st.columns(2)

with col1:
    window_size = st.slider(
        "Tamaño de ventana",
        min_value=2,
        max_value=10,
        value=5,
        help="Distancia máxima entre palabras para considerarlas relacionadas"
    )

with col2:
    top_cooccur = st.slider(
        "Top co-ocurrencias",
        min_value=10,
        max_value=50,
        value=30
    )

# Calcular co-ocurrencias
if st.button("🔍 Calcular Co-ocurrencias", type="primary"):
    with st.spinner("Calculando co-ocurrencias..."):
        try:
            analyzer = FrequencyAnalyzer()
            df_cooccur = analyzer.get_cooccurrences(processed, window_size=window_size)

            if not df_cooccur.empty:
                # Tabla
                st.markdown("### 📋 Tabla de Co-ocurrencias")
                st.dataframe(
                    df_cooccur.head(top_cooccur),
                    use_container_width=True,
                    height=300
                )

                # Gráfico de red
                st.markdown("### 🕸️ Red de Co-ocurrencias")

                try:
                    fig = create_cooccurrence_network(
                        df_cooccur,
                        top_n=min(30, top_cooccur),
                        title="Red de Términos Co-ocurrentes"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo generar red de co-ocurrencias: {str(e)}")

                # Descarga
                csv_cooccur = df_cooccur.head(top_cooccur).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Co-ocurrencias CSV",
                    data=csv_cooccur,
                    file_name=f"coocurrencias_{selected_doc_cooccur}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No se encontraron co-ocurrencias significativas")
        except Exception as e:
            st.error(f"Error calculando co-ocurrencias: {str(e)}")

# ==================== COMPARACIÓN ENTRE DOCUMENTOS ====================
if compare_mode and len(selected_docs) > 1:
    st.markdown("---")
    st.subheader("📊 Comparación entre Documentos")

    st.info("""
    Compara el vocabulario y términos principales entre los documentos seleccionados
    para identificar similitudes y diferencias.
    """)

    # Análisis de términos únicos vs compartidos
    all_terms = set()
    doc_terms = {}

    for doc_name in selected_docs:
        terms = set(analyses[doc_name].term_frequencies['term'])
        doc_terms[doc_name] = terms
        all_terms.update(terms)

    # Términos compartidos por todos
    shared_terms = set.intersection(*doc_terms.values())

    # Métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Términos totales únicos", len(all_terms))
    col2.metric("Términos compartidos", len(shared_terms))
    col3.metric("% Compartidos", f"{len(shared_terms)/len(all_terms)*100:.1f}%")

    # Mostrar términos compartidos
    if shared_terms:
        with st.expander(f"Ver {len(shared_terms)} términos compartidos"):
            shared_sorted = sorted(list(shared_terms))
            st.write(", ".join(shared_sorted))

    # Términos únicos por documento
    st.markdown("### 📑 Términos Únicos por Documento")

    for doc_name in selected_docs:
        unique = doc_terms[doc_name] - set.union(*[doc_terms[d] for d in selected_docs if d != doc_name])

        if unique:
            with st.expander(f"📄 {doc_name}: {len(unique)} términos únicos"):
                unique_sorted = sorted(list(unique))[:50]  # Primeros 50
                st.write(", ".join(unique_sorted))
                if len(unique) > 50:
                    st.caption(f"... y {len(unique) - 50} más")

# ==================== SIDEBAR: TIPS E INFORMACIÓN ====================
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 Interpretación")
    st.info("""
    **TF-IDF**: Mide la importancia relativa del término en el documento vs el corpus.

    **Frecuencia**: Conteo simple de apariciones del término.

    **N-gramas**: Frases de N palabras que aparecen juntas frecuentemente.

    **Co-ocurrencias**: Palabras que aparecen cerca una de otra en el texto.
    """)

    # Estadísticas de análisis
    st.markdown("---")
    st.markdown("### 📈 Estadísticas")
    st.metric("Documentos analizados", len(analyses))

    if analyses:
        total_vocab = sum(a.vocabulary_size for a in analyses.values())
        avg_vocab = total_vocab / len(analyses)
        st.metric("Vocabulario promedio", f"{avg_vocab:.0f}")
