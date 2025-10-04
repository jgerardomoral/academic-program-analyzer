"""
Página de búsqueda de términos específicos en documentos procesados.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

# Setup path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

# Imports
from utils.session_manager import init_session_state
from src.visualization.plotly_charts import create_frequency_bar_chart
import plotly.graph_objects as go

# Inicializar
init_session_state()

st.title("🔍 Búsqueda de Términos Específicos")

st.markdown("""
Busca términos específicos en los documentos procesados y analiza su contexto,
frecuencia y distribución.
""")

# Verificar prerequisitos
if not st.session_state.processed_texts:
    st.warning("⚠️ No hay documentos procesados. Ve a **📁 Subir PDFs** primero.")
    st.stop()

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración de Búsqueda")

# Selector de documentos
available_docs = list(st.session_state.processed_texts.keys())
selected_docs = st.sidebar.multiselect(
    "Documentos a buscar:",
    available_docs,
    default=available_docs[:min(3, len(available_docs))]
)

if not selected_docs:
    st.info("👈 Selecciona al menos un documento en el sidebar")
    st.stop()

# Tipo de búsqueda
search_type = st.sidebar.radio(
    "Tipo de búsqueda:",
    ["Exacta", "Contiene", "Expresión Regular"]
)

# Opciones avanzadas
st.sidebar.subheader("Opciones Avanzadas")
case_sensitive = st.sidebar.checkbox("Sensible a mayúsculas/minúsculas", value=False)
show_context = st.sidebar.checkbox("Mostrar contexto", value=True)
context_words = st.sidebar.slider("Palabras de contexto:", 3, 20, 10)

# Búsqueda principal
st.header("📝 Ingresa los términos a buscar")

col1, col2 = st.columns([3, 1])

with col1:
    search_input = st.text_area(
        "Términos (uno por línea):",
        placeholder="programa\nestudio\ninvestigación\nmetodología",
        height=150
    )

with col2:
    st.markdown("**Ejemplos:**")
    if st.button("Ejemplo: Términos académicos"):
        search_input = "investigación\nmetodología\nanálisis\nestudio"
    if st.button("Ejemplo: Habilidades"):
        search_input = "pensamiento crítico\nresolución\ncomunicación\ncolaboración"

# Botón de búsqueda
if st.button("🔍 Buscar", type="primary", use_container_width=True):
    if not search_input.strip():
        st.error("Por favor ingresa al menos un término de búsqueda")
    else:
        # Procesar términos de búsqueda
        search_terms = [t.strip() for t in search_input.split('\n') if t.strip()]

        with st.spinner(f"Buscando {len(search_terms)} términos en {len(selected_docs)} documentos..."):
            # Resultados
            results = defaultdict(lambda: defaultdict(list))
            term_frequencies = defaultdict(lambda: defaultdict(int))

            for doc_name in selected_docs:
                processed = st.session_state.processed_texts[doc_name]
                text = processed.clean_text
                tokens = processed.tokens

                if not case_sensitive:
                    text_lower = text.lower()
                else:
                    text_lower = text

                for term in search_terms:
                    search_term = term if case_sensitive else term.lower()

                    # Realizar búsqueda según tipo
                    if search_type == "Exacta":
                        # Búsqueda exacta en tokens
                        matches = [i for i, t in enumerate(tokens) if
                                 (t if case_sensitive else t.lower()) == search_term]
                        count = len(matches)

                    elif search_type == "Contiene":
                        # Búsqueda que contiene el término
                        matches = [i for i, t in enumerate(tokens) if
                                 search_term in (t if case_sensitive else t.lower())]
                        count = len(matches)

                    else:  # Expresión Regular
                        try:
                            pattern = re.compile(term, re.IGNORECASE if not case_sensitive else 0)
                            matches = [i for i, t in enumerate(tokens) if pattern.search(t)]
                            count = len(matches)
                        except re.error:
                            st.error(f"Error en expresión regular: {term}")
                            continue

                    # Guardar frecuencia
                    term_frequencies[term][doc_name] = count

                    # Extraer contextos si se encontraron matches
                    if count > 0 and show_context:
                        for match_idx in matches[:5]:  # Limitar a 5 contextos por término
                            start = max(0, match_idx - context_words)
                            end = min(len(tokens), match_idx + context_words + 1)
                            context = ' '.join(tokens[start:end])

                            # Resaltar el término encontrado
                            highlight_term = tokens[match_idx]
                            context = context.replace(
                                highlight_term,
                                f"**{highlight_term}**"
                            )

                            results[term][doc_name].append(context)

        # Mostrar resultados
        st.success(f"✅ Búsqueda completada en {len(selected_docs)} documentos")

        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Resumen",
            "📈 Frecuencias",
            "📝 Contextos",
            "📋 Tabla Completa"
        ])

        with tab1:
            st.subheader("Resumen de Resultados")

            # Métricas generales
            col1, col2, col3, col4 = st.columns(4)

            total_matches = sum(sum(freqs.values()) for freqs in term_frequencies.values())
            terms_found = sum(1 for term_freqs in term_frequencies.values()
                            if sum(term_freqs.values()) > 0)

            col1.metric("Términos buscados", len(search_terms))
            col2.metric("Términos encontrados", terms_found)
            col3.metric("Total ocurrencias", total_matches)
            col4.metric("Documentos analizados", len(selected_docs))

            # Top términos
            st.markdown("### Top 10 Términos Más Frecuentes")

            term_totals = {term: sum(freqs.values())
                          for term, freqs in term_frequencies.items()}
            top_terms = sorted(term_totals.items(), key=lambda x: x[1], reverse=True)[:10]

            if top_terms:
                df_top = pd.DataFrame(top_terms, columns=['Término', 'Frecuencia Total'])

                fig = go.Figure(go.Bar(
                    x=df_top['Frecuencia Total'],
                    y=df_top['Término'],
                    orientation='h',
                    marker_color='lightblue'
                ))
                fig.update_layout(
                    title="Top 10 Términos",
                    xaxis_title="Frecuencia",
                    yaxis_title="Término",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Distribución de Frecuencias")

            # Crear matriz de frecuencias
            freq_matrix = []
            for term in search_terms:
                row = {'Término': term}
                for doc in selected_docs:
                    row[doc] = term_frequencies[term][doc]
                row['Total'] = sum(term_frequencies[term].values())
                freq_matrix.append(row)

            df_freq = pd.DataFrame(freq_matrix)

            # Heatmap de frecuencias
            if len(selected_docs) > 1:
                st.markdown("### Mapa de Calor de Frecuencias")

                # Preparar datos para heatmap
                heatmap_data = df_freq[selected_docs].values

                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=selected_docs,
                    y=df_freq['Término'],
                    colorscale='Blues',
                    text=heatmap_data,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Frecuencia")
                ))

                fig.update_layout(
                    title="Distribución de Términos por Documento",
                    xaxis_title="Documento",
                    yaxis_title="Término",
                    height=max(400, len(search_terms) * 30)
                )

                st.plotly_chart(fig, use_container_width=True)

            # Tabla de frecuencias
            st.markdown("### Tabla de Frecuencias")
            st.dataframe(
                df_freq.style.background_gradient(subset=selected_docs, cmap='Blues'),
                use_container_width=True
            )

            # Exportar
            csv = df_freq.to_csv(index=False)
            st.download_button(
                "📥 Descargar tabla CSV",
                csv,
                "frecuencias_busqueda.csv",
                "text/csv"
            )

        with tab3:
            if show_context:
                st.subheader("Contextos de Aparición")

                # Selector de término
                terms_with_results = [t for t in search_terms
                                    if sum(term_frequencies[t].values()) > 0]

                if terms_with_results:
                    selected_term = st.selectbox(
                        "Selecciona un término:",
                        terms_with_results
                    )

                    st.markdown(f"### Contextos de: **{selected_term}**")

                    for doc_name in selected_docs:
                        if doc_name in results[selected_term]:
                            with st.expander(
                                f"📄 {doc_name} ({len(results[selected_term][doc_name])} ocurrencias)"
                            ):
                                for i, context in enumerate(results[selected_term][doc_name], 1):
                                    st.markdown(f"**{i}.** ...{context}...")
                                    if i < len(results[selected_term][doc_name]):
                                        st.divider()
                else:
                    st.info("No se encontraron resultados para ningún término")
            else:
                st.info("Habilita 'Mostrar contexto' en el sidebar para ver esta pestaña")

        with tab4:
            st.subheader("Tabla Completa de Resultados")

            # Crear tabla completa
            full_results = []
            for term in search_terms:
                for doc in selected_docs:
                    freq = term_frequencies[term][doc]
                    contexts_count = len(results[term][doc]) if doc in results[term] else 0

                    full_results.append({
                        'Término': term,
                        'Documento': doc,
                        'Frecuencia': freq,
                        'Contextos capturados': contexts_count,
                        'Encontrado': '✅' if freq > 0 else '❌'
                    })

            df_full = pd.DataFrame(full_results)

            # Filtros
            col1, col2 = st.columns(2)
            with col1:
                filter_found = st.checkbox("Solo mostrar encontrados", value=False)
            with col2:
                min_freq = st.number_input("Frecuencia mínima:", 0, 100, 0)

            # Aplicar filtros
            df_filtered = df_full.copy()
            if filter_found:
                df_filtered = df_filtered[df_filtered['Frecuencia'] > 0]
            if min_freq > 0:
                df_filtered = df_filtered[df_filtered['Frecuencia'] >= min_freq]

            st.dataframe(df_filtered, use_container_width=True)

            # Exportar
            csv_full = df_filtered.to_csv(index=False)
            st.download_button(
                "📥 Descargar resultados completos CSV",
                csv_full,
                "resultados_busqueda_completos.csv",
                "text/csv"
            )

# Sidebar - Estadísticas
st.sidebar.divider()
st.sidebar.subheader("📊 Estadísticas")
st.sidebar.metric("Documentos procesados", len(st.session_state.processed_texts))
st.sidebar.metric("Documentos seleccionados", len(selected_docs))

# Tips
with st.sidebar.expander("💡 Tips de Búsqueda"):
    st.markdown("""
    **Búsqueda Exacta:**
    - Busca el término exacto
    - Ej: "programa" solo encuentra "programa"

    **Búsqueda Contiene:**
    - Encuentra términos que contengan el texto
    - Ej: "program" encuentra "programa", "programación"

    **Expresión Regular:**
    - Usa patrones avanzados
    - Ej: `invest.*` encuentra "investigación", "investigador"
    - Ej: `(análisis|analizar)` encuentra ambos términos
    """)
