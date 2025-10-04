"""
Página de comparación entre programas/documentos.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Setup path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

# Ahora importar
from utils.session_manager import init_session_state
from utils.export import export_multiple_to_excel, export_comparison_report
from src.visualization.plotly_charts import PlotlyCharts
from src.analysis.skills_mapper import SkillsMapper

# Inicializar
init_session_state()

st.title("📈 Comparativa entre Programas")

# Verificar prerequisitos
if not st.session_state.skill_profiles:
    st.warning("⚠️ Primero debes mapear habilidades. Ve a **🎯 Topics y Habilidades**.")
    st.stop()

st.markdown("""
Compara perfiles de habilidades, vocabulario y características entre diferentes programas.
""")

# Selector de programas a comparar
st.subheader("Selección de Programas")

available_programs = list(st.session_state.skill_profiles.keys())

col1, col2 = st.columns([3, 1])

with col1:
    selected_programs = st.multiselect(
        "Programas a comparar (2-4 recomendado)",
        options=available_programs,
        default=available_programs[:min(2, len(available_programs))],
        help="Selecciona entre 2 y 4 programas para comparar"
    )

with col2:
    comparison_type = st.selectbox(
        "Tipo de comparación",
        options=['habilidades', 'vocabulario', 'topics', 'completa'],
        format_func=lambda x: {
            'habilidades': '🎯 Habilidades',
            'vocabulario': '📚 Vocabulario',
            'topics': '🔍 Topics',
            'completa': '📊 Completa'
        }[x]
    )

if len(selected_programs) < 2:
    st.warning("⚠️ Selecciona al menos 2 programas para comparar")
    st.stop()

if len(selected_programs) > 4:
    st.info("ℹ️ Para mejor visualización, se recomienda comparar máximo 4 programas a la vez.")

# Inicializar visualizaciones y análisis
charts = PlotlyCharts()
mapper = SkillsMapper()

# Crear matriz de habilidades una sola vez (usada en múltiples tabs)
selected_profiles = [
    st.session_state.skill_profiles[prog]
    for prog in selected_programs
]
skills_matrix = mapper.create_skills_matrix(selected_profiles)

# Tabs según tipo de comparación
if comparison_type == 'completa':
    tabs = st.tabs([
        "🎯 Habilidades",
        "📊 Radar Chart",
        "📚 Vocabulario",
        "🔍 Topics",
        "📋 Resumen"
    ])
else:
    tabs = [st.container()]

# ==================== TAB 1: HABILIDADES ====================
with tabs[0] if comparison_type == 'completa' else tabs[0]:
    if comparison_type in ['habilidades', 'completa']:
        st.subheader("🎯 Comparación de Habilidades")

        # Filtro de habilidades
        min_score = st.slider(
            "Score mínimo para mostrar",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            key="habilidades_min_score"
        )

        # Filtrar habilidades con score bajo en todos los programas
        skills_to_show = skills_matrix.index[
            (skills_matrix.max(axis=1) >= min_score)
        ].tolist()

        filtered_matrix = skills_matrix.loc[skills_to_show]

        # Heatmap
        if not filtered_matrix.empty:
            fig = charts.plot_skills_heatmap(
                filtered_matrix.T,  # Transponer: programas en filas
                title=f"Comparación de Habilidades ({len(filtered_matrix)} habilidades)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Análisis de diferencias
            st.markdown("### 📊 Análisis de Diferencias")

            # Calcular estadísticas por habilidad
            stats_data = []
            for skill in filtered_matrix.index:
                values = filtered_matrix.loc[skill].values
                stats_data.append({
                    'Habilidad': skill,
                    'Promedio': values.mean(),
                    'Desv. Std': values.std(),
                    'Máximo': values.max(),
                    'Mínimo': values.min(),
                    'Rango': values.max() - values.min()
                })

            df_stats = pd.DataFrame(stats_data).sort_values('Rango', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Habilidades más variables** (mayor diferencia)")
                st.dataframe(
                    df_stats.head(10)[['Habilidad', 'Rango', 'Desv. Std']],
                    use_container_width=True,
                    hide_index=True
                )

            with col2:
                st.markdown("**Habilidades más uniformes** (menor diferencia)")
                st.dataframe(
                    df_stats[df_stats['Promedio'] > 0.1].nsmallest(10, 'Rango')[['Habilidad', 'Rango', 'Promedio']],
                    use_container_width=True,
                    hide_index=True
                )

            # Tabla detallada
            with st.expander("📋 Ver Tabla Completa de Habilidades"):
                display_matrix = filtered_matrix.T.round(3)
                st.dataframe(display_matrix, use_container_width=True)

                # Exportar
                csv = display_matrix.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Descargar CSV",
                    data=csv,
                    file_name=f"comparativa_habilidades.csv",
                    mime="text/csv"
                )

        else:
            st.warning("No hay habilidades que cumplan el criterio de score mínimo")

# ==================== TAB 2: RADAR CHART ====================
if comparison_type == 'completa':
    with tabs[1]:
        st.subheader("📊 Radar Chart de Habilidades")

        st.markdown("""
        Visualización radial que permite ver el perfil de habilidades de cada programa.
        Ideal para identificar fortalezas y debilidades relativas.
        """)

        # Selector de habilidades para el radar
        top_n_skills = st.slider(
            "Número de habilidades a mostrar",
            min_value=5,
            max_value=15,
            value=8,
            help="Más habilidades puede hacer el gráfico difícil de leer"
        )

        # Seleccionar top habilidades por promedio
        avg_scores = skills_matrix.mean(axis=1).sort_values(ascending=False)
        top_skills = avg_scores.head(top_n_skills).index.tolist()

        radar_matrix = skills_matrix.loc[top_skills].T

        # Generar radar chart
        fig = charts.plot_skills_radar(
            radar_matrix,
            selected_programs,
            title=f"Perfil de Top {top_n_skills} Habilidades"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interpretación
        st.markdown("### 💡 Interpretación")

        for prog in selected_programs:
            prog_scores = radar_matrix.loc[prog]
            top_3 = prog_scores.nlargest(3)

            st.markdown(f"**{prog}:**")
            st.write(f"- Fortalezas principales: {', '.join(top_3.index[:3])}")

            if len(prog_scores[prog_scores < 0.3]) > 0:
                weak = prog_scores[prog_scores < 0.3].index.tolist()
                st.write(f"- Áreas de menor énfasis: {', '.join(weak[:3])}")

# ==================== TAB 3: VOCABULARIO ====================
with tabs[2 if comparison_type == 'completa' else 0]:
    if comparison_type in ['vocabulario', 'completa']:
        st.subheader("📚 Comparación de Vocabulario")

        # Obtener análisis de frecuencias
        if all(prog in st.session_state.frequency_analyses for prog in selected_programs):

            # Diagrama de Venn conceptual
            st.markdown("### 🔵 Términos Únicos vs Compartidos")

            # Calcular términos por programa
            program_terms = {}
            for prog in selected_programs:
                freq_analysis = st.session_state.frequency_analyses[prog]
                # Tomar términos con TF-IDF significativo
                significant_terms = freq_analysis.term_frequencies[
                    freq_analysis.term_frequencies['tfidf'] > 0.1
                ]['term'].tolist()
                program_terms[prog] = set(significant_terms)

            # Métricas
            col1, col2, col3, col4 = st.columns(4)

            all_terms = set.union(*program_terms.values())
            shared_all = set.intersection(*program_terms.values())

            col1.metric("Total términos únicos", len(all_terms))
            col2.metric("Compartidos por todos", len(shared_all))
            col3.metric("Promedio por programa",
                       int(np.mean([len(terms) for terms in program_terms.values()])))
            col4.metric("Overlap promedio",
                       f"{(len(shared_all)/len(all_terms)*100):.1f}%" if len(all_terms) > 0 else "0%")

            # Tabla de términos por programa
            st.markdown("### 📊 Análisis de Términos")

            terms_data = []
            for prog in selected_programs:
                unique = program_terms[prog] - set.union(
                    *[program_terms[p] for p in selected_programs if p != prog]
                )
                terms_data.append({
                    'Programa': prog,
                    'Total términos': len(program_terms[prog]),
                    'Únicos': len(unique),
                    '% Únicos': f"{(len(unique)/len(program_terms[prog])*100):.1f}%" if len(program_terms[prog]) > 0 else "0%"
                })

            df_terms = pd.DataFrame(terms_data)
            st.dataframe(df_terms, use_container_width=True, hide_index=True)

            # Mostrar términos compartidos
            if shared_all:
                with st.expander(f"🔍 Ver {len(shared_all)} términos compartidos por todos"):
                    shared_list = sorted(list(shared_all))
                    # Dividir en columnas
                    n_cols = 3
                    cols = st.columns(n_cols)
                    for i, term in enumerate(shared_list):
                        cols[i % n_cols].write(f"• {term}")

            # Términos únicos por programa
            st.markdown("### 🎯 Términos Distintivos por Programa")

            for prog in selected_programs:
                unique = program_terms[prog] - set.union(
                    *[program_terms[p] for p in selected_programs if p != prog]
                )

                if unique:
                    with st.expander(f"{prog}: {len(unique)} términos únicos"):
                        # Obtener TF-IDF de estos términos
                        freq_analysis = st.session_state.frequency_analyses[prog]
                        unique_df = freq_analysis.term_frequencies[
                            freq_analysis.term_frequencies['term'].isin(unique)
                        ].nlargest(20, 'tfidf')

                        # Gráfico
                        fig = charts.plot_top_terms(
                            unique_df,
                            top_n=20,
                            title=f"Top 20 Términos Únicos de {prog}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Comparación de N-grams
            st.markdown("---")
            st.markdown("### 📝 Comparación de N-grams")

            ngram_type = st.selectbox(
                "Tipo de n-gram",
                options=[1, 2, 3],
                format_func=lambda x: f"{x}-grams",
                key="vocab_ngram_selector"
            )

            # Recopilar n-grams
            ngrams_dict = {}
            for prog in selected_programs:
                freq_analysis = st.session_state.frequency_analyses[prog]
                if ngram_type in freq_analysis.ngrams:
                    ngrams_dict[prog] = freq_analysis.ngrams[ngram_type]

            if ngrams_dict:
                fig = charts.plot_ngrams_comparison(ngrams_dict, n=ngram_type, top=15)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No todos los programas tienen análisis de frecuencias")

# ==================== TAB 4: TOPICS ====================
with tabs[3 if comparison_type == 'completa' else 0]:
    if comparison_type in ['topics', 'completa']:
        st.subheader("🔍 Comparación de Topics")

        if st.session_state.topic_model is not None:
            topic_model = st.session_state.topic_model

            st.markdown("### 📊 Distribución de Topics por Programa")

            # Filtrar matriz para programas seleccionados
            doc_topic_filtered = topic_model.document_topic_matrix.loc[
                [prog for prog in selected_programs
                 if prog in topic_model.document_topic_matrix.index]
            ]

            if not doc_topic_filtered.empty:
                # Preparar labels
                topic_labels = [t.label for t in topic_model.topics]

                # Gráfico de distribución
                fig = charts.plot_topic_distribution(
                    doc_topic_filtered,
                    topic_labels
                )
                st.plotly_chart(fig, use_container_width=True)

                # Análisis de topics dominantes
                st.markdown("### 🏆 Topics Dominantes por Programa")

                for prog in doc_topic_filtered.index:
                    topic_scores = doc_topic_filtered.loc[prog]
                    top_topics_idx = topic_scores.nlargest(3).index

                    st.markdown(f"**{prog}:**")
                    for idx in top_topics_idx:
                        topic_num = int(idx.split('_')[1])
                        topic = topic_model.topics[topic_num]
                        score = topic_scores[idx]

                        st.write(
                            f"- {topic.label} ({score:.2%}): "
                            f"{', '.join(topic.keywords[:5])}"
                        )

                # Heatmap de topics
                st.markdown("### 🗺️ Mapa de Topics")

                fig_heatmap = charts.plot_skills_heatmap(
                    doc_topic_filtered,
                    title="Intensidad de Topics por Programa"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

            else:
                st.warning("Los programas seleccionados no están en el modelo de topics")

        else:
            st.info("Primero debes ejecutar Topic Modeling en **🎯 Topics y Habilidades**")

# ==================== TAB 5: RESUMEN ====================
if comparison_type == 'completa':
    with tabs[4]:
        st.subheader("📋 Resumen Ejecutivo")

        st.markdown("""
        Resumen de las principales diferencias y similitudes entre los programas comparados.
        """)

        # Generar reporte automático
        st.markdown("### 📊 Análisis Comparativo")

        for i, prog1 in enumerate(selected_programs):
            for prog2 in selected_programs[i+1:]:
                with st.expander(f"🔍 {prog1} vs {prog2}"):

                    profile1 = st.session_state.skill_profiles[prog1]
                    profile2 = st.session_state.skill_profiles[prog2]

                    # Comparar perfiles
                    comparison_df = mapper.compare_profiles(profile1, profile2)

                    # Principales diferencias
                    st.markdown("**Mayores diferencias:**")
                    top_diff = comparison_df.nlargest(5, 'difference')

                    for skill in top_diff.index[:5]:
                        score1 = comparison_df.loc[skill, prog1]
                        score2 = comparison_df.loc[skill, prog2]
                        diff = comparison_df.loc[skill, 'difference']

                        if score1 > score2:
                            st.write(
                                f"• **{skill}**: {prog1} más fuerte "
                                f"({score1:.2f} vs {score2:.2f}, diff: {diff:.2f})"
                            )
                        else:
                            st.write(
                                f"• **{skill}**: {prog2} más fuerte "
                                f"({score2:.2f} vs {score1:.2f}, diff: {diff:.2f})"
                            )

                    # Similitudes
                    st.markdown("**Principales similitudes:**")
                    similarities = comparison_df.nsmallest(5, 'difference')
                    similarities = similarities[similarities[prog1] > 0.3]  # Solo significativas

                    if not similarities.empty:
                        for skill in similarities.index[:5]:
                            score_avg = (comparison_df.loc[skill, prog1] +
                                       comparison_df.loc[skill, prog2]) / 2
                            st.write(f"• **{skill}**: ambos ~{score_avg:.2f}")
                    else:
                        st.write("No hay habilidades significativas en común")

        # Exportar reporte completo
        st.markdown("---")
        st.markdown("### 📥 Exportar Reporte")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Generar Reporte Excel"):
                # Preparar DataFrames para exportar
                dataframes = {
                    'Habilidades': skills_matrix.T,
                    'Estadísticas': df_stats if 'df_stats' in locals() else pd.DataFrame(),
                }

                # Agregar términos si existen
                if 'df_terms' in locals():
                    dataframes['Términos'] = df_terms

                # Generar Excel
                excel_bytes = export_multiple_to_excel(
                    dataframes,
                    f"comparativa_{len(selected_programs)}_programas.xlsx"
                )

                st.download_button(
                    label="💾 Descargar Excel",
                    data=excel_bytes,
                    file_name=f"comparativa_{len(selected_programs)}_programas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        with col2:
            if st.button("📄 Generar Reporte HTML"):
                # Preparar datos para reporte
                comparison_data = {
                    'common_terms': list(shared_all) if 'shared_all' in locals() else [],
                    'statistics': df_terms if 'df_terms' in locals() else pd.DataFrame()
                }

                # Generar HTML
                report_html = export_comparison_report(
                    selected_programs,
                    comparison_data,
                    f"reporte_comparativo.html"
                )

                st.download_button(
                    label="💾 Descargar Reporte HTML",
                    data=report_html,
                    file_name=f"reporte_comparativo_{'-'.join(selected_programs[:2])}.html",
                    mime="text/html"
                )

# Sidebar con estadísticas
with st.sidebar:
    st.markdown("### 📊 Estadísticas de Comparación")

    st.metric("Programas comparados", len(selected_programs))

    if st.session_state.skill_profiles:
        total_skills = len(skills_matrix)
        st.metric("Habilidades evaluadas", total_skills)

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - **Radar Chart**: Mejor para visualizar perfiles completos
    - **Heatmap**: Identifica patrones y clusters
    - **Términos únicos**: Revelan especialización
    - **Topics**: Muestran enfoques temáticos
    """)
