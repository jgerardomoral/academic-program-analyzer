"""
Página de topic modeling y mapeo de habilidades.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json

# Setup path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

# Ahora importar
from utils.session_manager import init_session_state
from utils.cache_manager import cached_topic_modeling
from src.visualization.plotly_charts import (
    create_topic_heatmap,
    create_topic_distribution_stacked,
    create_frequency_bar_chart,
    create_skills_radar_chart,
    create_skills_heatmap
)

# Inicializar
init_session_state()

st.title("🎯 Topics y Habilidades")

# Verificar prerequisitos
if not st.session_state.processed_texts:
    st.warning("⚠️ No hay documentos procesados.")
    st.info("👉 Ve a **📁 Subir PDFs** para cargar y procesar documentos primero.")
    st.stop()

if not st.session_state.frequency_analyses:
    st.warning("⚠️ Ejecuta el análisis de frecuencias primero.")
    st.info("👉 Ve a **📊 Análisis de Frecuencias** para analizar los documentos.")
    st.stop()

# Tabs principales
tab1, tab2, tab3 = st.tabs([
    "🔍 Topic Modeling",
    "🎯 Mapeo de Habilidades",
    "⚙️ Configurar Taxonomía"
])

# ============================================================================
# TAB 1: Topic Modeling
# ============================================================================
with tab1:
    st.subheader("Descubrimiento Automático de Topics")

    st.markdown("""
    El topic modeling identifica automáticamente temas recurrentes en los documentos.
    Cada topic es una distribución de palabras que aparecen juntas frecuentemente.
    """)

    # Sidebar para topic modeling
    with st.sidebar:
        st.header("🎛️ Configuración de Topics")

        n_topics = st.slider(
            "Número de topics",
            min_value=3,
            max_value=20,
            value=10,
            help="Más topics = más granularidad"
        )

        method = st.selectbox(
            "Método",
            options=['lda', 'nmf'],
            format_func=lambda x: 'LDA (Latent Dirichlet Allocation)' if x=='lda' else 'NMF (Non-negative Matrix Factorization)'
        )

        auto_label = st.checkbox(
            "Auto-etiquetar topics",
            value=True,
            help="Genera etiquetas automáticas basadas en keywords"
        )

        run_modeling = st.button("🚀 Ejecutar Topic Modeling", type="primary")

    # Ejecutar modeling
    if run_modeling or st.session_state.topic_model is not None:

        if run_modeling:
            # Preparar documentos
            processed_dicts = []
            for doc_name in st.session_state.processed_texts.keys():
                processed = st.session_state.processed_texts[doc_name]
                processed_dict = processed.to_dict()
                processed_dict['clean_text'] = processed.clean_text
                processed_dict['tokens'] = processed.tokens
                processed_dict['lemmas'] = processed.lemmas
                processed_dict['pos_tags'] = processed.pos_tags
                processed_dict['entities'] = processed.entities
                processed_dicts.append(processed_dict)

            with st.spinner(f"Entrenando modelo {method.upper()} con {n_topics} topics..."):
                topic_result = cached_topic_modeling(processed_dicts, n_topics, method)

                # Auto-etiquetar si está activado
                if auto_label:
                    from src.analysis.topics import TopicModeler
                    modeler = TopicModeler()
                    topic_result = modeler.auto_label_topics(topic_result)

                # Guardar en session state
                st.session_state.topic_model = topic_result

            st.success(f"✅ Modelo {method.upper()} entrenado con {n_topics} topics")

        topic_result = st.session_state.topic_model

        # Métricas del modelo
        col1, col2, col3 = st.columns(3)
        col1.metric("Topics", topic_result.n_topics)
        col2.metric("Coherence Score", f"{topic_result.coherence_score:.3f}")
        if topic_result.perplexity:
            col3.metric("Perplexity", f"{topic_result.perplexity:.1f}")

        st.markdown("---")

        # Mostrar topics
        st.subheader("📋 Topics Identificados")

        for topic in topic_result.topics:
            with st.expander(f"**Topic {topic.topic_id}: {topic.label}**", expanded=False):
                # Keywords con pesos
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Top Keywords:**")
                    keywords_df = pd.DataFrame({
                        'Keyword': topic.keywords[:10],
                        'Peso': [f"{w:.3f}" for w in topic.weights[:10]]
                    })
                    st.dataframe(keywords_df, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("**Documentos asociados:**")
                    st.write(f"Total: {len(topic.documents)}")
                    if topic.documents:
                        for doc in topic.documents[:5]:
                            st.write(f"- {doc}")
                        if len(topic.documents) > 5:
                            st.write(f"... y {len(topic.documents)-5} más")

                # Editar etiqueta
                new_label = st.text_input(
                    "Renombrar topic:",
                    value=topic.label,
                    key=f"label_{topic.topic_id}"
                )

                if new_label != topic.label:
                    topic.label = new_label
                    st.success(f"Topic renombrado a: {new_label}")

        # Visualización de distribución de topics
        st.markdown("---")
        st.subheader("📊 Distribución de Topics en Documentos")

        # Preparar labels
        topic_labels = [t.label for t in topic_result.topics]

        # Heatmap de doc-topic
        fig_heatmap = create_topic_heatmap(
            topic_result.document_topic_matrix,
            topic_labels=topic_labels,
            title="Distribución de Topics por Documento"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Gráfico de barras apiladas
        st.markdown("#### Distribución Acumulativa")
        fig_stacked = create_topic_distribution_stacked(
            topic_result.document_topic_matrix,
            topic_labels=topic_labels,
            title="Distribución de Topics (Barras Apiladas)"
        )
        st.plotly_chart(fig_stacked, use_container_width=True)

        # Tabla de doc-topic matrix
        with st.expander("📋 Ver Matriz Documento-Topic"):
            # Redondear valores
            display_matrix = topic_result.document_topic_matrix.round(3)
            display_matrix.columns = topic_labels

            st.dataframe(display_matrix, use_container_width=True)

            # Exportar
            csv = display_matrix.to_csv().encode('utf-8')
            st.download_button(
                "📥 Descargar CSV",
                data=csv,
                file_name="doc_topic_matrix.csv",
                mime="text/csv"
            )

    else:
        st.info("👈 Configura los parámetros en el sidebar y presiona **Ejecutar Topic Modeling**")

# ============================================================================
# TAB 2: Mapeo de Habilidades
# ============================================================================
with tab2:
    st.subheader("Mapeo de Habilidades")

    st.markdown("""
    Identifica qué habilidades se trabajan en cada programa basándose en la taxonomía definida.
    """)

    # Botón para ejecutar mapeo
    if st.button("🎯 Mapear Habilidades", type="primary"):
        from src.analysis.skills_mapper import SkillsMapper

        mapper = SkillsMapper()

        with st.spinner("Mapeando habilidades..."):
            # Mapear cada documento
            for doc_name in st.session_state.frequency_analyses.keys():
                freq_analysis = st.session_state.frequency_analyses[doc_name]
                processed = st.session_state.processed_texts[doc_name]

                profile = mapper.map_document(freq_analysis, processed)
                st.session_state.skill_profiles[doc_name] = profile

        st.success(f"✅ {len(st.session_state.skill_profiles)} documentos mapeados")

    # Mostrar resultados si existen
    if st.session_state.skill_profiles:
        st.markdown("---")

        # Selector de documento
        selected_doc = st.selectbox(
            "Selecciona documento",
            options=list(st.session_state.skill_profiles.keys())
        )

        profile = st.session_state.skill_profiles[selected_doc]

        # Métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("Habilidades detectadas", len(profile.skill_scores))
        col2.metric("Habilidades relevantes", len(profile.top_skills))
        col3.metric("Cobertura", f"{profile.skill_coverage*100:.1f}%")

        # Top habilidades
        st.markdown("### 🏆 Top Habilidades")

        df_skills = profile.to_dataframe()

        # Filtrar por confidence mínima
        min_confidence = st.slider(
            "Confianza mínima",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05
        )

        df_filtered = df_skills[df_skills['confidence'] >= min_confidence]

        # Visualización de barras
        if not df_filtered.empty:
            # Gráfico de barras de habilidades
            df_chart = df_filtered.rename(columns={'skill': 'term', 'score': 'tfidf'})
            df_chart['frequency'] = df_chart['confidence']  # Usar confidence como frequency para hover

            fig_skills = create_frequency_bar_chart(
                df_chart.head(20),
                title="Habilidades Identificadas (ordenadas por score)",
                top_n=20,
                value_column='tfidf'
            )
            st.plotly_chart(fig_skills, use_container_width=True)

            # Gráfico de radar
            st.markdown("#### Radar de Habilidades")
            fig_radar = create_skills_radar_chart(
                profile.skill_scores,
                title=f"Perfil de Habilidades: {selected_doc}",
                top_n=10
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Tabla detallada
            st.markdown("#### Tabla Detallada")
            st.dataframe(df_filtered, use_container_width=True)

            # Mostrar snippets de contexto para top habilidades
            st.markdown("#### Contexto de Habilidades")
            top_skills_with_context = [
                ss for ss in profile.skill_scores
                if ss.confidence >= min_confidence and ss.context_snippets
            ][:5]

            if top_skills_with_context:
                for skill_score in top_skills_with_context:
                    with st.expander(f"📝 {skill_score.skill_name} (Score: {skill_score.score:.2f})"):
                        st.write("**Términos encontrados:**")
                        st.write(", ".join(skill_score.matched_terms[:10]))

                        if skill_score.context_snippets:
                            st.write("**Ejemplos en contexto:**")
                            for snippet in skill_score.context_snippets:
                                st.info(snippet)

        else:
            st.warning("No hay habilidades que cumplan el criterio de confianza")

        # Matriz de habilidades (si hay múltiples documentos)
        if len(st.session_state.skill_profiles) > 1:
            st.markdown("---")
            st.subheader("📊 Matriz de Habilidades")

            from src.analysis.skills_mapper import SkillsMapper
            mapper = SkillsMapper()

            profiles_list = list(st.session_state.skill_profiles.values())
            skills_matrix = mapper.create_skills_matrix(profiles_list)

            # Heatmap
            fig_matrix = create_skills_heatmap(
                profiles_list,
                title="Habilidades por Documento"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

            # Tabla
            with st.expander("📋 Ver Tabla Completa"):
                st.dataframe(skills_matrix.round(3), use_container_width=True)

                # Exportar
                csv = skills_matrix.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Descargar Matriz CSV",
                    data=csv,
                    file_name="skills_matrix.csv",
                    mime="text/csv"
                )

    else:
        st.info("Presiona **Mapear Habilidades** para comenzar")

# ============================================================================
# TAB 3: Configurar Taxonomía
# ============================================================================
with tab3:
    st.subheader("⚙️ Gestión de Taxonomía de Habilidades")

    from src.analysis.skills_mapper import SkillsMapper
    mapper = SkillsMapper()

    st.markdown("""
    Define o modifica las habilidades que el sistema buscará en los documentos.
    Cada habilidad tiene keywords que se usarán para identificarla.
    """)

    # Mostrar taxonomía actual
    st.markdown("### 📚 Habilidades Actuales")

    # Convertir a DataFrame para visualización
    taxonomia_data = []
    for skill_id, data in st.session_state.taxonomia.items():
        taxonomia_data.append({
            'ID': skill_id,
            'Nombre': data['name'],
            'Keywords': ', '.join(data['keywords'][:5]) + ('...' if len(data['keywords']) > 5 else ''),
            'Categoría': data.get('category', 'general'),
            'Peso': data.get('weight', 1.0)
        })

    df_taxonomia = pd.DataFrame(taxonomia_data)
    st.dataframe(df_taxonomia, use_container_width=True)

    st.markdown("---")

    # Agregar nueva habilidad
    st.markdown("### ➕ Agregar Nueva Habilidad")

    col1, col2 = st.columns(2)

    with col1:
        new_skill_id = st.text_input(
            "ID de habilidad",
            placeholder="ej: comunicacion_efectiva",
            help="Identificador único (sin espacios)"
        )

        new_skill_name = st.text_input(
            "Nombre",
            placeholder="ej: Comunicación Efectiva"
        )

    with col2:
        new_skill_category = st.selectbox(
            "Categoría",
            options=['cognitiva', 'tecnica', 'interpersonal', 'general']
        )

        new_skill_weight = st.number_input(
            "Peso",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1
        )

    new_keywords = st.text_area(
        "Keywords (una por línea)",
        placeholder="comunicar\nexponer\npresentar\nredactar",
        height=100
    )

    new_synonyms = st.text_area(
        "Sinónimos (una por línea)",
        placeholder="expresar\narticular",
        height=80
    )

    if st.button("➕ Agregar Habilidad"):
        if new_skill_id and new_skill_name and new_keywords:
            # Parsear keywords y synonyms
            keywords_list = [k.strip() for k in new_keywords.split('\n') if k.strip()]
            synonyms_list = [s.strip() for s in new_synonyms.split('\n') if s.strip()]

            # Agregar a taxonomía
            st.session_state.taxonomia[new_skill_id] = {
                'name': new_skill_name,
                'keywords': keywords_list,
                'synonyms': synonyms_list,
                'weight': new_skill_weight,
                'category': new_skill_category
            }

            # Actualizar mapper
            mapper.save_taxonomy(st.session_state.taxonomia)

            st.success(f"✅ Habilidad '{new_skill_name}' agregada")
            st.rerun()
        else:
            st.error("Debes completar al menos ID, Nombre y Keywords")

    st.markdown("---")

    # Editar/Eliminar habilidades existentes
    st.markdown("### ✏️ Editar Habilidades")

    if st.session_state.taxonomia:
        skill_to_edit = st.selectbox(
            "Selecciona habilidad para editar",
            options=list(st.session_state.taxonomia.keys()),
            format_func=lambda x: st.session_state.taxonomia[x]['name']
        )

        if skill_to_edit:
            skill_data = st.session_state.taxonomia[skill_to_edit]

            col1, col2 = st.columns([3, 1])

            with col1:
                # Mostrar datos actuales
                st.json(skill_data)

            with col2:
                if st.button("🗑️ Eliminar", key="delete_skill"):
                    del st.session_state.taxonomia[skill_to_edit]
                    mapper.save_taxonomy(st.session_state.taxonomia)
                    st.success(f"Habilidad eliminada")
                    st.rerun()
    else:
        st.info("No hay habilidades en la taxonomía. Agrega una nueva para comenzar.")

    st.markdown("---")

    # Importar/Exportar
    st.markdown("### 📤📥 Importar/Exportar Taxonomía")

    col1, col2 = st.columns(2)

    with col1:
        # Exportar
        taxonomia_json = json.dumps(st.session_state.taxonomia, indent=2, ensure_ascii=False)

        st.download_button(
            label="📥 Exportar JSON",
            data=taxonomia_json,
            file_name="taxonomia_habilidades.json",
            mime="application/json"
        )

    with col2:
        # Importar
        uploaded_taxonomia = st.file_uploader(
            "📤 Importar JSON",
            type=['json'],
            help="Sube un archivo JSON con la estructura de taxonomía"
        )

        if uploaded_taxonomia:
            new_taxonomia = json.load(uploaded_taxonomia)

            if st.button("✅ Confirmar Importación"):
                st.session_state.taxonomia = new_taxonomia
                mapper.save_taxonomy(new_taxonomia)
                st.success("Taxonomía importada correctamente")
                st.rerun()

# Tips en sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    **Topic Modeling:**
    - LDA: Mejor para textos largos y académicos
    - NMF: Más rápido, buenos resultados en textos cortos
    - Más topics = temas más específicos

    **Mapeo de Habilidades:**
    - Define una buena taxonomía primero
    - Usa confianza mínima para filtrar
    - Revisa el contexto de términos detectados
    """)

    # Mostrar estadísticas
    if st.session_state.topic_model:
        st.markdown("---")
        st.markdown("### 📊 Estadísticas")
        st.metric("Topics Generados", st.session_state.topic_model.n_topics)
        st.metric("Habilidades Mapeadas", len(st.session_state.skill_profiles))
        st.metric("Habilidades en Taxonomía", len(st.session_state.taxonomia))
