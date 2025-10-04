"""
Página de carga y procesamiento de PDFs.
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import io

# Setup path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

# Ahora importar
from utils.session_manager import init_session_state, set_state
from utils.cache_manager import cached_pdf_extraction, cached_text_processing

# Inicializar
init_session_state()

st.title("📁 Subir Programas de Estudio")

st.markdown("""
Sube los PDFs de programas de estudio que quieres analizar.
Puedes subir múltiples archivos a la vez.
""")

# Upload widget
uploaded_files = st.file_uploader(
    "Arrastra archivos PDF o haz clic para seleccionar",
    type=['pdf'],
    accept_multiple_files=True,
    help="Máximo 50MB por archivo"
)

if uploaded_files:
    st.markdown("---")
    st.subheader(f"📋 Archivos Recibidos: {len(uploaded_files)}")

    # Tabla de archivos
    file_data = []
    for file in uploaded_files:
        size_mb = len(file.getvalue()) / (1024 * 1024)
        file_data.append({
            'Archivo': file.name,
            'Tamaño (MB)': f"{size_mb:.2f}",
            'Estado': '⏳ Nuevo' if file.name not in st.session_state.uploaded_pdfs else '✅ Ya cargado'
        })

    st.table(file_data)

    # Metadata adicional
    st.markdown("### 📝 Información Adicional (Opcional)")

    col1, col2, col3 = st.columns(3)

    with col1:
        programa = st.text_input(
            "Programa",
            placeholder="Ej: Ingeniería en Computación",
            help="Programa académico"
        )

    with col2:
        facultad = st.text_input(
            "Facultad",
            placeholder="Ej: Ciencias",
            help="Facultad o departamento"
        )

    with col3:
        año = st.number_input(
            "Año",
            min_value=2000,
            max_value=2030,
            value=datetime.now().year,
            help="Año del programa"
        )

    # Botón de procesamiento
    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

    with col_btn1:
        process_button = st.button(
            "🚀 Procesar PDFs",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        skip_existing = st.checkbox(
            "Saltar ya procesados",
            value=True,
            help="No reprocesar PDFs ya cargados"
        )

    if process_button:
        # Metadata común
        common_metadata = {}
        if programa:
            common_metadata['programa'] = programa
        if facultad:
            common_metadata['facultad'] = facultad
        if año:
            common_metadata['año'] = str(año)

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        total_files = len(uploaded_files)
        success_count = 0
        error_count = 0

        for idx, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name

            # Skip si ya existe
            if skip_existing and filename in st.session_state.uploaded_pdfs:
                status_text.info(f"⏭️ Saltando {filename} (ya procesado)")
                continue

            try:
                # Actualizar status
                status_text.text(f"📄 Procesando {filename}... ({idx+1}/{total_files})")

                # 1. Guardar bytes
                pdf_bytes = uploaded_file.getvalue()
                st.session_state.uploaded_pdfs[filename] = pdf_bytes

                # 2. Extraer texto (con caché)
                with st.spinner(f"Extrayendo texto de {filename}..."):
                    extracted = cached_pdf_extraction(pdf_bytes, filename)

                    # Agregar metadata adicional
                    extracted.metadata.update(common_metadata)

                    # Guardar
                    st.session_state.extracted_texts[filename] = extracted

                # 3. Procesar texto (con caché)
                with st.spinner(f"Procesando texto de {filename}..."):
                    # Convertir a dict para caché
                    extracted_dict = extracted.to_dict()
                    extracted_dict['raw_text'] = extracted.raw_text  # Agregar texto completo

                    processed = cached_text_processing(extracted_dict)
                    st.session_state.processed_texts[filename] = processed

                # Success
                with results_container:
                    st.success(f"✅ {filename} procesado correctamente")
                    with st.expander(f"📊 Estadísticas de {filename}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Páginas", extracted.page_count)
                        col2.metric("Tokens", len(processed.tokens))
                        col3.metric("Vocabulario", len(set(processed.tokens)))

                        # Mostrar preview del texto
                        st.markdown("#### 📝 Preview del Texto Procesado")
                        st.text_area(
                            "Primeros 500 caracteres",
                            processed.clean_text[:500] + "..." if len(processed.clean_text) > 500 else processed.clean_text,
                            height=150,
                            disabled=True,
                            key=f"preview_text_{idx}"
                        )

                        # Mostrar metadata
                        if extracted.metadata:
                            st.markdown("#### 📋 Metadata")
                            st.json(extracted.metadata)

                        # Mostrar información de tablas si existen
                        if extracted.has_tables:
                            st.markdown(f"#### 📊 Tablas encontradas: {len(extracted.tables)}")

                        # Botón de descarga del texto extraído
                        st.download_button(
                            label="💾 Descargar Texto Extraído",
                            data=extracted.raw_text,
                            file_name=f"{filename}_extracted.txt",
                            mime="text/plain",
                            key=f"download_{idx}"
                        )

                success_count += 1

            except Exception as e:
                error_count += 1
                with results_container:
                    st.error(f"❌ Error procesando {filename}: {str(e)}")

                # Log error
                import logging
                logging.error(f"Error processing {filename}: {str(e)}", exc_info=True)

            # Actualizar progress
            progress_bar.progress((idx + 1) / total_files)

        # Resumen final
        status_text.empty()
        progress_bar.empty()

        st.markdown("---")
        st.subheader("📊 Resumen de Procesamiento")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total_files)
        col2.metric("Exitosos", success_count, delta=success_count)
        col3.metric("Errores", error_count, delta=-error_count if error_count > 0 else None)

        if success_count > 0:
            st.success(f"🎉 {success_count} documentos listos para análisis!")
            st.info("👉 Ve a **📊 Análisis de Frecuencias** para explorar los datos")

# Sección de documentos ya cargados
if st.session_state.uploaded_pdfs:
    st.markdown("---")
    st.subheader("📚 Documentos en el Sistema")

    # Filtros
    col1, col2 = st.columns([3, 1])

    with col1:
        search = st.text_input(
            "🔍 Buscar documento",
            placeholder="Nombre del archivo..."
        )

    with col2:
        show_details = st.checkbox("Ver detalles", value=False)

    # Lista de documentos
    for filename in st.session_state.uploaded_pdfs.keys():
        # Filtro de búsqueda
        if search and search.lower() not in filename.lower():
            continue

        with st.expander(f"📄 {filename}", expanded=False):
            # Estado
            is_extracted = filename in st.session_state.extracted_texts
            is_processed = filename in st.session_state.processed_texts
            is_analyzed = filename in st.session_state.frequency_analyses

            col1, col2, col3 = st.columns(3)
            col1.write(f"**Extraído:** {'✅' if is_extracted else '❌'}")
            col2.write(f"**Procesado:** {'✅' if is_processed else '❌'}")
            col3.write(f"**Analizado:** {'✅' if is_analyzed else '❌'}")

            # Detalles
            if show_details and is_extracted:
                extracted = st.session_state.extracted_texts[filename]
                st.json(extracted.metadata)

            # Acciones
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button(f"🔄 Reprocesar", key=f"reprocess_{filename}"):
                    # Limpiar caché de este archivo
                    if filename in st.session_state.processed_texts:
                        del st.session_state.processed_texts[filename]
                    if filename in st.session_state.frequency_analyses:
                        del st.session_state.frequency_analyses[filename]
                    st.success(f"Listo para reprocesar {filename}")
                    st.rerun()

            with col_b:
                if is_processed:
                    processed = st.session_state.processed_texts[filename]
                    if st.button(f"👁️ Preview", key=f"preview_{filename}"):
                        st.text_area(
                            "Texto Procesado (primeros 500 caracteres)",
                            processed.clean_text[:500],
                            height=150,
                            disabled=True
                        )

            with col_c:
                if st.button(f"🗑️ Eliminar", key=f"delete_{filename}"):
                    # Eliminar de todos los estados
                    del st.session_state.uploaded_pdfs[filename]
                    if filename in st.session_state.extracted_texts:
                        del st.session_state.extracted_texts[filename]
                    if filename in st.session_state.processed_texts:
                        del st.session_state.processed_texts[filename]
                    if filename in st.session_state.frequency_analyses:
                        del st.session_state.frequency_analyses[filename]
                    st.success(f"Eliminado {filename}")
                    st.rerun()

# Acciones globales
if st.session_state.uploaded_pdfs:
    st.markdown("---")
    st.subheader("⚙️ Acciones Globales")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reprocesar Todos", use_container_width=True):
            st.session_state.processed_texts = {}
            st.session_state.frequency_analyses = {}
            st.success("Todos los documentos listos para reprocesar")
            st.rerun()

    with col2:
        if st.button("🗑️ Eliminar Todos", use_container_width=True, type="secondary"):
            if st.checkbox("Confirmar eliminación de TODOS los documentos"):
                st.session_state.uploaded_pdfs = {}
                st.session_state.extracted_texts = {}
                st.session_state.processed_texts = {}
                st.session_state.frequency_analyses = {}
                st.success("Todos los documentos eliminados")
                st.rerun()

# Tips
with st.sidebar:
    st.markdown("### 💡 Tips")
    st.info("""
    - Los PDFs deben contener texto seleccionable (no escaneados)
    - Tamaño máximo: 50MB por archivo
    - Formatos soportados: PDF
    - Los datos se procesan con caché para mayor velocidad
    """)

    # Mostrar estadísticas de sesión
    if st.session_state.uploaded_pdfs:
        st.markdown("---")
        st.markdown("### 📈 Estadísticas de Sesión")
        st.metric("PDFs Cargados", len(st.session_state.uploaded_pdfs))
        st.metric("Textos Procesados", len(st.session_state.processed_texts))
        st.metric("Análisis Realizados", len(st.session_state.frequency_analyses))
