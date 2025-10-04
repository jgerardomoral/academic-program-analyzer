"""
Página principal de la aplicación Streamlit.
"""
import streamlit as st
import sys
from pathlib import Path

# Agregar directorios al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "app"))

# Ahora importar desde utils
from utils.session_manager import init_session_state


# Configuración de página
st.set_page_config(
    page_title="Análisis de Programas de Estudio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
init_session_state()

# CSS Custom
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎓 Análisis de Programas de Estudio</div>',
            unsafe_allow_html=True)

# Descripción
st.markdown("""
### Bienvenido al Sistema de Análisis Curricular

Esta aplicación te permite analizar programas de estudio universitarios para identificar:
- **Términos más frecuentes** y relevantes
- **Topics temáticos** automáticos
- **Habilidades trabajadas** en cada programa
- **Comparativas** entre programas

#### 🚀 Cómo empezar:
1. Ve a **📁 Subir PDFs** para cargar tus documentos
2. Explora **📊 Análisis de Frecuencias** para ver términos principales
3. Descubre **🎯 Topics y Habilidades** identificados automáticamente
4. Compara programas en **📈 Comparativa**
""")

# Estadísticas actuales
st.markdown("---")
st.subheader("📊 Estado Actual del Sistema")

col1, col2, col3, col4 = st.columns(4)

with col1:
    n_pdfs = len(st.session_state.uploaded_pdfs)
    st.metric("PDFs Cargados", n_pdfs, delta=None)

with col2:
    n_processed = len(st.session_state.processed_texts)
    st.metric("Documentos Procesados", n_processed)

with col3:
    n_analyzed = len(st.session_state.frequency_analyses)
    st.metric("Análisis Realizados", n_analyzed)

with col4:
    has_topics = st.session_state.topic_model is not None
    st.metric("Modelo de Topics", "✅" if has_topics else "❌")

# Información adicional
if n_pdfs == 0:
    st.info("👈 Comienza subiendo PDFs en la sección **📁 Subir PDFs** del menú lateral")
else:
    st.success(f"✅ Tienes {n_pdfs} PDFs listos para analizar")

    # Mostrar lista de documentos
    with st.expander("📄 Ver documentos cargados"):
        for filename in st.session_state.uploaded_pdfs.keys():
            status = "✅ Procesado" if filename in st.session_state.processed_texts else "⏳ Pendiente"
            st.write(f"- {filename} - {status}")

# Quick Actions
st.markdown("---")
st.subheader("⚡ Acciones Rápidas")

col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🔄 Recargar Configuración", use_container_width=True):
        from src.utils.config import load_config
        st.session_state.config = load_config()
        st.success("Configuración recargada")
        st.rerun()

with col_b:
    if st.button("🗑️ Limpiar Caché", use_container_width=True):
        st.cache_data.clear()
        st.success("Caché limpiado")

with col_c:
    if st.button("❌ Reset Completo", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Sistema reiniciado")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    💡 Desarrollado con Streamlit | Python | spaCy | scikit-learn
</div>
""", unsafe_allow_html=True)
