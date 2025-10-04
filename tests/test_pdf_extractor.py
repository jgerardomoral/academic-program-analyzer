"""
Tests para el módulo de extracción de PDF.
"""
import pytest
from src.extraction.pdf_extractor import PDFExtractor
from src.utils.schemas import ExtractedText
from datetime import datetime


def test_extractor_with_mock(tmp_path):
    """Test con PDF mock creado en memoria"""
    from reportlab.pdfgen import canvas

    # Crear PDF simple
    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Test PDF")
    c.drawString(100, 700, "Matemáticas y algoritmos")
    c.save()

    # Extraer
    extractor = PDFExtractor()
    result = extractor.extract(str(pdf_path))

    # Validar contrato
    assert isinstance(result, ExtractedText)
    assert result.filename == "test.pdf"
    assert "Test PDF" in result.raw_text
    assert result.page_count == 1
    assert isinstance(result.extraction_date, datetime)


def test_extractor_handles_empty_pdf(tmp_path):
    """Test que maneja PDFs vacíos correctamente"""
    from reportlab.pdfgen import canvas

    # Crear PDF vacío
    pdf_path = tmp_path / "empty.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.save()

    extractor = PDFExtractor()

    with pytest.raises(ValueError, match="vacío"):
        extractor.extract(str(pdf_path))


def test_extractor_with_custom_metadata(tmp_path):
    """Test que metadata personalizada se incluye en el resultado"""
    from reportlab.pdfgen import canvas

    # Crear PDF simple
    pdf_path = tmp_path / "metadata_test.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Contenido de prueba")
    c.save()

    # Extraer con metadata personalizada
    extractor = PDFExtractor()
    custom_metadata = {
        'programa': 'Ingeniería en Computación',
        'facultad': 'Ciencias',
        'año': '2024'
    }
    result = extractor.extract(str(pdf_path), metadata=custom_metadata)

    # Validar que metadata personalizada está incluida
    assert result.metadata['programa'] == 'Ingeniería en Computación'
    assert result.metadata['facultad'] == 'Ciencias'
    assert result.metadata['año'] == '2024'


def test_extractor_without_tables(tmp_path):
    """Test que extractor sin tablas funciona correctamente"""
    from reportlab.pdfgen import canvas

    # Crear PDF simple
    pdf_path = tmp_path / "no_tables.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Contenido sin tablas")
    c.save()

    # Extraer sin procesar tablas
    extractor = PDFExtractor(extract_tables=False)
    result = extractor.extract(str(pdf_path))

    # Validar
    assert isinstance(result, ExtractedText)
    assert len(result.tables) == 0
    assert result.has_tables is False


def test_extract_metadata_only(tmp_path):
    """Test de extracción solo de metadata"""
    from reportlab.pdfgen import canvas

    # Crear PDF simple
    pdf_path = tmp_path / "metadata_only.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Texto de prueba")
    c.save()

    # Extraer solo metadata
    extractor = PDFExtractor()
    metadata = extractor.extract_metadata_only(str(pdf_path))

    # Validar
    assert 'page_count' in metadata
    assert metadata['page_count'] == 1
    assert 'metadata' in metadata
    assert 'has_text' in metadata
    assert metadata['has_text'] is True


def test_extractor_multipage_pdf(tmp_path):
    """Test con PDF de múltiples páginas"""
    from reportlab.pdfgen import canvas

    # Crear PDF de 3 páginas
    pdf_path = tmp_path / "multipage.pdf"
    c = canvas.Canvas(str(pdf_path))

    # Página 1
    c.drawString(100, 750, "Página 1: Introducción")
    c.showPage()

    # Página 2
    c.drawString(100, 750, "Página 2: Desarrollo")
    c.showPage()

    # Página 3
    c.drawString(100, 750, "Página 3: Conclusión")
    c.save()

    # Extraer
    extractor = PDFExtractor()
    result = extractor.extract(str(pdf_path))

    # Validar
    assert result.page_count == 3
    assert "Página 1" in result.raw_text
    assert "Página 2" in result.raw_text
    assert "Página 3" in result.raw_text


def test_extractor_preserves_pdf_metadata(tmp_path):
    """Test que extrae metadata del PDF"""
    from reportlab.pdfgen import canvas

    # Crear PDF con metadata
    pdf_path = tmp_path / "with_metadata.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.setTitle("Documento de Prueba")
    c.setAuthor("Test Author")
    c.drawString(100, 750, "Contenido")
    c.save()

    # Extraer
    extractor = PDFExtractor()
    result = extractor.extract(str(pdf_path))

    # Validar que metadata del PDF está presente
    assert 'pdf_title' in result.metadata
    assert 'pdf_author' in result.metadata
    assert 'pdf_created' in result.metadata
