"""
Extractor de texto de PDFs usando pdfplumber.

CONTRATO:
- Input: archivo PDF (bytes o path)
- Output: ExtractedText object
- Maneja: PDFs con texto, imágenes, tablas
"""
import pdfplumber
from src.utils.schemas import ExtractedText
from datetime import datetime
from typing import Union, BinaryIO
import logging

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extrae texto y metadatos de PDFs"""

    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables

    def extract(self, pdf_source: Union[str, BinaryIO],
                metadata: dict = None) -> ExtractedText:
        """
        Extrae texto completo de un PDF.

        Args:
            pdf_source: Path al PDF o file-like object
            metadata: Metadata adicional (programa, facultad, etc.)

        Returns:
            ExtractedText object con texto y metadata

        Raises:
            ValueError: Si el PDF está corrupto o vacío
        """
        try:
            with pdfplumber.open(pdf_source) as pdf:
                # Extraer texto de todas las páginas
                full_text = []
                tables = []

                for page in pdf.pages:
                    # Texto
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(page_text)

                    # Tablas (opcional)
                    if self.extract_tables:
                        page_tables = page.extract_tables()
                        if page_tables:
                            # Convertir a DataFrame
                            import pandas as pd
                            for table in page_tables:
                                if table:  # Verificar no vacía
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                    tables.append(df)

                # Validar contenido
                combined_text = '\n'.join(full_text)
                if not combined_text.strip():
                    raise ValueError("PDF vacío o sin texto extraíble")

                # Extraer metadata del PDF
                pdf_metadata = pdf.metadata or {}

                # Combinar con metadata provista
                final_metadata = {
                    'pdf_title': pdf_metadata.get('Title', ''),
                    'pdf_author': pdf_metadata.get('Author', ''),
                    'pdf_created': str(pdf_metadata.get('CreationDate', '')),
                    **(metadata or {})
                }

                # Inferir filename
                if isinstance(pdf_source, str):
                    import os
                    filename = os.path.basename(pdf_source)
                else:
                    filename = getattr(pdf_source, 'name', 'unknown.pdf')
                    if hasattr(filename, 'split'):
                        import os
                        filename = os.path.basename(filename)

                return ExtractedText(
                    filename=filename,
                    raw_text=combined_text,
                    metadata=final_metadata,
                    page_count=len(pdf.pages),
                    extraction_date=datetime.now(),
                    has_tables=len(tables) > 0,
                    tables=tables
                )

        except Exception as e:
            logger.error(f"Error extrayendo PDF: {str(e)}")
            raise ValueError(f"No se pudo extraer texto del PDF: {str(e)}")

    def extract_metadata_only(self, pdf_source: Union[str, BinaryIO]) -> dict:
        """Extrae solo metadata sin procesar texto completo"""
        with pdfplumber.open(pdf_source) as pdf:
            return {
                'page_count': len(pdf.pages),
                'metadata': pdf.metadata,
                'has_text': bool(pdf.pages[0].extract_text()) if pdf.pages else False
            }
