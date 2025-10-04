"""
Utilidades para exportar datos y resultados de análisis a diferentes formatos.
Soporta exportación a Excel, CSV, JSON y HTML para uso con st.download_button.
"""
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import json
import io


def export_to_excel(data: pd.DataFrame, filename: str) -> bytes:
    """
    Exporta un DataFrame a formato Excel (.xlsx).

    Args:
        data: DataFrame de pandas a exportar
        filename: Nombre sugerido para el archivo (solo para referencia)

    Returns:
        Bytes del archivo Excel que pueden usarse con st.download_button

    Example:
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> excel_bytes = export_to_excel(df, 'results.xlsx')
        >>> st.download_button('Download', excel_bytes, 'results.xlsx')
    """
    output = io.BytesIO()

    # Crear writer de Excel con xlsxwriter engine
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Datos')

        # Obtener el workbook y worksheet para formatear
        workbook = writer.book
        worksheet = writer.sheets['Datos']

        # Formato para el encabezado
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#1f77b4',
            'font_color': 'white',
            'border': 1
        })

        # Aplicar formato al encabezado
        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num, value, header_format)

        # Ajustar ancho de columnas
        for i, col in enumerate(data.columns):
            max_length = max(
                data[col].astype(str).apply(len).max(),
                len(str(col))
            )
            worksheet.set_column(i, i, min(max_length + 2, 50))

    output.seek(0)
    return output.getvalue()


def export_to_csv(data: pd.DataFrame, filename: str, encoding: str = 'utf-8') -> bytes:
    """
    Exporta un DataFrame a formato CSV.

    Args:
        data: DataFrame de pandas a exportar
        filename: Nombre sugerido para el archivo (solo para referencia)
        encoding: Codificación del archivo (default: 'utf-8')

    Returns:
        Bytes del archivo CSV que pueden usarse con st.download_button

    Example:
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> csv_bytes = export_to_csv(df, 'results.csv')
        >>> st.download_button('Download', csv_bytes, 'results.csv')
    """
    output = io.StringIO()
    data.to_csv(output, index=False, encoding=encoding)
    output.seek(0)
    return output.getvalue().encode(encoding)


def export_to_json(data: Union[pd.DataFrame, Dict, List], filename: str, indent: int = 2) -> bytes:
    """
    Exporta datos a formato JSON.

    Args:
        data: DataFrame, diccionario o lista a exportar
        filename: Nombre sugerido para el archivo (solo para referencia)
        indent: Nivel de indentación para el JSON (default: 2)

    Returns:
        Bytes del archivo JSON que pueden usarse con st.download_button

    Example:
        >>> data = {'key': 'value', 'list': [1, 2, 3]}
        >>> json_bytes = export_to_json(data, 'results.json')
        >>> st.download_button('Download', json_bytes, 'results.json')
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient='records')

    json_str = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    return json_str.encode('utf-8')


def export_report_html(analysis_results: Dict[str, Any], filename: str) -> bytes:
    """
    Genera un reporte HTML completo con los resultados de análisis.

    Args:
        analysis_results: Diccionario con resultados de análisis que puede incluir:
            - 'document_name': Nombre del documento
            - 'analysis_date': Fecha de análisis
            - 'frequency_analysis': Resultados de análisis de frecuencias
            - 'top_terms': Lista de términos principales
            - 'topics': Resultados de topic modeling
            - 'skills': Perfiles de habilidades
            - 'statistics': Estadísticas generales
        filename: Nombre sugerido para el archivo (solo para referencia)

    Returns:
        Bytes del archivo HTML que pueden usarse con st.download_button

    Example:
        >>> results = {
        ...     'document_name': 'programa.pdf',
        ...     'top_terms': ['matemáticas', 'análisis', 'cálculo']
        ... }
        >>> html_bytes = export_report_html(results, 'reporte.html')
        >>> st.download_button('Download', html_bytes, 'reporte.html')
    """
    # Extraer datos
    doc_name = analysis_results.get('document_name', 'Documento sin nombre')
    analysis_date = analysis_results.get('analysis_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    top_terms = analysis_results.get('top_terms', [])
    statistics = analysis_results.get('statistics', {})
    frequency_df = analysis_results.get('frequency_analysis')
    topics = analysis_results.get('topics', [])
    skills = analysis_results.get('skills', [])

    # Construir HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte de Análisis - {doc_name}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .meta-info {{
                margin-top: 10px;
                opacity: 0.9;
            }}
            .section {{
                background: white;
                padding: 25px;
                margin-bottom: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h2 {{
                color: #1f77b4;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #1f77b4;
            }}
            .stat-label {{
                font-size: 0.9em;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #1f77b4;
                margin-top: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #1f77b4;
                color: white;
                font-weight: 600;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .term-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
            }}
            .term-tag {{
                background: #e7f3ff;
                color: #0066cc;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 500;
                border: 1px solid #0066cc;
            }}
            .topic-box {{
                background: #f8f9fa;
                padding: 15px;
                margin: 15px 0;
                border-radius: 8px;
                border-left: 4px solid #2ca02c;
            }}
            .topic-title {{
                font-weight: bold;
                color: #2ca02c;
                margin-bottom: 10px;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Reporte de Análisis de Programa</h1>
            <div class="meta-info">
                <strong>Documento:</strong> {doc_name}<br>
                <strong>Fecha de análisis:</strong> {analysis_date}
            </div>
        </div>
    """

    # Estadísticas generales
    if statistics:
        html_content += """
        <div class="section">
            <h2>Estadísticas Generales</h2>
            <div class="stats-grid">
        """
        for key, value in statistics.items():
            label = key.replace('_', ' ').title()
            html_content += f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                </div>
            """
        html_content += """
            </div>
        </div>
        """

    # Términos principales
    if top_terms:
        html_content += """
        <div class="section">
            <h2>Términos Principales</h2>
            <div class="term-list">
        """
        for term in top_terms[:20]:  # Mostrar top 20
            html_content += f'<span class="term-tag">{term}</span>'
        html_content += """
            </div>
        </div>
        """

    # Análisis de frecuencias
    if frequency_df is not None and isinstance(frequency_df, pd.DataFrame):
        html_content += """
        <div class="section">
            <h2>Análisis de Frecuencias</h2>
        """
        html_content += frequency_df.head(20).to_html(classes='table', index=False)
        html_content += """
        </div>
        """

    # Topics
    if topics:
        html_content += """
        <div class="section">
            <h2>Tópicos Identificados</h2>
        """
        for i, topic in enumerate(topics[:10], 1):  # Mostrar top 10 topics
            keywords = ', '.join(topic.get('keywords', [])[:10])
            html_content += f"""
            <div class="topic-box">
                <div class="topic-title">Tópico {i}: {topic.get('label', 'Sin etiqueta')}</div>
                <div><strong>Palabras clave:</strong> {keywords}</div>
            </div>
            """
        html_content += """
        </div>
        """

    # Habilidades
    if skills:
        html_content += """
        <div class="section">
            <h2>Habilidades Identificadas</h2>
            <table>
                <thead>
                    <tr>
                        <th>Habilidad</th>
                        <th>Categoría</th>
                        <th>Puntaje</th>
                        <th>Confianza</th>
                    </tr>
                </thead>
                <tbody>
        """
        for skill in skills[:15]:  # Mostrar top 15 habilidades
            html_content += f"""
                <tr>
                    <td>{skill.get('skill_name', 'N/A')}</td>
                    <td>{skill.get('category', 'N/A')}</td>
                    <td>{skill.get('score', 0):.2f}</td>
                    <td>{skill.get('confidence', 0):.2f}</td>
                </tr>
            """
        html_content += """
                </tbody>
            </table>
        </div>
        """

    # Footer
    html_content += """
        <div class="footer">
            <p>Reporte generado por el Sistema de Análisis de Programas de Estudio</p>
            <p>© 2025 - Todos los derechos reservados</p>
        </div>
    </body>
    </html>
    """

    return html_content.encode('utf-8')


def export_multiple_to_excel(
    dataframes: Dict[str, pd.DataFrame],
    filename: str
) -> bytes:
    """
    Exporta múltiples DataFrames a un archivo Excel con múltiples hojas.

    Args:
        dataframes: Diccionario donde la clave es el nombre de la hoja y el valor es el DataFrame
        filename: Nombre sugerido para el archivo (solo para referencia)

    Returns:
        Bytes del archivo Excel que pueden usarse con st.download_button

    Example:
        >>> dfs = {
        ...     'Frecuencias': freq_df,
        ...     'Habilidades': skills_df,
        ...     'Estadísticas': stats_df
        ... }
        >>> excel_bytes = export_multiple_to_excel(dfs, 'analisis_completo.xlsx')
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Formato para encabezados
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#1f77b4',
            'font_color': 'white',
            'border': 1
        })

        for sheet_name, df in dataframes.items():
            # Limpiar nombre de hoja (Excel tiene límites)
            clean_sheet_name = sheet_name[:31]  # Max 31 caracteres
            df.to_excel(writer, sheet_name=clean_sheet_name, index=False)

            # Obtener worksheet para formatear
            worksheet = writer.sheets[clean_sheet_name]

            # Aplicar formato al encabezado
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Ajustar ancho de columnas
            for i, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.set_column(i, i, min(max_length + 2, 50))

    output.seek(0)
    return output.getvalue()


def export_comparison_report(
    documents: List[str],
    comparison_data: Dict[str, Any],
    filename: str
) -> bytes:
    """
    Genera un reporte HTML de comparación entre múltiples documentos.

    Args:
        documents: Lista de nombres de documentos comparados
        comparison_data: Diccionario con datos de comparación
        filename: Nombre sugerido para el archivo

    Returns:
        Bytes del archivo HTML que pueden usarse con st.download_button
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Reporte Comparativo</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            h1 {{ margin: 0; }}
            .section {{
                background: white;
                padding: 25px;
                margin-bottom: 25px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h2 {{
                color: #1f77b4;
                border-bottom: 3px solid #1f77b4;
                padding-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #1f77b4;
                color: white;
            }}
            .doc-list {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
            }}
            .doc-tag {{
                background: #e7f3ff;
                color: #0066cc;
                padding: 8px 16px;
                border-radius: 20px;
                border: 1px solid #0066cc;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Reporte Comparativo de Programas</h1>
            <p>Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="section">
            <h2>Documentos Comparados</h2>
            <div class="doc-list">
    """

    for doc in documents:
        html_content += f'<span class="doc-tag">{doc}</span>'

    html_content += """
            </div>
        </div>
    """

    # Agregar datos de comparación
    if 'common_terms' in comparison_data:
        html_content += """
        <div class="section">
            <h2>Términos Comunes</h2>
            <p>Términos que aparecen en todos los documentos:</p>
            <div class="doc-list">
        """
        for term in comparison_data['common_terms'][:30]:
            html_content += f'<span class="doc-tag">{term}</span>'
        html_content += """
            </div>
        </div>
        """

    if 'statistics' in comparison_data and isinstance(comparison_data['statistics'], pd.DataFrame):
        html_content += """
        <div class="section">
            <h2>Estadísticas Comparativas</h2>
        """
        html_content += comparison_data['statistics'].to_html(classes='table')
        html_content += """
        </div>
        """

    html_content += """
        <div style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
            <p>Reporte generado por el Sistema de Análisis de Programas de Estudio</p>
        </div>
    </body>
    </html>
    """

    return html_content.encode('utf-8')
