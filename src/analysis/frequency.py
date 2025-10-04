"""
Análisis de frecuencias: TF-IDF, n-grams, colocaciones.

CONTRATO:
- Input: ProcessedText o List[ProcessedText]
- Output: FrequencyAnalysis object
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Usaremos custom
import pandas as pd
import numpy as np
from typing import List, Dict
from src.utils.schemas import ProcessedText, FrequencyAnalysis
from src.utils.config import load_config
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """Analiza frecuencias de términos en documentos"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.freq_config = self.config['frequency']

    def analyze_single(self, processed: ProcessedText) -> FrequencyAnalysis:
        """Analiza un solo documento"""
        return self.analyze_multiple([processed])[0]

    def analyze_multiple(self, documents: List[ProcessedText]) -> List[FrequencyAnalysis]:
        """
        Analiza múltiples documentos con TF-IDF.

        Args:
            documents: Lista de ProcessedText objects

        Returns:
            Lista de FrequencyAnalysis objects (uno por documento)
        """
        # Preparar corpus
        corpus = [doc.clean_text for doc in documents]
        doc_ids = [doc.filename for doc in documents]

        # Ajustar min_df y max_df para corpus pequeños
        # min_df debe ser como máximo max_df * n_docs para evitar conflictos
        if len(documents) == 1:
            min_df = 1
            max_df = 1.0
        else:
            max_df = self.freq_config['max_df']
            # Calcular el max_doc_count efectivo
            max_doc_count = int(max_df * len(documents)) if isinstance(max_df, float) else max_df
            # min_df no puede ser mayor que max_doc_count
            min_df = min(self.freq_config['min_df'], max(1, max_doc_count))

        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer(
            max_features=self.freq_config['tfidf_max_features'],
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 1)  # Solo unigrams para TF-IDF
        )

        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Frecuencias simples - usar el mismo vocabulario que TF-IDF
            count_vectorizer = CountVectorizer(
                vocabulary=tfidf_vectorizer.vocabulary_
            )
            count_matrix = count_vectorizer.fit_transform(corpus)
        except ValueError as e:
            # Manejar documentos vacíos o solo con stopwords
            logger.warning(f"Error al procesar corpus: {e}. Retornando resultados vacíos.")
            return [FrequencyAnalysis(
                document_id=doc.filename,
                term_frequencies=pd.DataFrame(columns=['term', 'frequency', 'tfidf']),
                ngrams={n: pd.DataFrame(columns=['ngram', 'frequency'])
                        for n in range(self.freq_config['ngram_range'][0],
                                       self.freq_config['ngram_range'][1] + 1)},
                top_terms=[],
                vocabulary_size=0,
                analysis_date=datetime.now()
            ) for doc in documents]

        # Crear FrequencyAnalysis para cada documento
        results = []

        for idx, doc in enumerate(documents):
            # Extraer TF-IDF para este documento
            doc_tfidf = tfidf_matrix[idx].toarray().flatten()
            doc_counts = count_matrix[idx].toarray().flatten()

            # DataFrame de términos
            term_df = pd.DataFrame({
                'term': feature_names,
                'frequency': doc_counts,
                'tfidf': doc_tfidf
            })

            # Ordenar por TF-IDF
            term_df = term_df.sort_values('tfidf', ascending=False)

            # Top términos
            top_terms = term_df.nlargest(
                self.freq_config['top_n_terms'],
                'tfidf'
            )['term'].tolist()

            # N-grams
            ngrams = self._extract_ngrams(doc.clean_text)

            results.append(FrequencyAnalysis(
                document_id=doc.filename,
                term_frequencies=term_df,
                ngrams=ngrams,
                top_terms=top_terms,
                vocabulary_size=len([f for f in doc_counts if f > 0]),
                analysis_date=datetime.now()
            ))

        return results

    def _extract_ngrams(self, text: str) -> Dict[int, pd.DataFrame]:
        """Extrae n-grams (1, 2, 3)"""
        ngram_results = {}

        min_n, max_n = self.freq_config['ngram_range']

        for n in range(min_n, max_n + 1):
            vectorizer = CountVectorizer(
                ngram_range=(n, n),
                max_features=50  # Top 50 por cada n
            )

            try:
                matrix = vectorizer.fit_transform([text])
                features = vectorizer.get_feature_names_out()
                counts = matrix.toarray().flatten()

                df = pd.DataFrame({
                    'ngram': features,
                    'frequency': counts
                }).sort_values('frequency', ascending=False)

                ngram_results[n] = df
            except ValueError:
                # No hay suficiente texto para este n-gram
                ngram_results[n] = pd.DataFrame(columns=['ngram', 'frequency'])

        return ngram_results

    def compare_documents(self, analyses: List[FrequencyAnalysis]) -> pd.DataFrame:
        """
        Compara términos entre documentos.

        Returns:
            DataFrame con términos en filas, documentos en columnas
        """
        # Obtener todos los términos únicos
        all_terms = set()
        for analysis in analyses:
            all_terms.update(analysis.term_frequencies['term'])

        # Crear matriz documento-término
        data = {}
        for analysis in analyses:
            term_dict = dict(zip(
                analysis.term_frequencies['term'],
                analysis.term_frequencies['tfidf']
            ))
            data[analysis.document_id] = [
                term_dict.get(term, 0.0) for term in all_terms
            ]

        return pd.DataFrame(data, index=list(all_terms))

    def get_cooccurrences(self, processed: ProcessedText,
                          window_size: int = 5) -> pd.DataFrame:
        """
        Calcula co-ocurrencias de términos (palabras que aparecen juntas).

        Args:
            processed: ProcessedText object
            window_size: Ventana para considerar co-ocurrencia

        Returns:
            DataFrame con pares de términos y sus frecuencias
        """
        from collections import defaultdict

        tokens = processed.tokens
        cooccur = defaultdict(int)

        for i, token1 in enumerate(tokens):
            # Ventana de contexto
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)

            for j in range(start, end):
                if i != j:
                    token2 = tokens[j]
                    # Ordenar alfabéticamente para evitar duplicados
                    pair = tuple(sorted([token1, token2]))
                    cooccur[pair] += 1

        # Convertir a DataFrame
        df = pd.DataFrame([
            {'term1': pair[0], 'term2': pair[1], 'frequency': count}
            for pair, count in cooccur.items()
        ]).sort_values('frequency', ascending=False)

        return df
