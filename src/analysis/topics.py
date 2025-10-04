"""
Topic Modeling con LDA y NMF.

CONTRATO:
- Input: List[ProcessedText]
- Output: TopicModelResult
"""
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
from typing import List, Literal
from src.utils.schemas import ProcessedText, Topic, TopicModelResult
from src.utils.config import load_config
import logging

logger = logging.getLogger(__name__)


class TopicModeler:
    """Descubre topics automaticamente en corpus"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.topic_config = self.config['topics']

    def fit(self,
            documents: List[ProcessedText],
            n_topics: int = None,
            method: Literal['lda', 'nmf'] = 'lda') -> TopicModelResult:
        """
        Entrena modelo de topics.

        Args:
            documents: Lista de documentos procesados
            n_topics: Numero de topics (None = usar config)
            method: 'lda' o 'nmf'

        Returns:
            TopicModelResult con topics y asignaciones
        """
        if n_topics is None:
            n_topics = self.topic_config['default_n_topics']

        # Validar rango
        n_topics = max(
            self.topic_config['min_topics'],
            min(n_topics, self.topic_config['max_topics'])
        )

        # Preparar corpus
        corpus = [doc.clean_text for doc in documents]
        doc_ids = [doc.filename for doc in documents]

        if method == 'lda':
            return self._fit_lda(corpus, doc_ids, n_topics)
        elif method == 'nmf':
            return self._fit_nmf(corpus, doc_ids, n_topics)
        else:
            raise ValueError(f"Metodo no soportado: {method}")

    def _fit_lda(self, corpus: List[str], doc_ids: List[str],
                 n_topics: int) -> TopicModelResult:
        """Latent Dirichlet Allocation"""
        # Vectorizacion con CountVectorizer (LDA necesita frecuencias)
        # Ajustar parámetros para corpus pequeños
        n_docs = len(corpus)
        min_df = 1 if n_docs < 3 else 2
        max_df = 1.0 if n_docs < 3 else 0.8

        vectorizer = CountVectorizer(
            max_features=500,
            min_df=min_df,
            max_df=max_df
        )
        doc_term_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=self.topic_config['lda_iterations'],
            learning_method='online',
            random_state=42,
            n_jobs=1  # Use 1 for Windows compatibility
        )

        doc_topic_matrix = lda.fit_transform(doc_term_matrix)

        # Extraer topics
        topics = []
        for topic_idx, topic_dist in enumerate(lda.components_):
            # Top keywords
            top_indices = topic_dist.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            weights = [topic_dist[i] for i in top_indices]

            # Documentos donde aparece fuerte
            docs_with_topic = [
                doc_ids[i] for i in range(len(doc_ids))
                if doc_topic_matrix[i, topic_idx] > 0.3
            ]

            topics.append(Topic(
                topic_id=topic_idx,
                keywords=keywords,
                weights=weights,
                label=f"Topic {topic_idx}",  # Auto-etiquetar despues
                coherence_score=0.0,  # Calcular despues
                documents=docs_with_topic
            ))

        # Matriz documento-topic
        doc_topic_df = pd.DataFrame(
            doc_topic_matrix,
            index=doc_ids,
            columns=[f"topic_{i}" for i in range(n_topics)]
        )

        # Calcular coherence (simplificado)
        coherence = self._calculate_coherence_simple(lda, doc_term_matrix, feature_names)

        return TopicModelResult(
            model_type='LDA',
            n_topics=n_topics,
            topics=topics,
            document_topic_matrix=doc_topic_df,
            coherence_score=coherence,
            perplexity=lda.perplexity(doc_term_matrix)
        )

    def _fit_nmf(self, corpus: List[str], doc_ids: List[str],
                 n_topics: int) -> TopicModelResult:
        """Non-negative Matrix Factorization"""
        # Vectorizacion con TF-IDF (NMF funciona mejor con TF-IDF)
        # Ajustar parámetros para corpus pequeños
        n_docs = len(corpus)
        min_df = 1 if n_docs < 3 else 2
        max_df = 1.0 if n_docs < 3 else 0.8

        vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=min_df,
            max_df=max_df
        )
        doc_term_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # NMF - adjust n_components if needed for initialization
        n_samples, n_features = doc_term_matrix.shape
        effective_n_topics = min(n_topics, n_samples, n_features)

        # Choose initialization method based on constraints
        init_method = 'nndsvda' if effective_n_topics <= min(n_samples, n_features) else 'random'

        nmf = NMF(
            n_components=effective_n_topics,
            random_state=42,
            max_iter=200,
            init=init_method
        )

        doc_topic_matrix = nmf.fit_transform(doc_term_matrix)

        # Extraer topics (similar a LDA)
        topics = []
        for topic_idx, topic_dist in enumerate(nmf.components_):
            top_indices = topic_dist.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            weights = [topic_dist[i] for i in top_indices]

            docs_with_topic = [
                doc_ids[i] for i in range(len(doc_ids))
                if doc_topic_matrix[i, topic_idx] > 0.3
            ]

            topics.append(Topic(
                topic_id=topic_idx,
                keywords=keywords,
                weights=weights,
                label=f"Topic {topic_idx}",
                coherence_score=0.0,
                documents=docs_with_topic
            ))

        doc_topic_df = pd.DataFrame(
            doc_topic_matrix,
            index=doc_ids,
            columns=[f"topic_{i}" for i in range(effective_n_topics)]
        )

        coherence = self._calculate_coherence_simple(nmf, doc_term_matrix, feature_names)

        return TopicModelResult(
            model_type='NMF',
            n_topics=effective_n_topics,
            topics=topics,
            document_topic_matrix=doc_topic_df,
            coherence_score=coherence,
            perplexity=None  # NMF no tiene perplexity
        )

    def _calculate_coherence_simple(self, model, doc_term_matrix,
                                    feature_names) -> float:
        """
        Coherence score simplificado (UMass).
        Para coherence real, usar gensim.
        """
        # Esta es una version simplificada
        # TODO: Implementar coherence real con gensim si es necesario

        # Por ahora, devolver un score basado en la distribucion
        scores = []
        for topic_dist in model.components_:
            # Entropia normalizada del topic
            topic_dist_norm = topic_dist / topic_dist.sum()
            entropy = -np.sum(topic_dist_norm * np.log(topic_dist_norm + 1e-10))
            scores.append(entropy)

        return float(np.mean(scores))

    def auto_label_topics(self, result: TopicModelResult) -> TopicModelResult:
        """
        Etiqueta topics automaticamente basado en keywords.
        Usa heuristicas simples.
        """
        for topic in result.topics:
            # Top 3 keywords
            top_3 = ' + '.join(topic.keywords[:3])
            topic.label = top_3.title()

        return result

    def find_optimal_topics(self, documents: List[ProcessedText],
                            min_topics: int = None,
                            max_topics: int = None,
                            method: str = 'lda') -> pd.DataFrame:
        """
        Encuentra numero optimo de topics probando multiples valores.

        Returns:
            DataFrame con n_topics, coherence_score, perplexity (si LDA)
        """
        if min_topics is None:
            min_topics = self.topic_config['min_topics']
        if max_topics is None:
            max_topics = self.topic_config['max_topics']

        results = []

        for n in range(min_topics, max_topics + 1):
            logger.info(f"Probando {n} topics...")
            result = self.fit(documents, n_topics=n, method=method)

            results.append({
                'n_topics': n,
                'coherence': result.coherence_score,
                'perplexity': result.perplexity
            })

        return pd.DataFrame(results)
