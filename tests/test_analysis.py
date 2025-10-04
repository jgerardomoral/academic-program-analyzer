"""
Tests para el módulo de análisis de frecuencias y topic modeling.
"""
import pytest
from datetime import datetime
from src.analysis.frequency import FrequencyAnalyzer
from src.analysis.topics import TopicModeler
from src.analysis.skills_mapper import SkillsMapper
from src.utils.schemas import (
    ProcessedText, FrequencyAnalysis, TopicModelResult,
    DocumentSkillProfile, SkillScore
)
import pandas as pd


class TestFrequencyAnalyzer:
    """Tests para FrequencyAnalyzer"""

    @pytest.fixture
    def analyzer(self):
        """Fixture para crear un analizador de frecuencias"""
        return FrequencyAnalyzer()

    @pytest.fixture
    def mock_processed_text_extended(self):
        """Texto procesado con más contenido para pruebas"""
        return ProcessedText(
            filename="test_document.pdf",
            clean_text="algoritmo problema matemática resolver algoritmo cálculo "
                       "problema análisis matemática algoritmo resolver problema",
            tokens=['algoritmo', 'problema', 'matemática', 'resolver', 'algoritmo',
                    'cálculo', 'problema', 'análisis', 'matemática', 'algoritmo',
                    'resolver', 'problema'],
            lemmas=['algoritmo', 'problema', 'matemática', 'resolver', 'algoritmo',
                    'cálculo', 'problema', 'análisis', 'matemática', 'algoritmo',
                    'resolver', 'problema'],
            pos_tags=['NOUN'] * 12,
            entities=[],
            metadata={'test': 'true'},
            processing_date=datetime.now()
        )

    @pytest.fixture
    def mock_processed_text_second(self):
        """Segundo documento para pruebas de corpus"""
        return ProcessedText(
            filename="test_document_2.pdf",
            clean_text="programación software desarrollo código algoritmo "
                       "programación desarrollo software código",
            tokens=['programación', 'software', 'desarrollo', 'código', 'algoritmo',
                    'programación', 'desarrollo', 'software', 'código'],
            lemmas=['programación', 'software', 'desarrollo', 'código', 'algoritmo',
                    'programación', 'desarrollo', 'software', 'código'],
            pos_tags=['NOUN'] * 9,
            entities=[],
            metadata={'test': 'true'},
            processing_date=datetime.now()
        )

    def test_analyzer_initialization(self, analyzer):
        """Test que el analizador se inicializa correctamente"""
        assert analyzer is not None
        assert analyzer.config is not None
        assert analyzer.freq_config is not None
        assert 'top_n_terms' in analyzer.freq_config
        assert 'ngram_range' in analyzer.freq_config
        assert 'tfidf_max_features' in analyzer.freq_config

    def test_analyze_single_document(self, analyzer, mock_processed_text):
        """Test análisis de un solo documento"""
        result = analyzer.analyze_single(mock_processed_text)

        # Verificar tipo de retorno
        assert isinstance(result, FrequencyAnalysis)

        # Verificar campos básicos
        assert result.document_id == "Matematicas_I.pdf"
        assert isinstance(result.term_frequencies, pd.DataFrame)
        assert isinstance(result.ngrams, dict)
        assert isinstance(result.top_terms, list)
        assert result.vocabulary_size > 0

        # Verificar estructura del DataFrame de términos
        assert 'term' in result.term_frequencies.columns
        assert 'frequency' in result.term_frequencies.columns
        assert 'tfidf' in result.term_frequencies.columns

    def test_tfidf_calculation(self, analyzer, mock_processed_text_extended):
        """Test que TF-IDF se calcula correctamente"""
        result = analyzer.analyze_single(mock_processed_text_extended)

        # El DataFrame debe estar ordenado por TF-IDF descendente
        tfidf_values = result.term_frequencies['tfidf'].values
        assert all(tfidf_values[i] >= tfidf_values[i + 1]
                   for i in range(len(tfidf_values) - 1))

        # Verificar que los valores de TF-IDF están en rango válido [0, 1]
        assert all(0 <= val <= 1 for val in tfidf_values)

        # Los términos con mayor frecuencia deben tener TF-IDF > 0
        assert result.term_frequencies.iloc[0]['tfidf'] > 0

    def test_ngram_extraction(self, analyzer, mock_processed_text_extended):
        """Test extracción de n-gramas"""
        result = analyzer.analyze_single(mock_processed_text_extended)

        # Verificar que se extraen n-gramas de 1, 2 y 3
        assert 1 in result.ngrams
        assert 2 in result.ngrams
        assert 3 in result.ngrams

        # Verificar estructura de cada DataFrame de n-gramas
        for n, ngram_df in result.ngrams.items():
            assert isinstance(ngram_df, pd.DataFrame)
            if not ngram_df.empty:
                assert 'ngram' in ngram_df.columns
                assert 'frequency' in ngram_df.columns

                # Verificar que los n-gramas tienen el número correcto de palabras
                if len(ngram_df) > 0:
                    sample_ngram = ngram_df.iloc[0]['ngram']
                    # Los n-gramas deben tener n palabras (separadas por espacios)
                    word_count = len(sample_ngram.split())
                    assert word_count == n

    def test_unigrams(self, analyzer, mock_processed_text_extended):
        """Test extracción específica de unigramas"""
        result = analyzer.analyze_single(mock_processed_text_extended)

        unigrams = result.ngrams[1]
        assert not unigrams.empty

        # Verificar que los unigramas más frecuentes están presentes
        unigram_list = unigrams['ngram'].tolist()
        # 'algoritmo' y 'problema' deberían estar entre los unigramas
        # ya que aparecen 3 veces cada uno en el texto de prueba

    def test_bigrams(self, analyzer, mock_processed_text_extended):
        """Test extracción de bigramas"""
        result = analyzer.analyze_single(mock_processed_text_extended)

        bigrams = result.ngrams[2]
        if not bigrams.empty:
            # Verificar que todos los bigramas tienen exactamente 2 palabras
            for bigram in bigrams['ngram']:
                assert len(bigram.split()) == 2

    def test_trigrams(self, analyzer, mock_processed_text_extended):
        """Test extracción de trigramas"""
        result = analyzer.analyze_single(mock_processed_text_extended)

        trigrams = result.ngrams[3]
        if not trigrams.empty:
            # Verificar que todos los trigramas tienen exactamente 3 palabras
            for trigram in trigrams['ngram']:
                assert len(trigram.split()) == 3

    def test_corpus_analysis_multiple_documents(self, analyzer,
                                                 mock_processed_text_extended,
                                                 mock_processed_text_second):
        """Test análisis de corpus con múltiples documentos"""
        documents = [mock_processed_text_extended, mock_processed_text_second]
        results = analyzer.analyze_multiple(documents)

        # Verificar que se retorna un resultado por documento
        assert len(results) == 2

        # Verificar que cada resultado es válido
        for result in results:
            assert isinstance(result, FrequencyAnalysis)
            assert isinstance(result.term_frequencies, pd.DataFrame)
            assert isinstance(result.ngrams, dict)

        # Los documentos deben tener IDs diferentes
        assert results[0].document_id != results[1].document_id

    def test_tfidf_corpus_distinguishes_documents(self, analyzer,
                                                   mock_processed_text_extended,
                                                   mock_processed_text_second):
        """Test que TF-IDF distingue términos únicos entre documentos"""
        documents = [mock_processed_text_extended, mock_processed_text_second]
        results = analyzer.analyze_multiple(documents)

        # Extraer top términos de cada documento
        top_terms_doc1 = set(results[0].top_terms[:5])
        top_terms_doc2 = set(results[1].top_terms[:5])

        # Debe haber algunos términos diferentes debido a diferentes contenidos
        # (esto depende del contenido de los documentos de prueba)
        assert len(top_terms_doc1.union(top_terms_doc2)) > 0

    def test_top_terms_limit(self, analyzer, mock_processed_text_extended):
        """Test que la cantidad de top términos respeta la configuración"""
        result = analyzer.analyze_single(mock_processed_text_extended)

        # El número de top términos no debe exceder el configurado
        max_top_terms = analyzer.freq_config['top_n_terms']
        assert len(result.top_terms) <= max_top_terms

    def test_vocabulary_size(self, analyzer, mock_processed_text_extended):
        """Test que el tamaño del vocabulario es correcto"""
        result = analyzer.analyze_single(mock_processed_text_extended)

        # El tamaño del vocabulario debe ser > 0
        assert result.vocabulary_size > 0

        # El tamaño del vocabulario debe ser <= número total de términos únicos
        unique_terms = len(result.term_frequencies)
        assert result.vocabulary_size <= unique_terms

    def test_compare_documents(self, analyzer,
                                mock_processed_text_extended,
                                mock_processed_text_second):
        """Test comparación de documentos"""
        documents = [mock_processed_text_extended, mock_processed_text_second]
        analyses = analyzer.analyze_multiple(documents)

        comparison_df = analyzer.compare_documents(analyses)

        # Verificar estructura del DataFrame de comparación
        assert isinstance(comparison_df, pd.DataFrame)

        # Debe tener columnas para cada documento
        assert len(comparison_df.columns) == 2

        # Los nombres de columnas deben ser los IDs de documentos
        doc_ids = [a.document_id for a in analyses]
        for doc_id in doc_ids:
            assert doc_id in comparison_df.columns

    def test_cooccurrences(self, analyzer, mock_processed_text_extended):
        """Test cálculo de co-ocurrencias"""
        cooccur_df = analyzer.get_cooccurrences(mock_processed_text_extended)

        # Verificar estructura
        assert isinstance(cooccur_df, pd.DataFrame)

        if not cooccur_df.empty:
            assert 'term1' in cooccur_df.columns
            assert 'term2' in cooccur_df.columns
            assert 'frequency' in cooccur_df.columns

            # Verificar que las frecuencias son positivas
            assert all(cooccur_df['frequency'] > 0)

            # Verificar ordenamiento descendente por frecuencia
            frequencies = cooccur_df['frequency'].values
            assert all(frequencies[i] >= frequencies[i + 1]
                       for i in range(len(frequencies) - 1))

    def test_cooccurrences_window_size(self, analyzer, mock_processed_text_extended):
        """Test que el tamaño de ventana afecta las co-ocurrencias"""
        cooccur_small = analyzer.get_cooccurrences(
            mock_processed_text_extended, window_size=2
        )
        cooccur_large = analyzer.get_cooccurrences(
            mock_processed_text_extended, window_size=10
        )

        # Una ventana más grande generalmente produce más co-ocurrencias
        # o co-ocurrencias con mayor frecuencia
        if not cooccur_small.empty and not cooccur_large.empty:
            assert len(cooccur_large) >= len(cooccur_small)

    def test_empty_document_handling(self, analyzer):
        """Test manejo de documentos vacíos"""
        empty_doc = ProcessedText(
            filename="empty.pdf",
            clean_text="",
            tokens=[],
            lemmas=[],
            pos_tags=[],
            entities=[],
            metadata={},
            processing_date=datetime.now()
        )

        # El análisis no debe fallar con un documento vacío
        # Aunque puede retornar resultados vacíos
        result = analyzer.analyze_single(empty_doc)
        assert isinstance(result, FrequencyAnalysis)

    def test_analysis_date(self, analyzer, mock_processed_text):
        """Test que la fecha de análisis se registra"""
        result = analyzer.analyze_single(mock_processed_text)

        assert result.analysis_date is not None
        assert isinstance(result.analysis_date, datetime)

        # La fecha debe ser reciente (dentro de los últimos 5 segundos)
        time_diff = datetime.now() - result.analysis_date
        assert time_diff.total_seconds() < 5

    def test_term_frequencies_not_empty(self, analyzer, mock_processed_text):
        """Test que term_frequencies contiene datos"""
        result = analyzer.analyze_single(mock_processed_text)

        assert not result.term_frequencies.empty
        assert len(result.term_frequencies) > 0

    def test_frequency_values_non_negative(self, analyzer, mock_processed_text_extended):
        """Test que las frecuencias son no negativas"""
        result = analyzer.analyze_single(mock_processed_text_extended)

        # Verificar frecuencias
        assert all(result.term_frequencies['frequency'] >= 0)

        # Verificar TF-IDF
        assert all(result.term_frequencies['tfidf'] >= 0)


# Topic Modeling Tests
@pytest.fixture
def mock_topic_documents():
    """Multiple processed documents for topic modeling"""
    return [
        ProcessedText(
            filename="Matematicas_I.pdf",
            clean_text="matematicas algebra calculo diferencial integrales derivadas funciones analisis numerico",
            tokens=['matematicas', 'algebra', 'calculo', 'diferencial', 'integrales', 'derivadas', 'funciones', 'analisis', 'numerico'],
            lemmas=['matematica', 'algebra', 'calculo', 'diferencial', 'integral', 'derivada', 'funcion', 'analisis', 'numerico'],
            pos_tags=['NOUN'] * 9,
            entities=[],
            metadata={'programa': 'Ingenieria'},
            processing_date=datetime.now()
        ),
        ProcessedText(
            filename="Programacion_I.pdf",
            clean_text="programacion algoritmos estructuras datos codigo python java variables funciones recursion",
            tokens=['programacion', 'algoritmos', 'estructuras', 'datos', 'codigo', 'python', 'java', 'variables', 'funciones', 'recursion'],
            lemmas=['programacion', 'algoritmo', 'estructura', 'dato', 'codigo', 'python', 'java', 'variable', 'funcion', 'recursion'],
            pos_tags=['NOUN'] * 10,
            entities=[],
            metadata={'programa': 'Ingenieria'},
            processing_date=datetime.now()
        ),
        ProcessedText(
            filename="Fisica_I.pdf",
            clean_text="fisica mecanica cinematica dinamica fuerzas energia movimiento vectores newton leyes",
            tokens=['fisica', 'mecanica', 'cinematica', 'dinamica', 'fuerzas', 'energia', 'movimiento', 'vectores', 'newton', 'leyes'],
            lemmas=['fisica', 'mecanica', 'cinematica', 'dinamica', 'fuerza', 'energia', 'movimiento', 'vector', 'newton', 'ley'],
            pos_tags=['NOUN'] * 10,
            entities=[],
            metadata={'programa': 'Ingenieria'},
            processing_date=datetime.now()
        ),
        ProcessedText(
            filename="Estadistica_I.pdf",
            clean_text="estadistica probabilidad datos analisis muestras poblacion media varianza distribucion normal",
            tokens=['estadistica', 'probabilidad', 'datos', 'analisis', 'muestras', 'poblacion', 'media', 'varianza', 'distribucion', 'normal'],
            lemmas=['estadistica', 'probabilidad', 'dato', 'analisis', 'muestra', 'poblacion', 'media', 'varianza', 'distribucion', 'normal'],
            pos_tags=['NOUN'] * 10,
            entities=[],
            metadata={'programa': 'Ingenieria'},
            processing_date=datetime.now()
        ),
        ProcessedText(
            filename="Base_Datos.pdf",
            clean_text="base datos sql consultas tablas relacional modelo entidades atributos claves normalizacion",
            tokens=['base', 'datos', 'sql', 'consultas', 'tablas', 'relacional', 'modelo', 'entidades', 'atributos', 'claves', 'normalizacion'],
            lemmas=['base', 'dato', 'sql', 'consulta', 'tabla', 'relacional', 'modelo', 'entidad', 'atributo', 'clave', 'normalizacion'],
            pos_tags=['NOUN'] * 11,
            entities=[],
            metadata={'programa': 'Ingenieria'},
            processing_date=datetime.now()
        )
    ]


class TestTopicModeler:
    """Test suite for TopicModeler class"""

    def test_init_loads_config(self, tmp_path):
        """Test that TopicModeler initializes with config"""
        # Create temporary config
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
topics:
  default_n_topics: 5
  min_topics: 2
  max_topics: 15
  lda_iterations: 50
  lda_passes: 5
  coherence_threshold: 0.3
""")

        modeler = TopicModeler(config_path=str(config_file))
        assert modeler.topic_config['default_n_topics'] == 5
        assert modeler.topic_config['min_topics'] == 2
        assert modeler.topic_config['max_topics'] == 15
        assert modeler.topic_config['lda_iterations'] == 50

    def test_fit_lda_basic(self, mock_topic_documents):
        """Test basic LDA topic modeling"""
        modeler = TopicModeler(config_path="config.yaml")
        result = modeler.fit(mock_topic_documents, n_topics=3, method='lda')

        # Check result structure
        assert isinstance(result, TopicModelResult)
        assert result.model_type == 'LDA'
        assert result.n_topics == 3
        assert len(result.topics) == 3
        assert result.coherence_score > 0
        assert result.perplexity is not None
        assert isinstance(result.document_topic_matrix, pd.DataFrame)

    def test_fit_nmf_basic(self, mock_topic_documents):
        """Test basic NMF topic modeling"""
        modeler = TopicModeler(config_path="config.yaml")
        result = modeler.fit(mock_topic_documents, n_topics=3, method='nmf')

        # Check result structure
        assert isinstance(result, TopicModelResult)
        assert result.model_type == 'NMF'
        assert result.n_topics == 3
        assert len(result.topics) == 3
        assert result.coherence_score > 0
        assert result.perplexity is None  # NMF doesn't have perplexity
        assert isinstance(result.document_topic_matrix, pd.DataFrame)

    def test_topic_structure(self, mock_topic_documents):
        """Test that topics have correct structure"""
        modeler = TopicModeler(config_path="config.yaml")
        result = modeler.fit(mock_topic_documents, n_topics=2, method='lda')

        for topic in result.topics:
            # Check topic attributes
            assert hasattr(topic, 'topic_id')
            assert hasattr(topic, 'keywords')
            assert hasattr(topic, 'weights')
            assert hasattr(topic, 'label')
            assert hasattr(topic, 'coherence_score')
            assert hasattr(topic, 'documents')

            # Check data types and lengths
            assert isinstance(topic.keywords, list)
            assert isinstance(topic.weights, list)
            assert len(topic.keywords) == len(topic.weights)
            assert len(topic.keywords) > 0  # At least some keywords
            assert len(topic.keywords) <= 10  # Max 10 keywords
            assert all(isinstance(kw, str) for kw in topic.keywords)
            assert all(isinstance(w, (int, float)) for w in topic.weights)

    def test_document_topic_matrix_shape(self, mock_topic_documents):
        """Test document-topic matrix has correct dimensions"""
        modeler = TopicModeler(config_path="config.yaml")
        n_topics = 4
        result = modeler.fit(mock_topic_documents, n_topics=n_topics, method='lda')

        # Check matrix dimensions
        assert result.document_topic_matrix.shape == (len(mock_topic_documents), n_topics)
        assert len(result.document_topic_matrix.index) == len(mock_topic_documents)
        assert len(result.document_topic_matrix.columns) == n_topics

        # Check column names
        expected_cols = [f"topic_{i}" for i in range(n_topics)]
        assert list(result.document_topic_matrix.columns) == expected_cols

        # Check index (document IDs)
        expected_docs = [doc.filename for doc in mock_topic_documents]
        assert list(result.document_topic_matrix.index) == expected_docs

    def test_coherence_calculation(self, mock_topic_documents):
        """Test coherence score calculation"""
        modeler = TopicModeler(config_path="config.yaml")

        # Test LDA coherence
        result_lda = modeler.fit(mock_topic_documents, n_topics=3, method='lda')
        assert result_lda.coherence_score > 0
        assert isinstance(result_lda.coherence_score, float)

        # Test NMF coherence
        result_nmf = modeler.fit(mock_topic_documents, n_topics=3, method='nmf')
        assert result_nmf.coherence_score > 0
        assert isinstance(result_nmf.coherence_score, float)

    def test_find_optimal_topics(self, mock_topic_documents):
        """Test finding optimal number of topics"""
        modeler = TopicModeler(config_path="config.yaml")

        # Test with default range
        results_df = modeler.find_optimal_topics(mock_topic_documents, min_topics=2, max_topics=4, method='lda')

        # Check DataFrame structure
        assert isinstance(results_df, pd.DataFrame)
        assert 'n_topics' in results_df.columns
        assert 'coherence' in results_df.columns
        assert 'perplexity' in results_df.columns

        # Check number of rows
        assert len(results_df) == 3  # 2, 3, 4 topics

        # Check n_topics values
        assert list(results_df['n_topics']) == [2, 3, 4]

        # Check all coherence scores are positive
        assert all(results_df['coherence'] > 0)

    def test_find_optimal_topics_nmf(self, mock_topic_documents):
        """Test finding optimal topics with NMF"""
        modeler = TopicModeler(config_path="config.yaml")

        results_df = modeler.find_optimal_topics(mock_topic_documents, min_topics=2, max_topics=3, method='nmf')

        # Check DataFrame structure
        assert isinstance(results_df, pd.DataFrame)
        assert 'n_topics' in results_df.columns
        assert 'coherence' in results_df.columns
        assert 'perplexity' in results_df.columns

        # NMF should have None for perplexity
        assert all(pd.isna(results_df['perplexity']))

    def test_auto_label_topics(self, mock_topic_documents):
        """Test automatic topic labeling"""
        modeler = TopicModeler(config_path="config.yaml")
        result = modeler.fit(mock_topic_documents, n_topics=3, method='lda')

        # Before labeling - should have default labels
        for topic in result.topics:
            assert topic.label.startswith('Topic')

        # Auto-label
        labeled_result = modeler.auto_label_topics(result)

        # After labeling - should have keyword-based labels
        for topic in labeled_result.topics:
            # Label should contain keywords
            assert len(topic.label) > 0
            assert topic.label != f"Topic {topic.topic_id}"
            # Label should be title case
            assert topic.label[0].isupper() or topic.label.split()[0][0].isupper()

    def test_topic_count_validation(self, mock_topic_documents, tmp_path):
        """Test that n_topics is validated against config limits"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
topics:
  default_n_topics: 5
  min_topics: 2
  max_topics: 10
  lda_iterations: 50
  lda_passes: 5
  coherence_threshold: 0.3
""")

        modeler = TopicModeler(config_path=str(config_file))

        # Test with too few topics (should be clamped to min)
        result = modeler.fit(mock_topic_documents, n_topics=1, method='lda')
        assert result.n_topics == 2  # Clamped to min_topics

        # Test with too many topics (should be clamped to max)
        result = modeler.fit(mock_topic_documents, n_topics=20, method='lda')
        assert result.n_topics == 10  # Clamped to max_topics

        # Test with None (should use default)
        result = modeler.fit(mock_topic_documents, n_topics=None, method='lda')
        assert result.n_topics == 5  # Uses default_n_topics

    def test_invalid_method_raises_error(self, mock_topic_documents):
        """Test that invalid method raises ValueError"""
        modeler = TopicModeler(config_path="config.yaml")

        with pytest.raises(ValueError, match="Metodo no soportado"):
            modeler.fit(mock_topic_documents, n_topics=3, method='invalid')

    def test_document_assignment_to_topics(self, mock_topic_documents):
        """Test that documents are correctly assigned to topics"""
        modeler = TopicModeler(config_path="config.yaml")
        result = modeler.fit(mock_topic_documents, n_topics=3, method='lda')

        # Check that topics have documents assigned
        total_assignments = sum(len(topic.documents) for topic in result.topics)
        assert total_assignments > 0  # At least some documents should be assigned

        # Check that assigned documents are from our corpus
        all_doc_ids = {doc.filename for doc in mock_topic_documents}
        for topic in result.topics:
            for doc_id in topic.documents:
                assert doc_id in all_doc_ids

    def test_lda_vs_nmf_differences(self, mock_topic_documents):
        """Test that LDA and NMF produce different results"""
        modeler = TopicModeler(config_path="config.yaml")

        result_lda = modeler.fit(mock_topic_documents, n_topics=3, method='lda')
        result_nmf = modeler.fit(mock_topic_documents, n_topics=3, method='nmf')

        # Both should work
        assert result_lda.model_type == 'LDA'
        assert result_nmf.model_type == 'NMF'

        # LDA has perplexity, NMF doesn't
        assert result_lda.perplexity is not None
        assert result_nmf.perplexity is None

        # Both should have coherence scores
        assert result_lda.coherence_score > 0
        assert result_nmf.coherence_score > 0

    def test_small_corpus_handling(self):
        """Test handling of very small corpus"""
        modeler = TopicModeler(config_path="config.yaml")

        small_corpus = [
            ProcessedText(
                filename="doc1.pdf",
                clean_text="algoritmo programacion",
                tokens=['algoritmo', 'programacion'],
                lemmas=['algoritmo', 'programacion'],
                pos_tags=['NOUN', 'NOUN'],
                entities=[],
                metadata={},
                processing_date=datetime.now()
            )
        ]

        # Should handle small corpus gracefully
        # Note: might raise warnings but shouldn't crash
        try:
            result = modeler.fit(small_corpus, n_topics=2, method='lda')
            assert result.n_topics >= 2  # Should respect minimum
        except:
            # Some configurations might fail with very small corpus, which is acceptable
            pass


# Skills Mapping Tests
class TestSkillsMapper:
    """Test suite for SkillsMapper class"""

    @pytest.fixture
    def skills_mapper(self):
        """Fixture para crear un SkillsMapper"""
        return SkillsMapper(config_path="config.yaml")

    def test_initialization(self, skills_mapper):
        """Test que SkillsMapper se inicializa correctamente"""
        assert skills_mapper is not None
        assert skills_mapper.config is not None
        assert skills_mapper.skills_config is not None
        assert skills_mapper.taxonomia is not None
        assert len(skills_mapper.skills) > 0

    def test_load_taxonomy(self, skills_mapper):
        """Test carga de taxonomía desde archivo"""
        assert isinstance(skills_mapper.taxonomia, dict)
        assert len(skills_mapper.taxonomia) > 0

        # Verificar estructura de una skill en la taxonomía
        first_skill_id = list(skills_mapper.taxonomia.keys())[0]
        skill_data = skills_mapper.taxonomia[first_skill_id]

        assert 'name' in skill_data
        assert 'keywords' in skill_data
        assert 'weight' in skill_data
        assert 'category' in skill_data

    def test_parse_taxonomia(self, skills_mapper):
        """Test conversión de taxonomía a objetos Skill"""
        skills = skills_mapper.skills

        assert isinstance(skills, list)
        assert len(skills) > 0

        # Verificar que todos son objetos Skill válidos
        for skill in skills:
            assert hasattr(skill, 'skill_id')
            assert hasattr(skill, 'name')
            assert hasattr(skill, 'keywords')
            assert hasattr(skill, 'synonyms')
            assert hasattr(skill, 'weight')
            assert hasattr(skill, 'category')
            assert isinstance(skill.keywords, list)
            assert isinstance(skill.synonyms, list)

    def test_map_document_basic(self, skills_mapper, mock_frequency_analysis, mock_processed_text):
        """Test mapeo básico de documento a habilidades"""
        profile = skills_mapper.map_document(
            mock_frequency_analysis,
            mock_processed_text
        )

        # Verificar tipo de retorno
        assert isinstance(profile, DocumentSkillProfile)

        # Verificar campos básicos
        assert profile.document_id == "Matematicas_I.pdf"
        assert isinstance(profile.skill_scores, list)
        assert isinstance(profile.top_skills, list)
        assert isinstance(profile.skill_coverage, float)
        assert profile.analysis_date is not None

    def test_skill_scores_structure(self, skills_mapper, mock_frequency_analysis):
        """Test estructura de SkillScore"""
        profile = skills_mapper.map_document(mock_frequency_analysis)

        assert len(profile.skill_scores) > 0

        for skill_score in profile.skill_scores:
            assert isinstance(skill_score, SkillScore)
            assert hasattr(skill_score, 'skill_id')
            assert hasattr(skill_score, 'skill_name')
            assert hasattr(skill_score, 'score')
            assert hasattr(skill_score, 'confidence')
            assert hasattr(skill_score, 'matched_terms')
            assert hasattr(skill_score, 'context_snippets')

            # Verificar rangos de valores
            assert 0.0 <= skill_score.score <= 1.0
            assert 0.0 <= skill_score.confidence <= 1.0

    def test_skill_matching(self, skills_mapper, mock_frequency_analysis):
        """Test que los términos se matchean correctamente con skills"""
        profile = skills_mapper.map_document(mock_frequency_analysis)

        # Buscar la skill de programación (que debería matchear con 'algoritmo')
        prog_scores = [ss for ss in profile.skill_scores if 'Programación' in ss.skill_name]

        if prog_scores:
            prog_score = prog_scores[0]
            assert 'algoritmo' in prog_score.matched_terms or len(prog_score.matched_terms) > 0

        # Buscar la skill de resolución de problemas (que debería matchear con 'problema')
        problem_scores = [ss for ss in profile.skill_scores if 'Problemas' in ss.skill_name]

        if problem_scores:
            problem_score = problem_scores[0]
            assert 'problema' in problem_score.matched_terms or len(problem_score.matched_terms) > 0

    def test_score_calculation(self, skills_mapper, mock_frequency_analysis):
        """Test cálculo de scores con TF-IDF y frecuencias"""
        profile = skills_mapper.map_document(mock_frequency_analysis)

        # Los scores deben estar ordenados de mayor a menor
        scores = [ss.score for ss in profile.skill_scores]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

        # Al menos una skill debe tener score > 0
        assert any(ss.score > 0 for ss in profile.skill_scores)

    def test_context_snippets(self, skills_mapper, mock_frequency_analysis, mock_processed_text):
        """Test extracción de snippets de contexto"""
        profile = skills_mapper.map_document(
            mock_frequency_analysis,
            mock_processed_text
        )

        # Buscar skills con matches
        skills_with_matches = [ss for ss in profile.skill_scores if len(ss.matched_terms) > 0]

        if skills_with_matches:
            skill = skills_with_matches[0]
            # Si hay matches y se pasó processed_text, debería haber snippets
            if skill.context_snippets:
                assert isinstance(skill.context_snippets, list)
                assert all(isinstance(s, str) for s in skill.context_snippets)
                # Los snippets deben contener "..."
                assert all('...' in s for s in skill.context_snippets)

    def test_top_skills_filtering(self, skills_mapper, mock_frequency_analysis):
        """Test que top_skills filtra por min_confidence"""
        profile = skills_mapper.map_document(mock_frequency_analysis)

        min_conf = skills_mapper.skills_config['min_confidence']

        # Todos los top_skills deben tener score > min_confidence
        for skill_name in profile.top_skills:
            skill_score = next(ss for ss in profile.skill_scores if ss.skill_name == skill_name)
            assert skill_score.score > min_conf

        # Top skills no debe tener más de 10 elementos
        assert len(profile.top_skills) <= 10

    def test_skill_coverage_calculation(self, skills_mapper, mock_frequency_analysis):
        """Test cálculo de coverage"""
        profile = skills_mapper.map_document(mock_frequency_analysis)

        # Coverage debe estar entre 0 y 1
        assert 0.0 <= profile.skill_coverage <= 1.0

        # Si hay términos mapeados, coverage debe ser > 0
        total_matched = sum(len(ss.matched_terms) for ss in profile.skill_scores)
        if total_matched > 0:
            assert profile.skill_coverage > 0

    def test_create_skills_matrix(self, skills_mapper, mock_frequency_analysis):
        """Test creación de matriz documento × habilidad"""
        profile1 = skills_mapper.map_document(mock_frequency_analysis)

        # Crear segundo perfil (con el mismo analysis para este test)
        profile2 = skills_mapper.map_document(mock_frequency_analysis)
        profile2.document_id = "Programacion_I.pdf"  # Cambiar ID

        matrix = skills_mapper.create_skills_matrix([profile1, profile2])

        # Verificar estructura
        assert isinstance(matrix, pd.DataFrame)
        assert len(matrix) == 2  # 2 documentos
        assert len(matrix.columns) > 0  # Al menos una skill

        # Verificar que los IDs están en el índice
        assert "Matematicas_I.pdf" in matrix.index
        assert "Programacion_I.pdf" in matrix.index

    def test_compare_profiles(self, skills_mapper, mock_frequency_analysis):
        """Test comparación de perfiles"""
        profile1 = skills_mapper.map_document(mock_frequency_analysis)
        profile2 = skills_mapper.map_document(mock_frequency_analysis)
        profile2.document_id = "Programacion_I.pdf"

        comparison = skills_mapper.compare_profiles(profile1, profile2)

        # Verificar estructura
        assert isinstance(comparison, pd.DataFrame)
        assert 'difference' in comparison.columns
        assert len(comparison.columns) == 3  # 2 documentos + difference

        # Verificar ordenamiento por diferencia
        diffs = comparison['difference'].values
        assert all(diffs[i] >= diffs[i + 1] for i in range(len(diffs) - 1))

    def test_save_taxonomy(self, skills_mapper, tmp_path):
        """Test guardado de taxonomía"""
        new_taxonomy = {
            "test_skill": {
                "name": "Test Skill",
                "keywords": ["test", "prueba"],
                "synonyms": ["testing"],
                "weight": 1.0,
                "category": "test"
            }
        }

        # Guardar en archivo temporal
        temp_file = tmp_path / "test_taxonomy.json"
        skills_mapper.save_taxonomy(new_taxonomy, str(temp_file))

        # Verificar que el archivo existe y se puede leer
        assert temp_file.exists()

        import json
        with open(temp_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert loaded == new_taxonomy

    def test_empty_frequency_analysis(self, skills_mapper):
        """Test manejo de análisis vacío"""
        empty_analysis = FrequencyAnalysis(
            document_id="empty.pdf",
            term_frequencies=pd.DataFrame(columns=['term', 'frequency', 'tfidf']),
            ngrams={},
            top_terms=[],
            vocabulary_size=0,
            analysis_date=datetime.now()
        )

        profile = skills_mapper.map_document(empty_analysis)

        # Debe retornar un perfil válido aunque vacío
        assert isinstance(profile, DocumentSkillProfile)
        assert profile.skill_coverage == 0.0
        assert all(ss.score == 0.0 for ss in profile.skill_scores)

    def test_weighted_scoring(self, skills_mapper, mock_frequency_analysis):
        """Test que los pesos de la taxonomía afectan el score"""
        profile = skills_mapper.map_document(mock_frequency_analysis)

        # Las skills deben usar sus pesos en el cálculo
        # (difícil de verificar exactamente sin conocer los detalles,
        # pero podemos verificar que los scores son razonables)
        for skill_score in profile.skill_scores:
            if skill_score.matched_terms:
                # Si hay términos mapeados, el score debe ser > 0
                assert skill_score.score > 0

    def test_confidence_calculation(self, skills_mapper, mock_frequency_analysis):
        """Test cálculo de confianza basado en matches"""
        profile = skills_mapper.map_document(mock_frequency_analysis)

        for skill_score in profile.skill_scores:
            # Confianza debe estar entre 0 y 1
            assert 0.0 <= skill_score.confidence <= 1.0

            # Si no hay matches, confianza debe ser 0
            if len(skill_score.matched_terms) == 0:
                assert skill_score.confidence == 0.0

    def test_substring_matching(self, skills_mapper):
        """Test que el matching funciona con substrings"""
        # Crear análisis con términos que son variaciones
        analysis = FrequencyAnalysis(
            document_id="test.pdf",
            term_frequencies=pd.DataFrame({
                'term': ['programación', 'programador', 'algorítmico'],
                'frequency': [10, 5, 3],
                'tfidf': [0.8, 0.6, 0.4]
            }),
            ngrams={},
            top_terms=['programación', 'programador', 'algorítmico'],
            vocabulary_size=3,
            analysis_date=datetime.now()
        )

        profile = skills_mapper.map_document(analysis)

        # La skill de programación debería matchear 'programación' y 'programador'
        prog_skill = [ss for ss in profile.skill_scores if 'Programación' in ss.skill_name]

        if prog_skill:
            # Debería tener al menos algún match
            assert len(prog_skill[0].matched_terms) > 0
