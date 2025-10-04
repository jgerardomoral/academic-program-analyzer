"""
Tests para modulo de extraccion y preprocesamiento de texto.
"""
import pytest
from src.extraction.preprocessor import TextPreprocessor
from src.utils.schemas import ProcessedText


class TestTextPreprocessor:
    """Tests para TextPreprocessor"""

    @pytest.fixture
    def preprocessor(self):
        """Fixture para crear un preprocessor"""
        return TextPreprocessor()

    def test_preprocessor_initialization(self, preprocessor):
        """Test que el preprocessor se inicializa correctamente"""
        assert preprocessor is not None
        assert preprocessor.nlp is not None
        assert preprocessor.min_length == 3
        assert preprocessor.max_length == 30
        assert len(preprocessor.pos_keep) == 3
        assert 'NOUN' in preprocessor.pos_keep
        assert 'VERB' in preprocessor.pos_keep
        assert 'ADJ' in preprocessor.pos_keep

    def test_preprocessing_pipeline(self, preprocessor, mock_extracted_text):
        """Test del pipeline completo de preprocesamiento"""
        result = preprocessor.process(mock_extracted_text)

        # Verificar que devuelve un ProcessedText
        assert isinstance(result, ProcessedText)

        # Verificar que tiene tokens
        assert len(result.tokens) > 0

        # Verificar que tokens, lemmas y pos_tags tienen la misma longitud
        assert len(result.tokens) == len(result.lemmas)
        assert len(result.tokens) == len(result.pos_tags)

        # Verificar que el filename se mantiene
        assert result.filename == mock_extracted_text.filename

        # Verificar que los metadata se mantienen
        assert result.metadata == mock_extracted_text.metadata

        # Verificar que el texto limpio no esta vacio
        assert len(result.clean_text) > 0

    def test_stopword_filtering(self, preprocessor, mock_extracted_text):
        """Test que las stopwords se filtran correctamente"""
        result = preprocessor.process(mock_extracted_text)

        # Verificar que las stopwords comunes no estan en los tokens
        common_stopwords = ['de', 'la', 'el', 'en', 'y', 'a']
        for stopword in common_stopwords:
            assert stopword not in result.tokens

        # Verificar que las stopwords custom no estan en los tokens
        custom_stopwords = ['universidad', 'asignatura', 'creditos']
        for stopword in custom_stopwords:
            assert stopword not in result.tokens

    def test_text_cleaning(self, preprocessor):
        """Test de limpieza de texto"""
        # Texto con URLs, emails y multiples espacios
        dirty_text = "Visita http://example.com o contacta a test@email.com    para mas   info"
        clean = preprocessor._clean_text(dirty_text)

        # Verificar que se removieron URLs y emails
        assert 'http://example.com' not in clean
        assert 'test@email.com' not in clean

        # Verificar que se normalizaron espacios
        assert '   ' not in clean
        assert '  ' not in clean

        # Verificar que esta en lowercase
        assert clean == clean.lower()

    def test_number_removal(self, preprocessor):
        """Test de remocion de numeros"""
        text_with_numbers = "El curso tiene 40% de teoria y 60% de practica en 2024"
        clean = preprocessor._clean_text(text_with_numbers)

        # Verificar que se removieron los numeros (segun config)
        if preprocessor.config['nlp'].get('remove_numbers', False):
            assert '40' not in clean
            assert '60' not in clean
            assert '2024' not in clean

    def test_pos_tag_filtering(self, preprocessor, mock_extracted_text):
        """Test que solo se mantienen los POS tags configurados"""
        result = preprocessor.process(mock_extracted_text)

        # Verificar que todos los POS tags estan en la lista permitida
        for pos_tag in result.pos_tags:
            assert pos_tag in preprocessor.pos_keep

    def test_lemmatization(self, preprocessor, mock_extracted_text):
        """Test que la lematizacion funciona correctamente"""
        result = preprocessor.process(mock_extracted_text)

        # Verificar que hay lemmas
        assert len(result.lemmas) > 0

        # Verificar que los lemmas estan en lowercase
        for lemma in result.lemmas:
            assert lemma == lemma.lower()

    def test_entity_extraction(self, preprocessor, mock_extracted_text):
        """Test de extraccion de entidades"""
        result = preprocessor.process(mock_extracted_text)

        # Verificar que entities es una lista
        assert isinstance(result.entities, list)

        # Si hay entidades, verificar su estructura
        for entity in result.entities:
            assert 'text' in entity
            assert 'label' in entity
            assert 'start' in entity
            assert 'end' in entity

    def test_add_custom_stopword(self, preprocessor):
        """Test para agregar stopwords personalizadas"""
        # Agregar una stopword
        preprocessor.add_stopword("CustomWord")

        # Verificar que se agrego en lowercase
        assert "customword" in preprocessor.custom_stopwords

    def test_remove_custom_stopword(self, preprocessor):
        """Test para remover stopwords personalizadas"""
        # Agregar y luego remover una stopword
        preprocessor.add_stopword("TempWord")
        assert "tempword" in preprocessor.custom_stopwords

        preprocessor.remove_stopword("TempWord")
        assert "tempword" not in preprocessor.custom_stopwords

    def test_minimum_token_length(self, preprocessor):
        """Test que tokens cortos se filtran"""
        from src.utils.schemas import ExtractedText
        from datetime import datetime

        # Texto con palabras cortas
        short_text = ExtractedText(
            filename="test.pdf",
            raw_text="A la de un en es",
            metadata={},
            page_count=1,
            extraction_date=datetime.now()
        )

        result = preprocessor.process(short_text)

        # Verificar que palabras de 1-2 caracteres se filtraron
        for token in result.tokens:
            assert len(token) >= preprocessor.min_length

    def test_processed_text_output_structure(self, preprocessor, mock_extracted_text):
        """Test que la estructura del ProcessedText es correcta"""
        result = preprocessor.process(mock_extracted_text)

        # Verificar todos los campos requeridos
        assert hasattr(result, 'filename')
        assert hasattr(result, 'clean_text')
        assert hasattr(result, 'tokens')
        assert hasattr(result, 'lemmas')
        assert hasattr(result, 'pos_tags')
        assert hasattr(result, 'entities')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'processing_date')

        # Verificar tipos
        assert isinstance(result.filename, str)
        assert isinstance(result.clean_text, str)
        assert isinstance(result.tokens, list)
        assert isinstance(result.lemmas, list)
        assert isinstance(result.pos_tags, list)
        assert isinstance(result.entities, list)
        assert isinstance(result.metadata, dict)
