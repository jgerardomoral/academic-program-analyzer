"""
Preprocessor de texto extraído de PDFs.

CONTRATO:
- Input: ExtractedText object
- Output: ProcessedText object
- Limpia, tokeniza, lematiza, etiqueta POS
"""
import spacy
import re
from typing import List, Set
from src.utils.schemas import ExtractedText, ProcessedText
from src.utils.config import load_config
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Limpia y procesa texto académico en español"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)

        # Cargar modelo spaCy
        model_name = self.config['nlp']['spacy_model']
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Modelo {model_name} no encontrado. Descargando...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)

        # Stopwords custom
        self.custom_stopwords = self._load_stopwords()

        # Configuración
        self.min_length = self.config['nlp']['min_word_length']
        self.max_length = self.config['nlp']['max_word_length']
        self.pos_keep = set(self.config['nlp']['pos_tags_keep'])

    def _load_stopwords(self) -> Set[str]:
        """Carga stopwords estándar + custom"""
        # spaCy stopwords
        stopwords = set(self.nlp.Defaults.stop_words)

        # Agregar custom
        custom = self.config.get('stopwords_custom', [])
        stopwords.update(custom)

        # Cargar de archivo si existe
        stopwords_file = self.config['paths'].get('stopwords')
        if stopwords_file:
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    file_stopwords = [line.strip() for line in f]
                    stopwords.update(file_stopwords)
            except FileNotFoundError:
                logger.warning(f"Archivo stopwords no encontrado: {stopwords_file}")

        return stopwords

    def process(self, extracted: ExtractedText) -> ProcessedText:
        """
        Procesa texto extraído.

        Pipeline:
        1. Limpieza básica (lowercase, remover especiales)
        2. Tokenización con spaCy
        3. Filtrado (stopwords, longitud, POS)
        4. Lematización
        5. Extracción de entidades

        Args:
            extracted: ExtractedText object

        Returns:
            ProcessedText object
        """
        # 1. Limpieza básica
        clean = self._clean_text(extracted.raw_text)

        # 2. Procesar con spaCy
        doc = self.nlp(clean)

        # 3. Filtrar tokens
        filtered_tokens = []
        lemmas = []
        pos_tags = []

        for token in doc:
            # Filtros
            if token.is_space or token.is_punct:
                continue
            if token.text.lower() in self.custom_stopwords:
                continue
            if len(token.text) < self.min_length or len(token.text) > self.max_length:
                continue
            if self.pos_keep and token.pos_ not in self.pos_keep:
                continue

            filtered_tokens.append(token.text.lower())
            lemmas.append(token.lemma_.lower())
            pos_tags.append(token.pos_)

        # 4. Extraer entidades
        entities = [
            {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            for ent in doc.ents
        ]

        return ProcessedText(
            filename=extracted.filename,
            clean_text=' '.join(filtered_tokens),
            tokens=filtered_tokens,
            lemmas=lemmas,
            pos_tags=pos_tags,
            entities=entities,
            metadata=extracted.metadata,
            processing_date=datetime.now()
        )

    def _clean_text(self, text: str) -> str:
        """Limpieza básica de texto"""
        # Lowercase
        text = text.lower()

        # Remover URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remover emails
        text = re.sub(r'\S+@\S+', '', text)

        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)

        # Remover números si configurado
        if self.config['nlp'].get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)

        return text.strip()

    def add_stopword(self, word: str):
        """Agrega stopword custom"""
        self.custom_stopwords.add(word.lower())

    def remove_stopword(self, word: str):
        """Remueve stopword custom"""
        self.custom_stopwords.discard(word.lower())
