"""
Mapeo de términos a habilidades usando taxonomía definida.

CONTRATO:
- Input: FrequencyAnalysis + taxonomía
- Output: DocumentSkillProfile
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import json
from src.utils.schemas import (
    FrequencyAnalysis,
    DocumentSkillProfile,
    SkillScore,
    Skill,
    ProcessedText
)
from src.utils.config import load_config
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SkillsMapper:
    """Mapea términos extraídos a habilidades definidas"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.skills_config = self.config['skills']

        # Cargar taxonomía
        self.taxonomia = self.load_taxonomy()
        self.skills = self._parse_taxonomia()

    def load_taxonomy(self, taxonomy_path: str = None) -> dict:
        """Carga taxonomía de habilidades desde JSON"""
        if taxonomy_path is None:
            taxonomy_path = self.config['paths']['taxonomia']

        try:
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Taxonomía no encontrada en {taxonomy_path}")
            return self._get_default_taxonomia()

    def _get_default_taxonomia(self) -> dict:
        """Taxonomía por defecto si no existe archivo"""
        return {
            "pensamiento_critico": {
                "name": "Pensamiento Crítico",
                "keywords": ["análisis", "evaluar", "criticar", "argumentar", "razonar"],
                "synonyms": ["analítico", "crítico", "evaluativo"],
                "weight": 1.0,
                "category": "cognitiva"
            },
            "programacion": {
                "name": "Programación",
                "keywords": ["código", "programar", "algoritmo", "software", "desarrollo"],
                "synonyms": ["codificar", "implementar", "desarrollar"],
                "weight": 1.0,
                "category": "tecnica"
            },
            "resolucion_problemas": {
                "name": "Resolución de Problemas",
                "keywords": ["resolver", "problema", "solución", "optimizar"],
                "synonyms": ["solucionar", "abordar"],
                "weight": 1.0,
                "category": "cognitiva"
            }
        }

    def _parse_taxonomia(self) -> List[Skill]:
        """Convierte taxonomía dict a objetos Skill"""
        skills = []
        for skill_id, data in self.taxonomia.items():
            skills.append(Skill(
                skill_id=skill_id,
                name=data['name'],
                keywords=data['keywords'],
                synonyms=data.get('synonyms', []),
                weight=data.get('weight', 1.0),
                category=data.get('category', 'general')
            ))
        return skills

    def map_document(self, frequency_analysis: FrequencyAnalysis,
                     processed_text: ProcessedText = None) -> DocumentSkillProfile:
        """
        Mapea un documento a perfil de habilidades.

        Args:
            frequency_analysis: Análisis de frecuencias del documento
            processed_text: Texto procesado (opcional, para snippets)

        Returns:
            DocumentSkillProfile con scores de habilidades
        """
        skill_scores = []

        for skill in self.skills:
            score_result = self._calculate_skill_score(
                skill,
                frequency_analysis,
                processed_text
            )
            skill_scores.append(score_result)

        # Ordenar por score
        skill_scores.sort(key=lambda x: x.score, reverse=True)

        # Top skills (score > min_confidence)
        min_conf = self.skills_config['min_confidence']
        top_skills = [
            ss.skill_name for ss in skill_scores
            if ss.score > min_conf
        ]

        # Calcular coverage (% de términos mapeados)
        total_terms = len(frequency_analysis.term_frequencies)
        mapped_terms = sum(len(ss.matched_terms) for ss in skill_scores)
        coverage = min(mapped_terms / total_terms, 1.0) if total_terms > 0 else 0.0

        return DocumentSkillProfile(
            document_id=frequency_analysis.document_id,
            skill_scores=skill_scores,
            top_skills=top_skills[:10],  # Top 10
            skill_coverage=coverage,
            analysis_date=datetime.now()
        )

    def _calculate_skill_score(self, skill: Skill,
                               freq_analysis: FrequencyAnalysis,
                               processed_text: ProcessedText = None) -> SkillScore:
        """
        Calcula score de una habilidad para un documento.

        Score = weighted_avg(tfidf_score, frequency_score)
        """
        term_df = freq_analysis.term_frequencies

        # Todos los términos relacionados con la skill
        all_keywords = set(skill.keywords + skill.synonyms)

        # Encontrar matches
        matched_terms = []
        tfidf_sum = 0.0
        freq_sum = 0

        for keyword in all_keywords:
            # Buscar keyword en términos (match exacto o substring)
            matches = term_df[term_df['term'].str.contains(keyword, case=False, na=False)]

            if not matches.empty:
                matched_terms.extend(matches['term'].tolist())
                tfidf_sum += matches['tfidf'].sum()
                freq_sum += matches['frequency'].sum()

        matched_terms = list(set(matched_terms))  # Únicos

        # Normalizar scores
        max_tfidf = term_df['tfidf'].max() if not term_df.empty else 1.0
        max_freq = term_df['frequency'].max() if not term_df.empty else 1.0

        tfidf_score = (tfidf_sum / max_tfidf) if max_tfidf > 0 else 0.0
        freq_score = (freq_sum / max_freq) if max_freq > 0 else 0.0

        # Score combinado
        w_tfidf = self.skills_config['weight_tfidf']
        w_freq = self.skills_config['weight_frequency']

        final_score = (w_tfidf * tfidf_score + w_freq * freq_score) * skill.weight
        final_score = min(final_score, 1.0)  # Cap at 1.0

        # Confianza basada en # de matches
        confidence = min(len(matched_terms) / len(skill.keywords), 1.0) if len(skill.keywords) > 0 else 0.0

        # Extraer snippets de contexto
        snippets = []
        if processed_text and matched_terms:
            snippets = self._find_context_snippets(
                matched_terms[:3],  # Primeros 3
                processed_text
            )

        return SkillScore(
            skill_id=skill.skill_id,
            skill_name=skill.name,
            score=final_score,
            confidence=confidence,
            matched_terms=matched_terms,
            context_snippets=snippets
        )

    def _find_context_snippets(self, terms: List[str],
                               processed: ProcessedText,
                               window: int = 50) -> List[str]:
        """Extrae fragmentos de texto donde aparecen términos"""
        snippets = []
        text = processed.clean_text
        text_lower = text.lower()

        for term in terms:
            pos = text_lower.find(term.lower())
            if pos != -1:
                start = max(0, pos - window)
                end = min(len(text), pos + len(term) + window)
                snippet = "..." + text[start:end] + "..."
                snippets.append(snippet)

        return snippets[:3]  # Max 3 snippets

    def create_skills_matrix(self, profiles: List[DocumentSkillProfile]) -> pd.DataFrame:
        """
        Crea matriz documento × habilidad.

        Args:
            profiles: Lista de perfiles de documentos

        Returns:
            DataFrame con documentos en filas, habilidades en columnas
        """
        # Todas las skills únicas
        all_skills = set()
        for profile in profiles:
            all_skills.update([ss.skill_name for ss in profile.skill_scores])

        # Matriz
        data = {}
        for profile in profiles:
            skill_dict = {
                ss.skill_name: ss.score
                for ss in profile.skill_scores
            }
            data[profile.document_id] = [
                skill_dict.get(skill, 0.0) for skill in all_skills
            ]

        df = pd.DataFrame(data, index=list(all_skills)).T
        return df

    def compare_profiles(self, profile1: DocumentSkillProfile,
                        profile2: DocumentSkillProfile) -> pd.DataFrame:
        """Compara dos perfiles de habilidades"""
        # Crear DataFrame comparativo
        skills = set(
            [ss.skill_name for ss in profile1.skill_scores] +
            [ss.skill_name for ss in profile2.skill_scores]
        )

        score1 = {ss.skill_name: ss.score for ss in profile1.skill_scores}
        score2 = {ss.skill_name: ss.score for ss in profile2.skill_scores}

        comparison = pd.DataFrame({
            profile1.document_id: [score1.get(s, 0.0) for s in skills],
            profile2.document_id: [score2.get(s, 0.0) for s in skills],
        }, index=list(skills))

        comparison['difference'] = abs(
            comparison.iloc[:, 0] - comparison.iloc[:, 1]
        )

        return comparison.sort_values('difference', ascending=False)

    def save_taxonomy(self, taxonomy: dict, path: str = None):
        """
        Actualiza taxonomía y guarda a archivo.

        Args:
            taxonomy: Diccionario con nueva taxonomía
            path: Ruta donde guardar (opcional, usa config por defecto)
        """
        if path is None:
            path = self.config['paths']['taxonomia']

        # Actualizar taxonomía en memoria
        self.taxonomia = taxonomy
        self.skills = self._parse_taxonomia()

        # Guardar a archivo
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(taxonomy, f, indent=2, ensure_ascii=False)

        logger.info(f"Taxonomía actualizada con {len(self.skills)} habilidades en {path}")
