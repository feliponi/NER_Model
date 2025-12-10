"""
utils.py

Utilitários comuns para o pipeline de skill extraction.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set
import pandas as pd
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


def load_json(filepath: str) -> List[Dict]:
    """Carrega arquivo JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict], filepath: str, indent: int = 2):
    """Salva dados em JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_csv(filepath: str) -> pd.DataFrame:
    """Carrega CSV."""
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: str):
    """Salva DataFrame em CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def extract_unique_skills(data: List[Dict], skill_column: str = "skills") -> Set[str]:
    """
    Extrai skills únicas de um dataset.

    Args:
        data: Lista de dicts ou DataFrame
        skill_column: Nome da coluna com skills

    Returns:
        Set de skills únicas normalizadas
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_dict("records")

    all_skills = set()
    for item in data:
        skills_str = item.get(skill_column, "")
        if isinstance(skills_str, str):
            skills = [s.strip().lower() for s in skills_str.split(",")]
            all_skills.update(s for s in skills if s)

    return all_skills


def analyze_label_distribution(labeled_data: List[Dict]) -> Dict:
    """
    Analisa distribuição de labels em dataset NER.

    Args:
        labeled_data: Lista de {text, entities}

    Returns:
        Dict com estatísticas
    """
    stats = {
        "total_examples": len(labeled_data),
        "examples_with_entities": 0,
        "total_entities": 0,
        "avg_entities_per_example": 0,
        "entity_lengths": [],
        "examples_by_entity_count": Counter(),
    }

    for item in labeled_data:
        entities = item.get("entities", [])
        num_entities = len(entities)

        if num_entities > 0:
            stats["examples_with_entities"] += 1
            stats["total_entities"] += num_entities
            stats["examples_by_entity_count"][num_entities] += 1

            # Calcula comprimento das entities
            for start, end, _ in entities:
                stats["entity_lengths"].append(end - start)

    if stats["total_examples"] > 0:
        stats["avg_entities_per_example"] = (
            stats["total_entities"] / stats["total_examples"]
        )

    if stats["entity_lengths"]:
        stats["avg_entity_length"] = np.mean(stats["entity_lengths"])
        stats["median_entity_length"] = np.median(stats["entity_lengths"])

    return stats


def print_label_stats(stats: Dict):
    """Imprime estatísticas de forma legível."""
    print("\n" + "=" * 60)
    print("LABEL STATISTICS")
    print("=" * 60)
    print(f"Total examples:              {stats['total_examples']}")
    print(f"Examples with entities:      {stats['examples_with_entities']}")
    print(f"Total entities:              {stats['total_entities']}")
    print(f"Avg entities/example:        {stats['avg_entities_per_example']:.2f}")

    if "avg_entity_length" in stats:
        print(f"Avg entity length (chars):   {stats['avg_entity_length']:.1f}")
        print(f"Median entity length:        {stats['median_entity_length']:.1f}")

    print("\nDistribution by entity count:")
    for count, freq in sorted(stats["examples_by_entity_count"].items())[:10]:
        print(f"  {count} entities: {freq} examples")

    print("=" * 60 + "\n")


def validate_dataset_format(data: List[Dict], required_keys: List[str]) -> bool:
    """
    Valida formato do dataset.

    Args:
        data: Dataset a validar
        required_keys: Chaves obrigatórias

    Returns:
        True se válido, False caso contrário
    """
    if not data:
        logger.error("Dataset vazio")
        return False

    for i, item in enumerate(data[:100]):  # Valida primeiros 100
        if not isinstance(item, dict):
            logger.error(f"Item {i} não é um dict")
            return False

        for key in required_keys:
            if key not in item:
                logger.error(f"Item {i} não contém chave '{key}'")
                return False

    return True


def normalize_skill_name(skill: str) -> str:
    """
    Normaliza nome de skill para comparação.

    Args:
        skill: Nome da skill

    Returns:
        Skill normalizada
    """
    if not isinstance(skill, str):
        return ""

    # Lowercase e remove espaços extras
    normalized = skill.lower().strip()

    # Remove pontuação comum
    normalized = normalized.replace(".", "").replace("-", " ")

    # Normaliza espaços
    normalized = " ".join(normalized.split())

    return normalized


def deduplicate_skills(skills: List[str]) -> List[str]:
    """
    Remove duplicatas de skills (case-insensitive).

    Args:
        skills: Lista de skills

    Returns:
        Lista de skills únicas (mantém primeira ocorrência)
    """
    seen = set()
    unique = []

    for skill in skills:
        normalized = normalize_skill_name(skill)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(skill)

    return unique
