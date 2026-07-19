"""Funções compartilhadas entre os estágios `train` e `evaluate` do pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from recommender.config.model_config import (
    EmbeddingConfig,
    HybridModelConfig,
    MlpConfig,
)
from recommender.pipeline.feature_eng import FEATURE_COLUMNS


def build_model_config(models_dir: Path) -> HybridModelConfig:
    """Monta a config do modelo a partir dos tamanhos de vocabulário salvos.

    Args:
        models_dir: Diretório onde `vocab_sizes.json` foi salvo pelo
            estágio `feature_eng`.

    Returns:
        Config pronta para `ModelFactory.create`.
    """
    vocab_sizes = json.loads((models_dir / "vocab_sizes.json").read_text())
    return HybridModelConfig(
        embedding=EmbeddingConfig(**vocab_sizes),
        mlp=MlpConfig(tabular_feature_dim=len(FEATURE_COLUMNS)),
    )
