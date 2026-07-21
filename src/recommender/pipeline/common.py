"""Funções compartilhadas entre os estágios `train` e `evaluate` do pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from recommender.config.model_config import (
    EmbeddingConfig,
    HybridModelConfig,
    MlpConfig,
)
from recommender.pipeline.feature_eng import FEATURE_COLUMNS

MODEL_CONFIG_PATH = Path("configs/model.yaml")


def build_model_config(
    models_dir: Path, config_path: Path = MODEL_CONFIG_PATH
) -> HybridModelConfig:
    """Monta a config do modelo combinando `configs/model.yaml` com o vocabulário.

    As dimensões de embedding e a arquitetura da MLP vêm do YAML (se
    existir; caso contrário usa os defaults das dataclasses); a
    cardinalidade do vocabulário (`num_users`, `num_products`) e o
    número de features tabulares são descobertos automaticamente a
    partir dos artefatos gerados pelo estágio `feature_eng` — não fazem
    sentido como valores fixos num YAML, já que dependem do dataset.

    Args:
        models_dir: Diretório onde `vocab_sizes.json` foi salvo pelo
            estágio `feature_eng`.
        config_path: Caminho do YAML de configuração do modelo
            (default: `configs/model.yaml`).

    Returns:
        Config pronta para `ModelFactory.create`.
    """
    vocab_sizes = json.loads((models_dir / "vocab_sizes.json").read_text())
    raw = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}

    embedding_overrides = raw.get("embedding", {})
    mlp_overrides = dict(raw.get("mlp", {}))
    if "hidden_dims" in mlp_overrides:
        mlp_overrides["hidden_dims"] = tuple(mlp_overrides["hidden_dims"])

    return HybridModelConfig(
        embedding=EmbeddingConfig(**vocab_sizes, **embedding_overrides),
        mlp=MlpConfig(tabular_feature_dim=len(FEATURE_COLUMNS), **mlp_overrides),
        model_type=raw.get("model_type", "hybrid_mlp"),
    )
