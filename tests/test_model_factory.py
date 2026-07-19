"""Testes unitários do ModelFactory."""

from __future__ import annotations

import pytest

from recommender.config.model_config import (
    EmbeddingConfig,
    HybridModelConfig,
    MlpConfig,
)
from recommender.models.factory import ModelFactory
from recommender.models.hybrid_mlp import HybridMlpRecommender


def _build_config(model_type: str = "hybrid_mlp") -> HybridModelConfig:
    """Cria uma configuração de modelo mínima para os testes."""
    return HybridModelConfig(
        embedding=EmbeddingConfig(num_users=100, num_products=200),
        mlp=MlpConfig(tabular_feature_dim=5),
        model_type=model_type,
    )


def test_factory_creates_hybrid_mlp() -> None:
    """O Factory deve instanciar HybridMlpRecommender para model_type='hybrid_mlp'."""
    model = ModelFactory.create(_build_config())
    assert isinstance(model, HybridMlpRecommender)


def test_factory_raises_on_unknown_model_type() -> None:
    """O Factory deve levantar ValueError para model_type desconhecido."""
    with pytest.raises(ValueError, match="desconhecido"):
        ModelFactory.create(_build_config(model_type="does_not_exist"))


def test_factory_register_adds_new_builder() -> None:
    """Registrar um novo builder deve torná-lo utilizável pelo Factory."""
    ModelFactory.register("hybrid_mlp_alias", HybridMlpRecommender)
    model = ModelFactory.create(_build_config(model_type="hybrid_mlp_alias"))
    assert isinstance(model, HybridMlpRecommender)
