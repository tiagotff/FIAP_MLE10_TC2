"""Factory Pattern para instanciação de modelos.

Centraliza a criação de modelos para que o pipeline de treino dependa
apenas de `ModelFactory.create`, e não de classes concretas. Adicionar
uma nova arquitetura significa registrá-la aqui, sem alterar quem chama
(Open-Closed Principle).
"""

from __future__ import annotations

from collections.abc import Callable

from recommender.config.model_config import HybridModelConfig
from recommender.models.base import RecommenderModel
from recommender.models.hybrid_mlp import HybridMlpRecommender

_ModelBuilder = Callable[[HybridModelConfig], RecommenderModel]


class ModelFactory:
    """Constrói instâncias de `RecommenderModel` a partir de `model_type`."""

    _registry: dict[str, _ModelBuilder] = {
        "hybrid_mlp": HybridMlpRecommender,
    }

    @classmethod
    def create(cls, config: HybridModelConfig) -> RecommenderModel:
        """Instancia o modelo registrado sob `config.model_type`.

        Args:
            config: Configuração do modelo; `config.model_type` seleciona
                o builder.

        Returns:
            Um `RecommenderModel` inicializado.

        Raises:
            ValueError: Se `config.model_type` não estiver registrado.
        """
        builder = cls._registry.get(config.model_type)
        if builder is None:
            known = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"model_type '{config.model_type}' desconhecido. "
                f"Tipos conhecidos: {known}"
            )
        return builder(config)

    @classmethod
    def register(cls, model_type: str, builder: _ModelBuilder) -> None:
        """Registra um novo builder de modelo sob `model_type`."""
        cls._registry[model_type] = builder
