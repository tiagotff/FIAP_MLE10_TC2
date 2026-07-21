"""Schemas de configuração para arquitetura de modelo e treino.

Dataclasses tipadas usadas pelo `ModelFactory` e pelo pipeline de treino.
São carregadas a partir de `configs/model.yaml` e `configs/training.yaml`
(ver `recommender.pipeline.common.load_model_config` e
`recommender.pipeline.train.load_training_config`) — não devem ser
confundidas com `recommender.config.settings.Settings`, que cobre
configurações de ambiente de execução (`.env`), não hiperparâmetros de
modelo/treino.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmbeddingConfig:
    """Dimensões das tabelas de embedding de usuário/produto.

    Attributes:
        num_users: Cardinalidade do vocabulário de usuários.
        num_products: Cardinalidade do vocabulário de produtos.
        user_embedding_dim: Tamanho do vetor de embedding de cada usuário.
        product_embedding_dim: Tamanho do vetor de embedding de cada produto.
    """

    num_users: int
    num_products: int
    user_embedding_dim: int = 32
    product_embedding_dim: int = 32


@dataclass(frozen=True)
class MlpConfig:
    """Hiperparâmetros da MLP no topo do modelo híbrido.

    Attributes:
        tabular_feature_dim: Número de features tabulares (não-embedding).
        hidden_dims: Tamanhos das camadas ocultas, em ordem.
        dropout: Probabilidade de dropout aplicada após cada camada oculta.
        use_batchnorm: Se aplica BatchNorm1d após cada camada oculta.
    """

    tabular_feature_dim: int
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.3
    use_batchnorm: bool = True


@dataclass(frozen=True)
class HybridModelConfig:
    """Configuração completa do modelo híbrido (embeddings + MLP).

    Attributes:
        embedding: Configuração das tabelas de embedding.
        mlp: Configuração da MLP no topo do modelo.
        model_type: Identificador consumido pelo ModelFactory.
    """

    embedding: EmbeddingConfig
    mlp: MlpConfig
    model_type: str = "hybrid_mlp"


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparâmetros do loop de treino.

    Attributes:
        learning_rate: Taxa de aprendizado do otimizador.
        batch_size: Tamanho do mini-batch.
        max_epochs: Limite superior de épocas de treino.
        early_stopping_patience: Épocas sem melhora no AUC de validação
            antes de interromper o treino.
        random_seed: Seed aplicada a torch/numpy/random para reprodutibilidade.
    """

    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 30
    early_stopping_patience: int = 5
    random_seed: int = 42
    metrics: list[str] = field(
        default_factory=lambda: ["auc_roc", "recall", "precision", "f1"]
    )
