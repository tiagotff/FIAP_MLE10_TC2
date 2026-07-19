"""Testes unitários das preprocessing strategies e do FeaturePipeline."""

from __future__ import annotations

import pandas as pd

from recommender.preprocessing.pipeline import FeaturePipeline
from recommender.preprocessing.strategies import (
    BasketSizeStrategy,
    RecencyFrequencyStrategy,
    TemporalPatternStrategy,
)


def _sample_orders() -> pd.DataFrame:
    """Cria um dataframe de pedidos de exemplo para os testes."""
    return pd.DataFrame(
        {
            "order_id": [1, 1, 2],
            "user_id": [10, 10, 11],
            "product_id": [100, 101, 100],
            "order_hour_of_day": [8, 8, 14],
            "order_dow": [1, 1, 3],
            "days_since_prior_order": [7, 7, 3],
        }
    )


def test_temporal_pattern_strategy_returns_expected_columns() -> None:
    """A estratégia temporal deve retornar exatamente as colunas esperadas."""
    features = TemporalPatternStrategy().fit(_sample_orders()).transform(
        _sample_orders()
    )
    assert list(features.columns) == ["order_hour_of_day", "order_dow"]
    assert len(features) == 3


def test_basket_size_strategy_counts_products_per_order() -> None:
    """A estratégia de tamanho de carrinho deve contar produtos por pedido."""
    features = BasketSizeStrategy().fit(_sample_orders()).transform(_sample_orders())
    assert features["basket_size"].tolist() == [2, 2, 1]


def test_recency_frequency_strategy_preserves_row_count() -> None:
    """A estratégia deve manter uma linha por registro original (não agregar)."""
    orders = _sample_orders()
    features = RecencyFrequencyStrategy().fit(orders).transform(orders)
    assert len(features) == len(orders)
    assert list(features.columns) == ["purchase_count", "days_since_last_order"]


def test_recency_frequency_strategy_broadcasts_group_stats() -> None:
    """purchase_count deve refletir a contagem do par usuário-produto, por linha."""
    orders = _sample_orders()
    features = RecencyFrequencyStrategy().fit(orders).transform(orders)
    # user_id=10, product_id=100 aparece 1x; user_id=11, product_id=100 aparece 1x
    assert features["purchase_count"].tolist() == [1, 1, 1]


def test_feature_pipeline_with_all_three_strategies_preserves_row_count() -> None:
    """Integração: as 3 estratégias combinadas não podem alterar o nº de linhas."""
    orders = _sample_orders()
    pipeline = FeaturePipeline(
        [RecencyFrequencyStrategy(), TemporalPatternStrategy(), BasketSizeStrategy()]
    )
    features = pipeline.fit_transform(orders)
    assert len(features) == len(orders)
    assert not features.isna().any().any()


def test_feature_pipeline_concatenates_all_strategies() -> None:
    """O pipeline deve concatenar as colunas de todas as estratégias registradas."""
    pipeline = FeaturePipeline([TemporalPatternStrategy(), BasketSizeStrategy()])
    features = pipeline.fit_transform(_sample_orders())
    assert set(features.columns) == {"order_hour_of_day", "order_dow", "basket_size"}
    assert len(features) == 3
