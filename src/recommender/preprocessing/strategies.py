"""Estratégias concretas de pré-processamento para dados do Instacart."""

from __future__ import annotations

import pandas as pd

from recommender.preprocessing.base import PreprocessingStrategy


class RecencyFrequencyStrategy(PreprocessingStrategy):
    """Calcula features de recência e frequência de compra por par usuário-produto."""

    def fit(self, raw_orders: pd.DataFrame) -> RecencyFrequencyStrategy:
        """Não há estatísticas a aprender; retorna self por consistência."""
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Calcula purchase_count e days_since_last_order por usuário-produto.

        Usa `groupby(...).transform(...)` (não `agg`) para propagar a
        estatística de volta a cada linha original, preservando o
        número de linhas — essencial para concatenar com as demais
        estratégias no `FeaturePipeline`.

        Agrupa por uma chave combinada (`user_id * n_products + product_id`)
        em vez de um groupby multi-coluna: evita o hashing de MultiIndex
        do pandas, que consome bem mais memória em datasets grandes
        (relevante para os ~32M registros do Instacart).
        """
        n_products = int(raw_orders["product_id"].max()) + 1
        combined_key = (
            raw_orders["user_id"].astype("int64") * n_products
            + raw_orders["product_id"].astype("int64")
        )
        grouped = raw_orders.groupby(combined_key)
        purchase_count = grouped["order_id"].transform("count")
        days_since_last_order = grouped["days_since_prior_order"].transform("min")
        return pd.DataFrame(
            {
                "purchase_count": purchase_count,
                "days_since_last_order": days_since_last_order,
            }
        ).reset_index(drop=True)


class TemporalPatternStrategy(PreprocessingStrategy):
    """Calcula features de horário do pedido e dia da semana."""

    def fit(self, raw_orders: pd.DataFrame) -> TemporalPatternStrategy:
        """Não há estatísticas a aprender; retorna self por consistência."""
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Extrai as colunas de hora do pedido e dia da semana."""
        return raw_orders[["order_hour_of_day", "order_dow"]].reset_index(drop=True)


class BasketSizeStrategy(PreprocessingStrategy):
    """Calcula o tamanho médio do carrinho (pedido) do usuário."""

    def fit(self, raw_orders: pd.DataFrame) -> BasketSizeStrategy:
        """Não há estatísticas a aprender; retorna self por consistência."""
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Calcula o número de produtos em cada pedido, por linha."""
        basket_sizes = raw_orders.groupby("order_id")["product_id"].transform("count")
        return pd.DataFrame({"basket_size": basket_sizes}).reset_index(drop=True)
