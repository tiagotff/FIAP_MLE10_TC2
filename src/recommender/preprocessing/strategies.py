"""Concrete preprocessing strategies for Instacart order data."""

from __future__ import annotations

import pandas as pd

from recommender.preprocessing.base import PreprocessingStrategy


class RecencyFrequencyStrategy(PreprocessingStrategy):
    """Computes per user-product recency and purchase-frequency features."""

    def __init__(self) -> None:
        self._global_mean_frequency: float | None = None

    def fit(self, raw_orders: pd.DataFrame) -> RecencyFrequencyStrategy:
        """Store the global mean purchase frequency for later reference."""
        counts = raw_orders.groupby(["user_id", "product_id"]).size()
        self._global_mean_frequency = float(counts.mean())
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Compute purchase_count and days_since_last_order per user-product."""
        grouped = raw_orders.groupby(["user_id", "product_id"])
        features = grouped.agg(
            purchase_count=("order_id", "count"),
            days_since_last_order=("days_since_prior_order", "min"),
        )
        return features.reset_index(drop=True)


class TemporalPatternStrategy(PreprocessingStrategy):
    """Computes order-time-of-day and day-of-week features."""

    def fit(self, raw_orders: pd.DataFrame) -> TemporalPatternStrategy:
        """No statistics to learn; returns self for interface consistency."""
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Extract order hour-of-day and day-of-week columns."""
        return raw_orders[["order_hour_of_day", "order_dow"]].reset_index(drop=True)


class BasketSizeStrategy(PreprocessingStrategy):
    """Computes the user's average basket (order) size."""

    def fit(self, raw_orders: pd.DataFrame) -> BasketSizeStrategy:
        """No statistics to learn; returns self for interface consistency."""
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Compute the number of products in each order, per row."""
        basket_sizes = raw_orders.groupby("order_id")["product_id"].transform("count")
        return pd.DataFrame({"basket_size": basket_sizes}).reset_index(drop=True)
