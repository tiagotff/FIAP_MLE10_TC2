"""Strategy Pattern for feature preprocessing.

Each concrete strategy encapsulates one way of turning raw Instacart
order data into model-ready features. Swapping strategies (e.g. for an
ablation study or a new feature set) requires no change to the
pipeline that consumes them (Single Responsibility / Open-Closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PreprocessingStrategy(ABC):
    """Contract for a single feature-engineering strategy."""

    @abstractmethod
    def fit(self, raw_orders: pd.DataFrame) -> PreprocessingStrategy:
        """Learn any statistics needed at transform time (e.g. means).

        Args:
            raw_orders: Raw orders/order_products dataframe.

        Returns:
            self, to allow fluent `strategy.fit(df).transform(df)` usage.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Produce the feature columns this strategy is responsible for.

        Args:
            raw_orders: Raw orders/order_products dataframe.

        Returns:
            A dataframe with the same row index as `raw_orders` and one
            column per feature this strategy computes.
        """
        raise NotImplementedError
