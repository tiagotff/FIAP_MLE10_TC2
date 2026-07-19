"""Context class that orchestrates a list of preprocessing strategies."""

from __future__ import annotations

import pandas as pd

from recommender.preprocessing.base import PreprocessingStrategy


class FeaturePipeline:
    """Runs a sequence of `PreprocessingStrategy` and concatenates outputs."""

    def __init__(self, strategies: list[PreprocessingStrategy]) -> None:
        self._strategies = strategies

    def fit(self, raw_orders: pd.DataFrame) -> FeaturePipeline:
        """Fit every strategy on the raw orders."""
        for strategy in self._strategies:
            strategy.fit(raw_orders)
        return self

    def transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Transform raw orders into the full tabular feature matrix."""
        feature_frames = [s.transform(raw_orders) for s in self._strategies]
        return pd.concat(feature_frames, axis=1)

    def fit_transform(self, raw_orders: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper for `fit(df).transform(df)`."""
        return self.fit(raw_orders).transform(raw_orders)
