"""Carrega o modelo treinado e os encoders para servir predições.

Carregado uma única vez, na subida da API — não a cada requisição.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch

from recommender.models.factory import ModelFactory
from recommender.pipeline.common import build_model_config
from recommender.pipeline.feature_eng import FEATURE_COLUMNS


class InferenceService:
    """Encapsula o modelo carregado e os encoders, prontos para predição."""

    def __init__(self, models_dir: Path, device: str = "cpu") -> None:
        self.device = device
        self._user_index = self._load_id_index(models_dir / "user_encoder.joblib")
        self._product_index = self._load_id_index(models_dir / "product_encoder.joblib")
        self._metrics = self._load_metrics(models_dir / "model_metrics.json")

        self.model_config = build_model_config(models_dir)
        self.model = ModelFactory.create(self.model_config).to(device)
        self.model.load_state_dict(
            torch.load(models_dir / "model.pt", map_location=device)
        )
        self.model.eval()

    @staticmethod
    def _load_id_index(encoder_path: Path) -> dict[int, int]:
        """Carrega um encoder salvo e monta um dict {id_original: índice}."""
        encoder = joblib.load(encoder_path)
        return {int(raw_id): idx for idx, raw_id in enumerate(encoder.classes_)}

    @staticmethod
    def _load_metrics(metrics_path: Path) -> dict[str, float]:
        """Carrega as métricas de avaliação do modelo, se o arquivo existir."""
        if not metrics_path.exists():
            return {}
        return json.loads(metrics_path.read_text())

    def info(self) -> dict:
        """Metadados do modelo carregado, para o endpoint `/metadata`."""
        return {
            "num_users": self.model_config.embedding.num_users,
            "num_products": self.model_config.embedding.num_products,
            "embedding_dim": self.model_config.embedding.user_embedding_dim,
            "mlp_hidden_dims": list(self.model_config.mlp.hidden_dims),
            "feature_columns": list(FEATURE_COLUMNS),
            "metrics": self._metrics,
        }

    def encode_ids(self, user_id: int, product_id: int) -> tuple[int, int] | None:
        """Traduz ids originais para os índices que o modelo conhece.

        Returns:
            `(user_idx, product_idx)`, ou `None` se algum dos dois nunca
            apareceu no treino (cold-start — o modelo não tem embedding
            para esse id).
        """
        user_idx = self._user_index.get(user_id)
        product_idx = self._product_index.get(product_id)
        if user_idx is None or product_idx is None:
            return None
        return user_idx, product_idx

    def predict(
        self, user_idx: int, product_idx: int, features: list[float]
    ) -> float:
        """Calcula a probabilidade de recompra para um par usuário-produto.

        Args:
            user_idx: Índice do usuário já codificado (ver `encode_ids`).
            product_idx: Índice do produto já codificado.
            features: Valores das features tabulares, na ordem de
                `FEATURE_COLUMNS`.

        Returns:
            Probabilidade de recompra, entre 0 e 1.
        """
        with torch.no_grad():
            logits = self.model(
                torch.tensor([user_idx], dtype=torch.long, device=self.device),
                torch.tensor([product_idx], dtype=torch.long, device=self.device),
                torch.tensor([features], dtype=torch.float32, device=self.device),
            )
            return float(torch.sigmoid(logits).item())

    def predict_batch(
        self, requests: list[tuple[int, int, list[float]]]
    ) -> list[float | None]:
        """Prediz em lote (um único forward pass) para vários pares.

        Args:
            requests: Lista de tuplas `(user_id, product_id, features)`
                com ids originais (não codificados).

        Returns:
            Probabilidades na mesma ordem da entrada; `None` nas posições
            com `user_id`/`product_id` fora do vocabulário de treino.
        """
        encoded = [self.encode_ids(u, p) for u, p, _ in requests]
        valid = [i for i, e in enumerate(encoded) if e is not None]
        results: list[float | None] = [None] * len(requests)
        if not valid:
            return results

        probs = self._forward_batch(
            [encoded[i][0] for i in valid],
            [encoded[i][1] for i in valid],
            [requests[i][2] for i in valid],
        )
        for i, prob in zip(valid, probs, strict=False):
            results[i] = prob
        return results

    def _forward_batch(
        self, user_idxs: list[int], product_idxs: list[int], features: list[list[float]]
    ) -> list[float]:
        """Roda um forward pass vetorizado para um lote já codificado."""
        with torch.no_grad():
            logits = self.model(
                torch.tensor(user_idxs, dtype=torch.long, device=self.device),
                torch.tensor(product_idxs, dtype=torch.long, device=self.device),
                torch.tensor(features, dtype=torch.float32, device=self.device),
            )
            return torch.sigmoid(logits).tolist()
