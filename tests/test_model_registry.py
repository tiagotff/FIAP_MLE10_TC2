"""Testes unitários do model registry (download de artefatos via GCS).

Usa mocks para o cliente do Cloud Storage — não depende de credenciais
nem de um bucket real.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from recommender.api.model_registry import sync_model_from_gcs


def test_sync_is_a_noop_when_bucket_name_is_none(tmp_path: Path) -> None:
    """Sem MODEL_BUCKET definido, não deve tentar acessar o GCS."""
    with patch("google.cloud.storage.Client") as mock_client:
        sync_model_from_gcs(None, tmp_path)
        mock_client.assert_not_called()


def test_sync_downloads_all_required_artifacts(tmp_path: Path) -> None:
    """Com um bucket definido, deve baixar os 4 artefatos obrigatórios."""
    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_bucket.name = "meu-bucket"

    with patch("google.cloud.storage.Client") as mock_client_cls:
        mock_client_cls.return_value.bucket.return_value = mock_bucket
        sync_model_from_gcs("meu-bucket", tmp_path)

    required = {
        "model.pt", "user_encoder.joblib", "product_encoder.joblib", "vocab_sizes.json"
    }
    downloaded = {call.args[0] for call in mock_bucket.blob.call_args_list}
    assert required.issubset(downloaded)


def test_sync_does_not_fail_when_optional_metrics_file_is_missing(
    tmp_path: Path,
) -> None:
    """Se model_metrics.json não existir no bucket, não deve derrubar o sync."""
    mock_bucket = MagicMock()
    mock_bucket.name = "meu-bucket"

    def blob_side_effect(filename: str) -> MagicMock:
        blob = MagicMock()
        if filename == "model_metrics.json":
            blob.download_to_filename.side_effect = FileNotFoundError("não existe")
        return blob

    mock_bucket.blob.side_effect = blob_side_effect

    with patch("google.cloud.storage.Client") as mock_client_cls:
        mock_client_cls.return_value.bucket.return_value = mock_bucket
        sync_model_from_gcs("meu-bucket", tmp_path)  # não deve levantar exceção
