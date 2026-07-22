"""Baixa os artefatos do modelo de um bucket GCS na subida da API.

Desacopla a promoção de um modelo novo do rebuild/redeploy da imagem
Docker: basta atualizar o bucket (ver `scripts/upload_model_to_gcs.sh`)
e reiniciar o serviço. Em desenvolvimento local, sem `MODEL_BUCKET`
definido, é um no-op — assume que os artefatos já estão em `models/`
(gerados pelo pipeline DVC).
"""

from __future__ import annotations

from pathlib import Path

from recommender.api.logging_config import logger

ARTIFACT_FILENAMES = (
    "model.pt",
    "user_encoder.joblib",
    "product_encoder.joblib",
    "vocab_sizes.json",
)
OPTIONAL_ARTIFACT_FILENAMES = ("model_metrics.json",)


def sync_model_from_gcs(bucket_name: str | None, models_dir: Path) -> None:
    """Baixa os artefatos do modelo de `gs://{bucket_name}/` para `models_dir`.

    Args:
        bucket_name: Nome do bucket GCS. Se `None`/vazio, não faz nada
            (modo de desenvolvimento local, com artefatos já presentes).
        models_dir: Diretório local onde os artefatos serão salvos.
    """
    if not bucket_name:
        logger.info(
            "MODEL_BUCKET não definido — usando artefatos locais.",
            extra={"models_dir": str(models_dir)},
        )
        return

    from google.cloud import storage

    models_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for filename in ARTIFACT_FILENAMES:
        _download_artifact(bucket, filename, models_dir / filename)
    for filename in OPTIONAL_ARTIFACT_FILENAMES:
        _download_optional_artifact(bucket, filename, models_dir / filename)


def _download_optional_artifact(bucket, filename: str, target: Path) -> None:  # noqa: ANN001
    """Baixa um artefato opcional; loga um aviso (não falha) se ausente."""
    try:
        _download_artifact(bucket, filename, target)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Artefato opcional não encontrado no GCS — seguindo sem ele.",
            extra={"path": filename, "bucket": bucket.name},
        )


def _download_artifact(bucket, filename: str, target: Path) -> None:  # noqa: ANN001
    """Baixa um único artefato do bucket, com log estruturado do resultado."""
    blob = bucket.blob(filename)
    blob.download_to_filename(str(target))
    logger.info(
        "Artefato do modelo baixado do GCS.",
        extra={"path": filename, "bucket": bucket.name},
    )
