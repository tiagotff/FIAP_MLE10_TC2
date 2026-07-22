"""Registro do modelo treinado no MLflow Model Registry.

Após cada treino, a versão gerada é registrada e promovida
automaticamente: primeiro para Staging, e para Production apenas se
superar (ou for a primeira) a versão atualmente em Production —
evita que um treino com resultado pior sobrescreva um modelo melhor
já em produção.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from recommender.config.model_config import HybridModelConfig
from recommender.models.factory import ModelFactory

REGISTERED_MODEL_NAME = "instacart-recommender"


def _log_and_register_model(
    model_path: Path, model_config: HybridModelConfig, device: str
) -> ModelVersion:
    """Recarrega os melhores pesos salvos e registra a versão no Registry."""
    model = ModelFactory.create(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    mlflow.pytorch.log_model(model, artifact_path="model")

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    return mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)


def _current_production_auc(client: MlflowClient) -> float | None:
    """Retorna o best_val_auc da versão atual em Production, se existir."""
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        return None
    run = client.get_run(versions[0].run_id)
    return run.data.metrics.get("best_val_auc")


def _promote(client: MlflowClient, version: str, new_auc: float) -> None:
    """Promove a versão para Staging, e para Production se for a melhor até agora."""
    client.transition_model_version_stage(REGISTERED_MODEL_NAME, version, "Staging")

    current_auc = _current_production_auc(client)
    if current_auc is not None and new_auc <= current_auc:
        print(
            f"[registry] v{version} fica em Staging "
            f"(auc={new_auc:.4f} <= produção {current_auc:.4f})"
        )
        return

    client.transition_model_version_stage(
        REGISTERED_MODEL_NAME, version, "Production", archive_existing_versions=True
    )
    print(f"[registry] v{version} promovida a Production (auc={new_auc:.4f})")


def register_and_promote(
    model_path: Path, model_config: HybridModelConfig, device: str, best_auc: float
) -> ModelVersion:
    """Registra a versão treinada e a promove no MLflow Model Registry.

    Deve ser chamado dentro de uma run ativa do MLflow (`with
    mlflow.start_run(): ...`), já que depende do `run_id` corrente para
    montar a URI do modelo a registrar.

    Args:
        model_path: Caminho dos pesos do melhor checkpoint salvo.
        model_config: Config usada para reconstruir a arquitetura.
        device: `"cpu"` ou `"cuda"`.
        best_auc: Melhor AUC de validação obtido no treino, usado para
            decidir a promoção a Production.

    Returns:
        A `ModelVersion` registrada.
    """
    client = MlflowClient()
    version = _log_and_register_model(model_path, model_config, device)
    _promote(client, version.version, best_auc)
    return version
