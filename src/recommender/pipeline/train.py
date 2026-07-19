"""Estágio 3 do pipeline DVC: treina o modelo híbrido com tracking no MLflow.

Lê os splits de treino/validação gerados pelo `feature_eng`, treina o
`HybridMlpRecommender` (via ModelFactory) com early stopping, registra
parâmetros/métricas/artefatos de cada run no MLflow, e salva os pesos
do melhor modelo em `models/model.pt`.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from recommender.config.model_config import TrainingConfig
from recommender.config.settings import get_settings
from recommender.data.dataset import InstacartReorderDataset
from recommender.models.factory import ModelFactory
from recommender.pipeline.common import build_model_config
from recommender.pipeline.feature_eng import FEATURE_COLUMNS, LABEL_COLUMN


def _load_datasets(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os parquets de treino e validação gerados pelo feature_eng."""
    train_df = pd.read_parquet(processed_dir / "features_train.parquet")
    val_df = pd.read_parquet(processed_dir / "features_val.parquet")
    return train_df, val_df


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> float:
    """Calcula o AUC-ROC do modelo no conjunto passado em `loader`."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for user_ids, product_ids, features, labels in loader:
            logits = model(
                user_ids.to(device), product_ids.to(device), features.to(device)
            )
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.tolist())
    return roc_auc_score(all_labels, all_probs)


def run(
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    training_config: TrainingConfig | None = None,
) -> Path:
    """Executa o treino completo, com tracking no MLflow.

    Args:
        processed_dir: Override do diretório de dados processados.
        models_dir: Override do diretório de artefatos de modelo.
        training_config: Override dos hiperparâmetros de treino.

    Returns:
        Caminho do arquivo de pesos do modelo salvo.
    """
    settings = get_settings()
    processed_dir = processed_dir or Path(settings.data_processed_dir)
    models_dir = models_dir or Path(settings.models_dir)
    training_config = training_config or TrainingConfig(
        random_seed=settings.random_seed
    )
    device = settings.device

    torch.manual_seed(training_config.random_seed)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("instacart-recommender")

    train_df, val_df = _load_datasets(processed_dir)
    train_ds = InstacartReorderDataset.from_dataframe(
        train_df, FEATURE_COLUMNS, LABEL_COLUMN
    )
    val_ds = InstacartReorderDataset.from_dataframe(
        val_df, FEATURE_COLUMNS, LABEL_COLUMN
    )
    train_loader = DataLoader(
        train_ds, batch_size=training_config.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=training_config.batch_size)

    model_config = build_model_config(models_dir)
    model = ModelFactory.create(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": model_config.model_type,
                "learning_rate": training_config.learning_rate,
                "batch_size": training_config.batch_size,
                "max_epochs": training_config.max_epochs,
            }
        )

        best_auc, patience_left = 0.0, training_config.early_stopping_patience
        model_path = models_dir / "model.pt"
        for epoch in range(training_config.max_epochs):
            model.train()
            for user_ids, product_ids, features, labels in train_loader:
                optimizer.zero_grad()
                logits = model(
                    user_ids.to(device), product_ids.to(device), features.to(device)
                )
                loss = criterion(logits, labels.to(device))
                loss.backward()
                optimizer.step()

            val_auc = _evaluate(model, val_loader, device)
            mlflow.log_metrics({"val_auc": val_auc}, step=epoch)
            print(f"[train] epoch={epoch} val_auc={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                patience_left = training_config.early_stopping_patience
                torch.save(model.state_dict(), model_path)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("[train] early stopping acionado")
                    break

        mlflow.log_metric("best_val_auc", best_auc)
        mlflow.log_artifact(str(model_path))

    return model_path


if __name__ == "__main__":
    run()
