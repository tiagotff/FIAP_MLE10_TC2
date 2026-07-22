"""Estágio 3 do pipeline DVC: treina o modelo híbrido com tracking no MLflow.

Lê os splits de treino/validação gerados pelo `feature_eng`, treina o
`HybridMlpRecommender` (via ModelFactory) com early stopping, registra
parâmetros/métricas/artefatos de cada run no MLflow, e salva os pesos
do melhor modelo em `models/model.pt`.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from recommender.config.model_config import HybridModelConfig, TrainingConfig
from recommender.config.settings import get_settings
from recommender.data.dataset import InstacartReorderDataset
from recommender.models.factory import ModelFactory
from recommender.pipeline.common import build_model_config
from recommender.pipeline.feature_eng import FEATURE_COLUMNS, LABEL_COLUMN
from recommender.pipeline.registry import register_and_promote

TRAINING_CONFIG_PATH = Path("configs/training.yaml")
MIN_AUC_DELTA = 1e-3


def load_training_config(path: Path = TRAINING_CONFIG_PATH) -> TrainingConfig:
    """Carrega os hiperparâmetros de treino a partir de um YAML.

    Args:
        path: Caminho do YAML (default: `configs/training.yaml`). Se o
            arquivo não existir, retorna `TrainingConfig()` com os
            defaults do código.

    Returns:
        `TrainingConfig` preenchido a partir do YAML.
    """
    if not path.exists():
        return TrainingConfig()
    raw = yaml.safe_load(path.read_text())
    return TrainingConfig(**raw)


def _load_datasets(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os parquets de treino e validação gerados pelo feature_eng."""
    train_df = pd.read_parquet(processed_dir / "features_train.parquet")
    val_df = pd.read_parquet(processed_dir / "features_val.parquet")
    return train_df, val_df


def _build_dataloaders(
    train_df: pd.DataFrame, val_df: pd.DataFrame, training_config: TrainingConfig
) -> tuple[DataLoader, DataLoader]:
    """Constrói os DataLoaders de treino/validação, com workers paralelos."""
    train_ds = InstacartReorderDataset.from_dataframe(
        train_df, FEATURE_COLUMNS, LABEL_COLUMN
    )
    val_ds = InstacartReorderDataset.from_dataframe(
        val_df, FEATURE_COLUMNS, LABEL_COLUMN
    )
    num_workers = max(0, min(2, (os.cpu_count() or 1) - 1))
    kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0}
    train_loader = DataLoader(
        train_ds, batch_size=training_config.batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=training_config.batch_size, **kwargs)
    return train_loader, val_loader


def _build_model_and_optimizer(
    models_dir: Path, training_config: TrainingConfig, device: str
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module, HybridModelConfig]:
    """Instancia o modelo (via ModelFactory), o otimizador Adam e a loss."""
    model_config = build_model_config(models_dir)
    model = ModelFactory.create(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    return model, optimizer, criterion, model_config


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
) -> None:
    """Roda uma época de treino (forward + backward + step) sobre `loader`."""
    model.train()
    for user_ids, product_ids, features, labels in loader:
        optimizer.zero_grad()
        logits = model(
            user_ids.to(device), product_ids.to(device), features.to(device)
        )
        loss = criterion(logits, labels.to(device))
        loss.backward()
        optimizer.step()


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


def _update_best_checkpoint(
    model: torch.nn.Module,
    model_path: Path,
    val_auc: float,
    best_auc: float,
    patience_left: int,
    training_config: TrainingConfig,
) -> tuple[float, int, bool]:
    """Atualiza o melhor checkpoint salvo e a paciência do early stopping.

    Returns:
        Tupla `(novo best_auc, nova patience_left, deve parar agora)`.
    """
    if val_auc > best_auc + MIN_AUC_DELTA:
        torch.save(model.state_dict(), model_path)
        return val_auc, training_config.early_stopping_patience, False

    patience_left -= 1
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), model_path)
    return best_auc, patience_left, patience_left <= 0


def _log_run_params(
    model_config: HybridModelConfig, training_config: TrainingConfig
) -> None:
    """Loga os hiperparâmetros da run atual no MLflow."""
    mlflow.log_params(
        {
            "model_type": model_config.model_type,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "max_epochs": training_config.max_epochs,
        }
    )


def _run_training_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: str,
    training_config: TrainingConfig,
    model_path: Path,
) -> float:
    """Roda o loop de épocas com early stopping; retorna o melhor AUC visto."""
    best_auc, patience_left = 0.0, training_config.early_stopping_patience
    for epoch in range(training_config.max_epochs):
        _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_auc = _evaluate(model, val_loader, device)
        mlflow.log_metrics({"val_auc": val_auc}, step=epoch)
        print(f"[train] epoch={epoch} val_auc={val_auc:.4f}")

        best_auc, patience_left, should_stop = _update_best_checkpoint(
            model, model_path, val_auc, best_auc, patience_left, training_config
        )
        if should_stop:
            print("[train] early stopping acionado (sem melhora significativa)")
            break
    return best_auc


def _prepare_training(
    processed_dir: Path, models_dir: Path, training_config: TrainingConfig, device: str
) -> tuple[
    DataLoader, DataLoader, torch.nn.Module, torch.optim.Optimizer,
    torch.nn.Module, HybridModelConfig,
]:
    """Carrega os dados, monta os DataLoaders e instancia modelo/otimizador/loss."""
    train_df, val_df = _load_datasets(processed_dir)
    train_loader, val_loader = _build_dataloaders(train_df, val_df, training_config)
    model, optimizer, criterion, model_config = _build_model_and_optimizer(
        models_dir, training_config, device
    )
    return train_loader, val_loader, model, optimizer, criterion, model_config


def _resolve_run_config(
    processed_dir: Path | None,
    models_dir: Path | None,
    training_config: TrainingConfig | None,
) -> tuple[Path, Path, TrainingConfig, str]:
    """Resolve overrides contra os defaults das Settings; inicializa seed e MLflow.

    Returns:
        Tupla `(processed_dir, models_dir, training_config, device)`.
    """
    settings = get_settings()
    processed_dir = processed_dir or Path(settings.data_processed_dir)
    models_dir = models_dir or Path(settings.models_dir)
    training_config = training_config or load_training_config()

    torch.manual_seed(training_config.random_seed)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("instacart-recommender")
    return processed_dir, models_dir, training_config, settings.device


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
    processed_dir, models_dir, training_config, device = _resolve_run_config(
        processed_dir, models_dir, training_config
    )
    train_loader, val_loader, model, optimizer, criterion, model_config = (
        _prepare_training(processed_dir, models_dir, training_config, device)
    )
    model_path = models_dir / "model.pt"

    with mlflow.start_run():
        _log_run_params(model_config, training_config)
        best_auc = _run_training_loop(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            training_config,
            model_path,
        )
        mlflow.log_metric("best_val_auc", best_auc)
        mlflow.log_artifact(str(model_path))
        register_and_promote(model_path, model_config, device, best_auc)

    return model_path


if __name__ == "__main__":
    run()
