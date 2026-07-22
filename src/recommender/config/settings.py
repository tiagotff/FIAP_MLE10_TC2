"""Settings carregadas de variáveis de ambiente via Pydantic Settings.

Diferente das dataclasses em `model_config.py` (que descrevem
hiperparâmetros de arquitetura/treino), este módulo cobre configurações
de ambiente de execução: onde estão os dados, para onde o MLflow reporta,
qual dispositivo usar. Mantém o princípio de Dependency Inversion: o
restante do código depende de `Settings`, nunca lê `os.environ`
diretamente.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações de ambiente, carregadas de variáveis de ambiente ou `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="URI do servidor de tracking do MLflow.",
    )
    data_raw_dir: str = Field(
        default="data/raw",
        description="Diretório dos dados brutos do Instacart.",
    )
    data_processed_dir: str = Field(
        default="data/processed",
        description="Diretório dos dados já pré-processados.",
    )
    models_dir: str = Field(
        default="models",
        description="Diretório onde artefatos de modelo são salvos.",
    )
    random_seed: int = Field(
        default=42,
        description="Seed usada para reprodutibilidade (torch/numpy/random).",
    )
    device: str = Field(
        default="cpu",
        description="Dispositivo de treino/inferência: 'cpu' ou 'cuda'.",
    )
    model_version: str = Field(
        default="unknown",
        description=(
            "Versão do modelo servido pela API — setada no build/deploy "
            "da imagem, não consultada dinamicamente do MLflow em runtime "
            "(a API não deve depender do tracking server estar no ar)."
        ),
    )
    model_bucket: str | None = Field(
        default=None,
        description=(
            "Bucket GCS de onde a API baixa os artefatos do modelo na "
            "subida. Se None, usa os artefatos já presentes localmente "
            "em `models_dir` (desenvolvimento local)."
        ),
    )


def get_settings() -> Settings:
    """Retorna uma instância de `Settings` carregada do ambiente atual.

    Função isolada (em vez de uma instância global no módulo) para
    facilitar substituição em testes via monkeypatch/fixture.
    """
    return Settings()
