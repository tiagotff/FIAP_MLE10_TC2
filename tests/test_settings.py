"""Testes unitários para Settings (Pydantic Settings)."""

from __future__ import annotations

from recommender.config.settings import Settings, get_settings


def test_settings_have_sane_defaults() -> None:
    """Sem variáveis de ambiente definidas, os defaults devem ser aplicados."""
    settings = Settings(_env_file=None)
    assert settings.device == "cpu"
    assert settings.random_seed == 42
    assert settings.data_raw_dir == "data/raw"


def test_settings_respect_environment_override(monkeypatch) -> None:  # noqa: ANN001
    """Variáveis de ambiente devem sobrescrever os valores default."""
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("RANDOM_SEED", "7")
    settings = Settings(_env_file=None)
    assert settings.device == "cuda"
    assert settings.random_seed == 7


def test_get_settings_returns_settings_instance() -> None:
    """get_settings() deve retornar uma instância válida de Settings."""
    settings = get_settings()
    assert isinstance(settings, Settings)
