"""Valida se o ambiente está pronto para desenvolvimento/treino.

Checa:
1. Versão do Python compatível.
2. Bibliotecas obrigatórias importáveis (torch, sklearn, mlflow, dvc...).
3. Settings carregam corretamente a partir do `.env`.
4. Diretórios de dados/modelos existem (ou podem ser criados).

Uso:
    python scripts/validate_env.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED_MODULES = [
    "torch",
    "sklearn",
    "mlflow",
    "dvc",
    "pandas",
    "numpy",
    "pydantic",
    "pydantic_settings",
    "yaml",
]

MIN_PYTHON = (3, 11)


def check_python_version() -> bool:
    """Confere se a versão do Python é >= MIN_PYTHON."""
    ok = sys.version_info >= MIN_PYTHON
    status = "OK" if ok else "FALHOU"
    print(f"[{status}] Python {sys.version.split()[0]} (mínimo exigido: 3.11)")
    return ok


def check_required_modules() -> bool:
    """Confere se todas as bibliotecas obrigatórias estão instaladas."""
    all_ok = True
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name} importado com sucesso")
        except ImportError as exc:
            print(f"[FALHOU] {module_name} não encontrado: {exc}")
            all_ok = False
    return all_ok


def check_settings_load() -> bool:
    """Confere se as Settings carregam sem erro a partir do ambiente/.env."""
    try:
        from recommender.config.settings import get_settings

        settings = get_settings()
        print(f"[OK] Settings carregadas (device={settings.device!r})")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FALHOU] Erro ao carregar Settings: {exc}")
        return False


def check_data_dirs() -> bool:
    """Confere se os diretórios esperados de dados/modelos existem."""
    all_ok = True
    for directory in ("data/raw", "data/processed", "models"):
        path = Path(directory)
        exists = path.is_dir()
        status = "OK" if exists else "FALHOU"
        print(f"[{status}] Diretório '{directory}' {'existe' if exists else 'ausente'}")
        all_ok = all_ok and exists
    return all_ok


def main() -> int:
    """Executa todas as validações e retorna o código de saída do processo."""
    checks = [
        check_python_version(),
        check_required_modules(),
        check_settings_load(),
        check_data_dirs(),
    ]
    if all(checks):
        print("\nAmbiente validado com sucesso.")
        return 0
    print("\nAmbiente incompleto — corrija os itens marcados como FALHOU acima.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
