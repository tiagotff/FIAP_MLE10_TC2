"""Logging estruturado (JSON) para a API — nenhum módulo de produção usa `print()`."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formata cada registro de log como uma linha JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializa o LogRecord como JSON, incluindo extras passados via `extra=`."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "path", "method", "status_code", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura o logger raiz da API para emitir JSON em stdout.

    Args:
        level: Nível mínimo de log a ser emitido.

    Returns:
        O logger nomeado `recommender.api`, pronto para uso.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger("recommender.api")
    root.setLevel(level)
    root.handlers = [handler]
    root.propagate = False
    return root


logger = configure_logging()
