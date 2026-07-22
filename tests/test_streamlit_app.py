"""Testes unitários da lógica do dashboard Streamlit (sem UI real).

Testa só as funções puras (resolução de URL, wrappers HTTP, validação
de CSV) — não a renderização em si, que exige um processo Streamlit
rodando.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from app.streamlit_app import (
    CSV_REQUIRED_COLUMNS,
    DEFAULT_API_URL,
    _classify_probability,
    call_predict,
    dataframe_to_predict_items,
    fetch_ready_status,
    missing_csv_columns,
    resolve_api_url,
)


def test_resolve_api_url_uses_default_when_env_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem RECOMMENDER_API_URL definida, deve usar o default local."""
    monkeypatch.delenv("RECOMMENDER_API_URL", raising=False)
    assert resolve_api_url() == DEFAULT_API_URL


def test_resolve_api_url_respects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com RECOMMENDER_API_URL definida, deve usar o valor da variável."""
    monkeypatch.setenv("RECOMMENDER_API_URL", "https://minha-api.run.app")
    assert resolve_api_url() == "https://minha-api.run.app"


def test_missing_csv_columns_detects_all_absent_columns() -> None:
    """Deve listar todas as colunas obrigatórias que faltam no CSV."""
    df = pd.DataFrame({"user_id": [1], "product_id": [2]})
    missing = missing_csv_columns(df)
    assert set(missing) == set(CSV_REQUIRED_COLUMNS) - {"user_id", "product_id"}


def test_missing_csv_columns_returns_empty_when_csv_is_complete() -> None:
    """Um CSV com todas as colunas esperadas não deve reportar nada faltando."""
    df = pd.DataFrame({col: [0] for col in CSV_REQUIRED_COLUMNS})
    assert missing_csv_columns(df) == []


def test_dataframe_to_predict_items_matches_predict_batch_schema() -> None:
    """Cada linha do CSV deve virar um dict com as chaves de /predict/batch."""
    df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "product_id": [10, 20],
            "purchase_count": [3, 1],
            "days_since_last_order": [7, 14],
            "order_hour_of_day": [10, 15],
            "order_dow": [2, 5],
            "basket_size": [8, 4],
        }
    )
    items = dataframe_to_predict_items(df)
    assert len(items) == 2
    assert items[0]["user_id"] == 1
    assert set(items[0].keys()) == set(CSV_REQUIRED_COLUMNS)


def test_fetch_ready_status_returns_none_when_api_unreachable() -> None:
    """Se a API não responder, deve retornar None (não deve levantar exceção)."""
    with patch(
        "app.streamlit_app.requests.get", side_effect=requests.ConnectionError
    ):
        assert fetch_ready_status("http://api-inexistente") is None


def test_fetch_ready_status_returns_parsed_json_on_success() -> None:
    """Com a API respondendo, deve retornar o JSON já parseado."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "ready", "model_loaded": True}
    with patch("app.streamlit_app.requests.get", return_value=mock_response):
        result = fetch_ready_status(DEFAULT_API_URL)
    assert result == {"status": "ready", "model_loaded": True}


def test_call_predict_returns_status_code_and_body() -> None:
    """O wrapper de /predict deve repassar o status code e o corpo da resposta."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"reorder_probability": 0.7, "model_version": "1"}
    with patch("app.streamlit_app.requests.post", return_value=mock_response):
        status_code, body = call_predict(DEFAULT_API_URL, {"user_id": 1})
    assert status_code == 200
    assert body["reorder_probability"] == 0.7


@pytest.mark.parametrize(
    ("probability", "expected_class"),
    [(0.9, "high"), (0.66, "high"), (0.5, "medium"), (0.33, "medium"), (0.1, "low")],
)
def test_classify_probability_thresholds(
    probability: float, expected_class: str
) -> None:
    """A classificação por faixa deve respeitar os limiares 0.66/0.33."""
    css_class, _ = _classify_probability(probability)
    assert css_class == expected_class
