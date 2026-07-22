"""Testes de integração da API de inferência (via TestClient, sem rede real).

Depende de um modelo já treinado em `models/` (model.pt + encoders +
vocab_sizes.json) — roda como parte da suíte só quando esses artefatos
existem localmente, já que a API os carrega na subida.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODELS_DIR = Path("models")
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "model.pt").exists(),
    reason="Requer um modelo treinado em models/ (rode o pipeline primeiro).",
)

_PAYLOAD = {
    "user_id": 1,
    "product_id": 100,
    "purchase_count": 3,
    "days_since_last_order": 7,
    "order_hour_of_day": 10,
    "order_dow": 2,
    "basket_size": 8,
}


@pytest.fixture
def client():  # noqa: ANN201
    """Cliente de teste com o lifespan da API executado (carrega o modelo)."""
    from recommender.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_root_returns_basic_api_info(client) -> None:  # noqa: ANN001
    """A raiz deve responder com nome da API e links para health/ready/docs."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["docs"] == "/docs"
    assert body["ready"] == "/ready"


def test_health_is_ok_regardless_of_model_state(client) -> None:  # noqa: ANN001
    """Liveness deve responder 'ok' independente do modelo (é só o processo vivo)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_model_loaded_true(client) -> None:  # noqa: ANN001
    """Readiness deve reportar model_loaded=True após o lifespan carregar o modelo."""
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True


def test_metadata_exposes_model_info_and_json_schemas(client) -> None:  # noqa: ANN001
    """/metadata deve expor versão/vocabulário do modelo e os JSON Schemas."""
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_info"]["num_users"] > 0
    assert "properties" in body["input_schema"]
    assert "properties" in body["output_schema"]


def test_metrics_endpoint_exposes_prometheus_format(client) -> None:  # noqa: ANN001
    """/metrics deve responder no formato de texto do Prometheus."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "recommender_api_requests_total" in response.text


def test_predict_returns_probability_between_zero_and_one(client) -> None:  # noqa: ANN001
    """Uma predição válida deve retornar uma probabilidade em [0, 1]."""
    response = client.post("/predict", json=_PAYLOAD)
    assert response.status_code == 200
    probability = response.json()["reorder_probability"]
    assert 0.0 <= probability <= 1.0


def test_predict_response_has_observability_headers(client) -> None:  # noqa: ANN001
    """Toda resposta deve trazer X-Request-ID e X-Process-Time-Ms."""
    response = client.post("/predict", json=_PAYLOAD)
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers


def test_predict_returns_422_for_unknown_user(client) -> None:  # noqa: ANN001
    """Um user_id fora do vocabulário de treino deve retornar 422, não 500."""
    response = client.post("/predict", json={**_PAYLOAD, "user_id": 999_999_999})
    assert response.status_code == 422


def test_predict_batch_returns_one_result_per_item_in_order(client) -> None:  # noqa: ANN001
    """O lote deve retornar exatamente um resultado por item, na mesma ordem."""
    cold_start_item = {**_PAYLOAD, "user_id": 999_999_999}
    response = client.post(
        "/predict/batch", json={"items": [_PAYLOAD, cold_start_item]}
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert 0.0 <= results[0]["reorder_probability"] <= 1.0


def test_predict_batch_returns_null_for_cold_start_items(client) -> None:  # noqa: ANN001
    """Itens cold-start no lote não devem derrubar a requisição inteira."""
    cold_start_item = {**_PAYLOAD, "product_id": 999_999_999}
    response = client.post("/predict/batch", json={"items": [cold_start_item]})
    assert response.status_code == 200
    assert response.json()["results"][0]["reorder_probability"] is None


def test_predict_batch_rejects_more_than_500_items(client) -> None:  # noqa: ANN001
    """O lote deve recusar mais de 500 itens (limite de payload) com 422."""
    response = client.post(
        "/predict/batch", json={"items": [_PAYLOAD] * 501}
    )
    assert response.status_code == 422


def test_cors_header_present_on_response(client) -> None:  # noqa: ANN001
    """A API deve responder com CORS habilitado, para consumo via navegador."""
    response = client.get("/health", headers={"Origin": "http://example.com"})
    assert response.headers.get("access-control-allow-origin") == "*"
