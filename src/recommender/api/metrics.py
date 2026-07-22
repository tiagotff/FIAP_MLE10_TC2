"""Métricas operacionais da API, expostas em `/metrics` no formato Prometheus."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS_TOTAL = Counter(
    "recommender_api_requests_total",
    "Total de requisições recebidas pela API.",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "recommender_api_request_latency_seconds",
    "Latência das requisições, por rota.",
    ["method", "path"],
)

PREDICTIONS_TOTAL = Counter(
    "recommender_predictions_total",
    "Total de predições feitas, por faixa de probabilidade.",
    ["reorder_likelihood"],
)


def record_request(
    method: str, path: str, status_code: int, duration_seconds: float
) -> None:
    """Registra uma requisição concluída (contagem + latência)."""
    REQUESTS_TOTAL.labels(method=method, path=path, status_code=str(status_code)).inc()
    REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(duration_seconds)


def record_prediction(probability: float | None) -> None:
    """Registra uma predição, classificada por faixa de probabilidade.

    Args:
        probability: Probabilidade de recompra, ou `None` para casos
            cold-start (não contabilizados, já que não houve predição
            de fato).
    """
    if probability is None:
        return
    likelihood = "high" if probability >= 0.5 else "low"
    PREDICTIONS_TOTAL.labels(reorder_likelihood=likelihood).inc()


def render_prometheus_metrics() -> tuple[bytes, str]:
    """Serializa as métricas atuais no formato de texto do Prometheus."""
    return generate_latest(), CONTENT_TYPE_LATEST
