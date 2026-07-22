"""API HTTP de inferência (FastAPI): serve predições de recompra.

Segue a convenção de plataformas de model serving em produção (KServe,
Seldon, BentoML): endpoints distintos para liveness, readiness,
inferência, metadados e métricas operacionais.

Endpoints:
    GET  /              — informações básicas e links para os demais.
    GET  /health         — liveness: o processo está vivo?
    GET  /ready           — readiness: o modelo está carregado?
    GET  /metadata         — versão/métricas do modelo + JSON Schema.
    GET  /metrics            — métricas operacionais (formato Prometheus).
    POST /predict              — predição para um par usuário-produto.
    POST /predict/batch          — predição em lote (até 500 itens).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from recommender.api.inference import InferenceService
from recommender.api.logging_config import logger
from recommender.api.metrics import (
    record_prediction,
    record_request,
    render_prometheus_metrics,
)
from recommender.api.model_registry import sync_model_from_gcs
from recommender.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    MetadataResponse,
    ModelMetadataInfo,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)
from recommender.config.settings import get_settings

_service: InferenceService | None = None
_settings = get_settings()

_COLD_START_DETAIL = (
    "user_id ou product_id desconhecido (fora do vocabulário de treino)."
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Baixa o modelo (se `MODEL_BUCKET` setado) e o carrega na subida da API."""
    global _service
    models_dir = Path(_settings.models_dir)
    sync_model_from_gcs(_settings.model_bucket, models_dir)
    _service = InferenceService(models_dir, _settings.device)
    logger.info("Modelo carregado e API pronta.")
    yield
    _service = None


app = FastAPI(
    title="Instacart Recommender API",
    description=(
        "Prevê a probabilidade de um usuário recomprar um produto, usando "
        "o modelo híbrido (embeddings + MLP) registrado no MLflow Model "
        "Registry, stage Production. Projeto do Tech Challenge Fase 02 — "
        "FIAP MLE10."
    ),
    version=_settings.model_version,
    lifespan=lifespan,
    contact={"name": "Tiago de Freitas Faustino"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):  # noqa: ANN001, ANN201
    """Adiciona request-id/latência aos headers, loga e registra métricas."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration * 1000:.2f}"
    record_request(request.method, request.url.path, response.status_code, duration)
    logger.info(
        "Requisição concluída.",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        },
    )
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Captura qualquer exceção não tratada; nunca expõe detalhes ao cliente."""
    logger.exception("Erro não tratado.", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500, content={"detail": "Erro interno. Tente novamente."}
    )


def _require_service() -> InferenceService:
    """Garante que o modelo já carregou; levanta 503 caso contrário."""
    if _service is None:
        raise HTTPException(status_code=503, detail="Modelo ainda carregando.")
    return _service


@app.get("/", tags=["Meta"], summary="Informações básicas da API")
def root() -> dict[str, Any]:
    """Ponto de entrada da API — aponta para a documentação interativa."""
    return {
        "name": "Instacart Recommender API",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "model_version": _settings.model_version,
    }


@app.get(
    "/health", tags=["Meta"], summary="Liveness probe", response_model=HealthResponse
)
def health() -> HealthResponse:
    """Liveness — confirma que o processo está vivo (rápido, não depende do modelo)."""
    return HealthResponse(status="ok")


@app.get(
    "/ready", tags=["Meta"], summary="Readiness probe", response_model=ReadyResponse
)
def ready() -> ReadyResponse:
    """Readiness — confirma que o modelo está carregado e pronto para inferência."""
    loaded = _service is not None
    return ReadyResponse(status="ready" if loaded else "loading", model_loaded=loaded)


@app.get(
    "/metadata",
    tags=["Meta"],
    summary="Metadados do modelo + JSON Schema de entrada/saída",
    response_model=MetadataResponse,
)
def metadata() -> MetadataResponse:
    """Versão/métricas do modelo em produção, e o contrato de `/predict`."""
    service = _require_service()
    return MetadataResponse(
        model_info=ModelMetadataInfo(
            model_version=_settings.model_version, **service.info()
        ),
        input_schema=PredictRequest.model_json_schema(),
        output_schema=PredictResponse.model_json_schema(),
    )


@app.get("/metrics", tags=["Meta"], summary="Métricas operacionais (Prometheus)")
def metrics() -> Response:
    """Métricas operacionais da API, no formato de texto do Prometheus."""
    body, content_type = render_prometheus_metrics()
    return Response(content=body, media_type=content_type)


@app.post(
    "/predict",
    tags=["Predição"],
    summary="Prevê a probabilidade de recompra de um par usuário-produto",
    response_model=PredictResponse,
    responses={
        422: {"model": ErrorResponse, "description": "user_id/product_id desconhecido"},
        503: {"model": ErrorResponse, "description": "Modelo ainda carregando"},
    },
)
def predict(request: PredictRequest) -> PredictResponse:
    """Prevê a probabilidade de recompra para um único par usuário-produto."""
    service = _require_service()
    encoded = service.encode_ids(request.user_id, request.product_id)
    if encoded is None:
        raise HTTPException(status_code=422, detail=_COLD_START_DETAIL)

    probability = service.predict(*encoded, _feature_list(request))
    record_prediction(probability)
    return PredictResponse(
        reorder_probability=probability, model_version=_settings.model_version
    )


@app.post(
    "/predict/batch",
    tags=["Predição"],
    summary="Prevê a recompra para até 500 pares usuário-produto de uma vez",
    response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK,
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Prevê em lote (um único forward pass) — mais eficiente que N chamadas a /predict.

    Pares cold-start (user_id/product_id fora do vocabulário) retornam
    `reorder_probability: null` em vez de derrubar a requisição inteira.
    """
    service = _require_service()
    raw_items = [
        (item.user_id, item.product_id, _feature_list(item)) for item in request.items
    ]
    probabilities = service.predict_batch(raw_items)
    for probability in probabilities:
        record_prediction(probability)

    results = [
        PredictResponse(reorder_probability=p, model_version=_settings.model_version)
        for p in probabilities
    ]
    return BatchPredictResponse(results=results)


def _feature_list(request: PredictRequest) -> list[float]:
    """Extrai as features tabulares do request, na ordem que o modelo espera."""
    return [
        request.purchase_count,
        request.days_since_last_order,
        request.order_hour_of_day,
        request.order_dow,
        request.basket_size,
    ]
