"""Schemas de request/response da API de inferência (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field

_EXAMPLE_PREDICT_PAYLOAD = {
    "user_id": 1,
    "product_id": 196,
    "purchase_count": 3,
    "days_since_last_order": 7,
    "order_hour_of_day": 10,
    "order_dow": 2,
    "basket_size": 8,
}


class PredictRequest(BaseModel):
    """Payload de entrada para uma predição de recompra."""

    user_id: int = Field(..., description="ID original do usuário (Instacart).")
    product_id: int = Field(..., description="ID original do produto (Instacart).")
    purchase_count: float = Field(
        ..., description="Nº de vezes que o usuário comprou esse produto."
    )
    days_since_last_order: float = Field(
        ..., description="Dias desde o último pedido do usuário."
    )
    order_hour_of_day: float = Field(..., description="Hora do pedido (0-23).")
    order_dow: float = Field(..., description="Dia da semana do pedido (0-6).")
    basket_size: float = Field(..., description="Nº de itens no carrinho.")

    model_config = {"json_schema_extra": {"example": _EXAMPLE_PREDICT_PAYLOAD}}


class PredictResponse(BaseModel):
    """Resultado de uma predição de recompra."""

    reorder_probability: float | None = Field(
        ...,
        description=(
            "Probabilidade de recompra, entre 0 e 1. `null` se o par "
            "usuário-produto for cold-start (só em respostas de lote)."
        ),
    )
    model_version: str = Field(..., description="Versão do modelo usada na predição.")

    model_config = {
        "json_schema_extra": {
            "example": {"reorder_probability": 0.73, "model_version": "1"}
        }
    }


class BatchPredictRequest(BaseModel):
    """Payload de entrada para predição em lote (um forward pass só)."""

    items: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Lista de pares usuário-produto a avaliar (máx. 500).",
    )

    model_config = {
        "json_schema_extra": {"example": {"items": [_EXAMPLE_PREDICT_PAYLOAD]}}
    }


class BatchPredictResponse(BaseModel):
    """Resultado de uma predição em lote, na mesma ordem da requisição."""

    results: list[PredictResponse] = Field(
        ..., description="Um resultado por item enviado, na mesma ordem."
    )


class HealthResponse(BaseModel):
    """Resposta do endpoint de liveness (`/health`) — não depende do modelo."""

    status: str = Field(..., description="'ok' enquanto o processo estiver vivo.")


class ReadyResponse(BaseModel):
    """Resposta do endpoint de readiness (`/ready`) — depende do modelo carregado."""

    status: str = Field(..., description="'ready' quando o modelo já foi carregado.")
    model_loaded: bool = Field(..., description="Se o modelo já está pronto para uso.")


class ModelMetadataInfo(BaseModel):
    """Bloco de metadados do modelo dentro de `/metadata`."""

    model_version: str
    num_users: int
    num_products: int
    embedding_dim: int
    mlp_hidden_dims: list[int]
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Métricas de avaliação do modelo (auc_roc, recall, etc.), "
            "se disponíveis."
        ),
    )


class MetadataResponse(BaseModel):
    """Metadados completos do modelo em produção + contrato de entrada/saída."""

    model_info: ModelMetadataInfo
    input_schema: dict = Field(
        ..., description="JSON Schema esperado por POST /predict."
    )
    output_schema: dict = Field(
        ..., description="JSON Schema retornado por POST /predict."
    )


class ErrorResponse(BaseModel):
    """Formato padrão de erro (usado na documentação OpenAPI)."""

    detail: str
