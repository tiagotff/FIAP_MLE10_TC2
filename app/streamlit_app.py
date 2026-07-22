"""Dashboard Streamlit: cliente visual da API de inferência.

Nenhuma lógica de ML roda neste app — ele só envia requisições à API
(`src/recommender/api/`) e exibe o resultado. Pensado para um usuário de
negócio sem conhecimento técnico de APIs.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000"
CSV_REQUIRED_COLUMNS = [
    "user_id",
    "product_id",
    "purchase_count",
    "days_since_last_order",
    "order_hour_of_day",
    "order_dow",
    "basket_size",
]
BATCH_LIMIT = 500


def resolve_api_url() -> str:
    """Resolve a URL da API a partir da variável de ambiente `RECOMMENDER_API_URL`."""
    return os.environ.get("RECOMMENDER_API_URL", DEFAULT_API_URL)


def fetch_ready_status(api_url: str) -> dict[str, Any] | None:
    """Consulta `/ready`; retorna `None` se a API não responder."""
    try:
        response = requests.get(f"{api_url}/ready", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def fetch_metadata(api_url: str) -> dict[str, Any] | None:
    """Consulta `/metadata`; retorna `None` se a API não responder."""
    try:
        response = requests.get(f"{api_url}/metadata", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def call_predict(api_url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Chama `POST /predict`; retorna `(status_code, corpo_da_resposta)`."""
    response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    return response.status_code, response.json()


def call_predict_batch(
    api_url: str, items: list[dict[str, Any]]
) -> tuple[int, dict[str, Any]]:
    """Chama `POST /predict/batch`; retorna `(status_code, corpo_da_resposta)`."""
    response = requests.post(
        f"{api_url}/predict/batch", json={"items": items}, timeout=30
    )
    return response.status_code, response.json()


def missing_csv_columns(df: pd.DataFrame) -> list[str]:
    """Retorna quais colunas obrigatórias faltam no CSV enviado."""
    return [col for col in CSV_REQUIRED_COLUMNS if col not in df.columns]


def dataframe_to_predict_items(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Converte as linhas do CSV no formato de item esperado por `/predict/batch`."""
    return df[CSV_REQUIRED_COLUMNS].to_dict(orient="records")


def _render_sidebar(api_url: str) -> None:
    """Renderiza o status da API e o resumo do modelo na sidebar."""
    st.sidebar.title("Instacart Recommender")
    st.sidebar.caption(f"API: {api_url}")

    ready = fetch_ready_status(api_url)
    if ready and ready.get("model_loaded"):
        st.sidebar.success("✅ Modelo carregado e pronto")
    else:
        st.sidebar.error("❌ Não foi possível conectar à API")
        return

    metadata = fetch_metadata(api_url)
    if not metadata:
        return
    info = metadata["model_info"]
    st.sidebar.metric("Versão do modelo", info["model_version"])
    st.sidebar.metric("Usuários no vocabulário", f"{info['num_users']:,}")
    st.sidebar.metric("Produtos no vocabulário", f"{info['num_products']:,}")
    metrics = info.get("metrics") or {}
    if metrics.get("auc_roc"):
        st.sidebar.metric("AUC-ROC (validação)", f"{metrics['auc_roc']:.4f}")


def _render_single_prediction_tab(api_url: str) -> None:
    """Formulário para avaliar um único par usuário-produto via /predict."""
    st.subheader("Avaliar um par usuário-produto")
    with st.form("single_prediction"):
        col1, col2 = st.columns(2)
        user_id = col1.number_input("user_id", min_value=0, step=1)
        product_id = col2.number_input("product_id", min_value=0, step=1)
        purchase_count = col1.number_input("purchase_count", min_value=0.0, value=1.0)
        days_since_last_order = col2.number_input(
            "days_since_last_order", min_value=0.0, value=7.0
        )
        order_hour_of_day = col1.slider("order_hour_of_day", 0, 23, 10)
        order_dow = col2.slider("order_dow (0=domingo)", 0, 6, 2)
        basket_size = st.number_input("basket_size", min_value=1.0, value=8.0)
        submitted = st.form_submit_button("Prever probabilidade de recompra")

    if not submitted:
        return
    payload = {
        "user_id": int(user_id),
        "product_id": int(product_id),
        "purchase_count": purchase_count,
        "days_since_last_order": days_since_last_order,
        "order_hour_of_day": order_hour_of_day,
        "order_dow": order_dow,
        "basket_size": basket_size,
    }
    status_code, body = call_predict(api_url, payload)
    if status_code == 200:
        probability = body["reorder_probability"]
        st.metric("Probabilidade de recompra", f"{probability:.1%}")
        st.progress(probability)
    else:
        st.error(body.get("detail", "Erro ao consultar a API."))


def _render_batch_tab(api_url: str) -> None:
    """Upload de CSV com até 500 pares, avaliados via /predict/batch."""
    st.subheader(f"Avaliar um lote via CSV (até {BATCH_LIMIT} linhas)")
    st.caption("Colunas esperadas: " + ", ".join(CSV_REQUIRED_COLUMNS))
    uploaded = st.file_uploader("Envie um CSV", type="csv")
    if uploaded is None:
        return

    df = pd.read_csv(uploaded)
    missing = missing_csv_columns(df)
    if missing:
        st.error(f"Colunas faltando no CSV: {', '.join(missing)}")
        return
    if len(df) > BATCH_LIMIT:
        st.error(f"O CSV tem {len(df)} linhas; o limite é {BATCH_LIMIT}.")
        return

    status_code, body = call_predict_batch(api_url, dataframe_to_predict_items(df))
    if status_code != 200:
        st.error(body.get("detail", "Erro ao consultar a API."))
        return

    results_df = df.copy()
    results_df["reorder_probability"] = [
        r["reorder_probability"] for r in body["results"]
    ]
    results_df = results_df.sort_values("reorder_probability", ascending=False)
    st.dataframe(results_df)
    st.download_button(
        "Baixar resultado em CSV",
        results_df.to_csv(index=False),
        file_name="predicoes.csv",
    )


def main() -> None:
    """Ponto de entrada do dashboard."""
    st.set_page_config(page_title="Instacart Recommender", page_icon="🛒")
    api_url = resolve_api_url()
    _render_sidebar(api_url)

    st.title("🛒 Instacart Recommender — Dashboard")
    tab_single, tab_batch = st.tabs(["Predição única", "Lote (CSV)"])
    with tab_single:
        _render_single_prediction_tab(api_url)
    with tab_batch:
        _render_batch_tab(api_url)


if __name__ == "__main__":
    main()
