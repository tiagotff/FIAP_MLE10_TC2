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


# ---------------------------------------------------------------------------
# Camada de apresentação (UI) — tudo abaixo usa st.*; a lógica testável
# fica só nas funções acima, sem nenhuma dependência do Streamlit.
# ---------------------------------------------------------------------------

_PRIMARY_COLOR = "#0F9D58"
_CUSTOM_CSS = f"""
<style>
    .block-container {{ padding-top: 2rem; max-width: 1100px; }}
    .app-header {{
        display: flex; align-items: center; gap: 0.75rem;
        margin-bottom: 0.25rem;
    }}
    .app-header .emoji {{ font-size: 2.5rem; line-height: 1; }}
    .app-header h1 {{ margin: 0; padding: 0; font-size: 2rem; }}
    .app-subtitle {{ color: #6b7280; margin-bottom: 1.5rem; }}
    div[data-testid="stMetric"] {{
        background-color: #f8f9fa; border-radius: 0.5rem;
        padding: 0.75rem 1rem; border: 1px solid #e5e7eb;
    }}
    .result-card {{
        border-radius: 0.75rem; padding: 1.5rem; margin-top: 1rem;
        border: 1px solid;
    }}
    .result-card.high {{ background-color: #ecfdf3; border-color: {_PRIMARY_COLOR}; }}
    .result-card.medium {{ background-color: #fffbeb; border-color: #d97706; }}
    .result-card.low {{ background-color: #fef2f2; border-color: #dc2626; }}
    .result-card .value {{ font-size: 2.5rem; font-weight: 700; margin: 0.25rem 0; }}
    .result-card .label {{ font-size: 0.9rem; color: #4b5563; margin: 0; }}
    .status-line {{
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.9rem; margin: 0.5rem 0 0.35rem;
    }}
    .status-dot {{
        width: 0.55rem; height: 0.55rem; border-radius: 50%; flex-shrink: 0;
    }}
    .status-dot.online {{ background-color: {_PRIMARY_COLOR}; }}
    .status-dot.offline {{ background-color: #dc2626; }}
    .metric-card {{
        background-color: #f8f9fa; border: 1px solid #e5e7eb;
        border-radius: 0.5rem; padding: 0.5rem 0.4rem; text-align: center;
    }}
    .metric-card .icon {{ font-size: 1.1rem; }}
    .metric-card .value {{ font-size: 1.25rem; font-weight: 700; margin: 0.1rem 0; }}
    .metric-card .label {{ font-size: 0.75rem; color: #6b7280; }}
</style>
"""

_RISK_LEVELS = (
    (0.66, "high", "🟢 Alta probabilidade de recompra"),
    (0.33, "medium", "🟡 Probabilidade média de recompra"),
    (0.0, "low", "🔴 Baixa probabilidade de recompra"),
)


def _classify_probability(probability: float) -> tuple[str, str]:
    """Classifica a probabilidade numa faixa (css_class, rótulo em texto)."""
    for threshold, css_class, label in _RISK_LEVELS:
        if probability >= threshold:
            return css_class, label
    return "low", _RISK_LEVELS[-1][2]


def _render_header() -> None:
    """Cabeçalho da página, com título e subtítulo."""
    st.markdown(
        '<div class="app-header"><span class="emoji">🛒</span>'
        "<h1>Instacart Recommender</h1></div>"
        '<p class="app-subtitle">Probabilidade de recompra por par '
        "usuário-produto, servida em tempo real pela API de inferência.</p>",
        unsafe_allow_html=True,
    )


def _render_sidebar(api_url: str) -> None:
    """Renderiza o status da conexão e o resumo do modelo na sidebar."""
    st.sidebar.markdown("### 🛒 Instacart Recommender")

    ready = fetch_ready_status(api_url)
    connected = bool(ready and ready.get("model_loaded"))
    dot = "status-dot online" if connected else "status-dot offline"
    label = "Conectado" if connected else "Sem conexão com a API"
    st.sidebar.markdown(
        f'<div class="status-line"><span class="{dot}"></span>{label}</div>',
        unsafe_allow_html=True,
    )
    # st.code em vez de markdown: evita o auto-link do Streamlit sobre a URL,
    # que sobrescreveria o estilo customizado com o azul/sublinhado padrão.
    st.sidebar.code(api_url, language=None)

    if not connected:
        return

    metadata = fetch_metadata(api_url)
    if not metadata:
        return
    st.sidebar.divider()
    _render_sidebar_model_summary(metadata["model_info"])


def _render_metric_card(icon: str, value: str, label: str) -> str:
    """Monta o HTML de um card de métrica compacto, com ícone."""
    return (
        f'<div class="metric-card"><div class="icon">{icon}</div>'
        f'<div class="value">{value}</div><div class="label">{label}</div></div>'
    )


def _render_sidebar_model_summary(info: dict[str, Any]) -> None:
    """Mostra versão, vocabulário e métricas do modelo na sidebar."""
    version = info.get("model_version")
    version_label = (
        f"v{version}" if version and version != "unknown" else "ambiente local"
    )
    st.sidebar.caption(f"Modelo em produção · **{version_label}**")

    col1, col2 = st.sidebar.columns(2)
    col1.markdown(
        _render_metric_card("👥", f"{info['num_users']:,}", "Usuários"),
        unsafe_allow_html=True,
    )
    col2.markdown(
        _render_metric_card("📦", f"{info['num_products']:,}", "Produtos"),
        unsafe_allow_html=True,
    )

    metrics = info.get("metrics") or {}
    if not metrics:
        st.sidebar.caption("Métricas de avaliação não publicadas neste ambiente.")
        return
    with st.sidebar.expander("📊 Métricas de validação"):
        for name, value in metrics.items():
            st.write(f"**{name.replace('_', ' ').upper()}**: {value:.4f}")


def _render_result_card(probability: float) -> None:
    """Card colorido com o resultado da predição, por faixa de risco."""
    css_class, label = _classify_probability(probability)
    st.markdown(
        f'<div class="result-card {css_class}">'
        f'<p class="label">{label}</p>'
        f'<p class="value">{probability:.1%}</p>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.progress(probability)


def _single_prediction_form() -> dict[str, Any] | None:
    """Renderiza o formulário de predição única; retorna o payload se enviado."""
    with st.container(border=True):
        st.markdown("#### Identificação")
        col1, col2 = st.columns(2)
        user_id_raw = col1.text_input(
            "user_id", value="1", help="ID do usuário no Instacart."
        )
        product_id_raw = col2.text_input(
            "product_id", value="196", help="ID do produto no Instacart."
        )

        st.markdown("#### Contexto da compra")
        col1, col2 = st.columns(2)
        purchase_count = col1.number_input(
            "Nº de vezes que já comprou este produto",
            min_value=0.0, value=3.0, step=1.0,
        )
        days_since_last_order = col2.number_input(
            "Dias desde o último pedido", min_value=0.0, value=7.0, step=1.0
        )
        order_hour_of_day = col1.slider("Hora do pedido", 0, 23, 10)
        order_dow = col2.slider(
            "Dia da semana do pedido (0–6, conforme o dataset)", 0, 6, 2
        )
        basket_size = st.slider("Tamanho do carrinho (itens)", 1, 50, 8)

        submitted = st.button(
            "Prever probabilidade de recompra",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None
    if not user_id_raw.strip().isdigit() or not product_id_raw.strip().isdigit():
        st.error("user_id e product_id precisam ser números inteiros.")
        return None

    return {
        "user_id": int(user_id_raw),
        "product_id": int(product_id_raw),
        "purchase_count": purchase_count,
        "days_since_last_order": days_since_last_order,
        "order_hour_of_day": order_hour_of_day,
        "order_dow": order_dow,
        "basket_size": basket_size,
    }


def _render_single_prediction_tab(api_url: str) -> None:
    """Aba de predição para um único par usuário-produto."""
    payload = _single_prediction_form()
    if payload is None:
        return

    status_code, body = call_predict(api_url, payload)
    if status_code == 200:
        _render_result_card(body["reorder_probability"])
    elif status_code == 422:
        st.warning(
            "⚠️ user_id ou product_id não existe no vocabulário de treino "
            "(cold-start) — o modelo não tem histórico suficiente para "
            "esse par."
        )
    else:
        st.error(body.get("detail", "Erro ao consultar a API."))


def _example_csv_bytes() -> str:
    """Gera um CSV de exemplo, no formato esperado pelo upload em lote."""
    example = pd.DataFrame(
        [
            {
                "user_id": 1, "product_id": 196, "purchase_count": 3,
                "days_since_last_order": 7, "order_hour_of_day": 10,
                "order_dow": 2, "basket_size": 8,
            }
        ]
    )
    return example.to_csv(index=False)


def _render_batch_summary(probabilities: list[float]) -> None:
    """Métricas-resumo do lote avaliado (média, contagem por faixa)."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Itens avaliados", len(probabilities))
    col2.metric("Probabilidade média", f"{pd.Series(probabilities).mean():.1%}")
    high_count = sum(1 for p in probabilities if p >= 0.66)
    col3.metric("Alta probabilidade (≥66%)", high_count)


def _render_batch_tab(api_url: str) -> None:
    """Aba de predição em lote via upload de CSV."""
    with st.container(border=True):
        st.markdown(f"#### Upload em lote (até {BATCH_LIMIT} linhas)")
        st.caption("Colunas esperadas: " + ", ".join(CSV_REQUIRED_COLUMNS))
        st.download_button(
            "📄 Baixar CSV de exemplo", _example_csv_bytes(), file_name="exemplo.csv"
        )
        uploaded = st.file_uploader(
            "Envie um CSV", type="csv", label_visibility="collapsed"
        )

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

    with st.spinner("Avaliando..."):
        status_code, body = call_predict_batch(api_url, dataframe_to_predict_items(df))
    if status_code != 200:
        st.error(body.get("detail", "Erro ao consultar a API."))
        return

    _render_batch_results(df, body["results"])


def _render_batch_results(df: pd.DataFrame, results: list[dict[str, Any]]) -> None:
    """Mostra o resumo, a tabela com barra de progresso e o botão de download."""
    probabilities = [r["reorder_probability"] for r in results]
    _render_batch_summary([p for p in probabilities if p is not None])

    results_df = df.copy()
    results_df["reorder_probability"] = probabilities
    results_df = results_df.sort_values(
        "reorder_probability", ascending=False, na_position="last"
    )
    st.dataframe(
        results_df,
        use_container_width=True,
        column_config={
            "reorder_probability": st.column_config.ProgressColumn(
                "Probabilidade de recompra",
                min_value=0.0, max_value=1.0, format="%.1%%",
            )
        },
    )
    st.download_button(
        "⬇️ Baixar resultado em CSV",
        results_df.to_csv(index=False),
        file_name="predicoes.csv",
    )


def main() -> None:
    """Ponto de entrada do dashboard."""
    st.set_page_config(
        page_title="Instacart Recommender",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    api_url = resolve_api_url()
    _render_sidebar(api_url)
    _render_header()

    tab_single, tab_batch = st.tabs(["🔍 Predição única", "📦 Lote (CSV)"])
    with tab_single:
        _render_single_prediction_tab(api_url)
    with tab_batch:
        _render_batch_tab(api_url)


if __name__ == "__main__":
    main()
