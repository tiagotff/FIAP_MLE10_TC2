"""Estágio 1 do pipeline DVC: junta os CSVs brutos do Instacart.

Lê `orders.csv` e `order_products__prior.csv` de `data/raw/`, junta os
dois pelo `order_id` para trazer `user_id` e metadados do pedido para
cada linha produto-pedido, e grava o resultado em
`data/processed/orders_merged.parquet`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommender.config.settings import get_settings

ORDERS_DTYPES = {
    "order_id": "int32",
    "user_id": "int32",
    "order_dow": "int8",
    "order_hour_of_day": "int8",
    "days_since_prior_order": "float32",
}
ORDER_PRODUCTS_DTYPES = {
    "order_id": "int32",
    "product_id": "int32",
    "add_to_cart_order": "int16",
    "reordered": "int8",
}


def load_raw_orders(raw_dir: Path) -> pd.DataFrame:
    """Carrega e junta orders.csv com order_products__prior.csv.

    Usa dtypes reduzidos (int32/int8 em vez do int64 default do pandas)
    para manter o dataset completo (~32M linhas) dentro de memória.

    Args:
        raw_dir: Diretório contendo os CSVs brutos do Instacart.

    Returns:
        DataFrame com uma linha por (order_id, product_id), contendo
        user_id, reordered e os metadados temporais do pedido.
    """
    orders = pd.read_csv(
        raw_dir / "orders.csv",
        usecols=list(ORDERS_DTYPES),
        dtype=ORDERS_DTYPES,
    )
    order_products = pd.read_csv(
        raw_dir / "order_products__prior.csv",
        usecols=["order_id", "product_id", "reordered"],
        dtype=ORDER_PRODUCTS_DTYPES,
    )
    merged = order_products.merge(
        orders[
            ["order_id", "user_id", "order_dow", "order_hour_of_day",
             "days_since_prior_order"]
        ],
        on="order_id",
        how="left",
    )
    return merged


def run(raw_dir: Path | None = None, processed_dir: Path | None = None) -> Path:
    """Executa o estágio de preprocess e grava o parquet de saída.

    Args:
        raw_dir: Override do diretório de dados brutos (default: Settings).
        processed_dir: Override do diretório de saída (default: Settings).

    Returns:
        Caminho do arquivo parquet gerado.
    """
    settings = get_settings()
    raw_dir = raw_dir or Path(settings.data_raw_dir)
    processed_dir = processed_dir or Path(settings.data_processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    merged = load_raw_orders(raw_dir)
    output_path = processed_dir / "orders_merged.parquet"
    merged.to_parquet(output_path, index=False)
    print(f"[preprocess] {len(merged):,} linhas gravadas em {output_path}")
    return output_path


if __name__ == "__main__":
    run()
