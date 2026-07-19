"""Estágio 2 do pipeline DVC: engenharia de features + split treino/validação.

Lê `data/processed/orders_merged.parquet`, codifica user_id/product_id
como inteiros contíguos (necessário para as camadas de embedding),
roda o `FeaturePipeline` (Strategy pattern) para gerar as features
tabulares, e grava os splits de treino/validação prontos para o
`InstacartReorderDataset`.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from recommender.config.settings import get_settings
from recommender.preprocessing.pipeline import FeaturePipeline
from recommender.preprocessing.strategies import (
    BasketSizeStrategy,
    RecencyFrequencyStrategy,
    TemporalPatternStrategy,
)

FEATURE_COLUMNS = [
    "purchase_count",
    "days_since_last_order",
    "order_hour_of_day",
    "order_dow",
    "basket_size",
]
LABEL_COLUMN = "reordered"


def _default_pipeline() -> FeaturePipeline:
    """Monta o FeaturePipeline com as três estratégias da Etapa 1."""
    return FeaturePipeline(
        [RecencyFrequencyStrategy(), TemporalPatternStrategy(), BasketSizeStrategy()]
    )


def _encode_ids(merged: pd.DataFrame, models_dir: Path) -> pd.DataFrame:
    """Codifica user_id/product_id como inteiros contíguos e salva os encoders.

    Usa `pd.factorize` em vez de `sklearn.LabelEncoder.fit_transform`:
    ambos produzem o mesmo resultado, mas `factorize` usa uma única
    passada por hash table, com bem menos overhead de memória — decisivo
    para caber o dataset completo do Instacart (~32M linhas) na RAM.
    """
    user_codes, user_uniques = pd.factorize(merged["user_id"], sort=True)
    product_codes, product_uniques = pd.factorize(merged["product_id"], sort=True)
    merged["user_id_enc"] = user_codes.astype("int32")
    merged["product_id_enc"] = product_codes.astype("int32")

    user_encoder, product_encoder = LabelEncoder(), LabelEncoder()
    user_encoder.classes_ = user_uniques.to_numpy()
    product_encoder.classes_ = product_uniques.to_numpy()

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(user_encoder, models_dir / "user_encoder.joblib")
    joblib.dump(product_encoder, models_dir / "product_encoder.joblib")
    vocab_sizes = {
        "num_users": int(len(user_uniques)),
        "num_products": int(len(product_uniques)),
    }
    (models_dir / "vocab_sizes.json").write_text(json.dumps(vocab_sizes, indent=2))
    return merged


def run(processed_dir: Path | None = None, models_dir: Path | None = None) -> None:
    """Executa o estágio de feature engineering e grava treino/validação.

    Args:
        processed_dir: Override do diretório de dados processados.
        models_dir: Override do diretório de artefatos de modelo
            (onde os encoders são salvos).
    """
    settings = get_settings()
    processed_dir = processed_dir or Path(settings.data_processed_dir)
    models_dir = models_dir or Path(settings.models_dir)

    merged = pd.read_parquet(processed_dir / "orders_merged.parquet")
    merged = _encode_ids(merged, models_dir)

    features = _default_pipeline().fit_transform(merged)
    for col in FEATURE_COLUMNS:
        features[col] = features[col].astype("float32")

    ids_and_label = merged[["user_id_enc", "product_id_enc", LABEL_COLUMN]].reset_index(
        drop=True
    )
    del merged
    gc.collect()

    dataset = pd.concat([ids_and_label, features], axis=1).rename(
        columns={"user_id_enc": "user_id", "product_id_enc": "product_id"}
    )
    dataset[FEATURE_COLUMNS] = dataset[FEATURE_COLUMNS].fillna(0)
    del ids_and_label, features
    gc.collect()

    train_df, val_df = train_test_split(
        dataset, test_size=0.2, random_state=settings.random_seed
    )
    train_df.to_parquet(processed_dir / "features_train.parquet", index=False)
    val_df.to_parquet(processed_dir / "features_val.parquet", index=False)
    print(f"[feature_eng] treino={len(train_df):,} val={len(val_df):,}")


if __name__ == "__main__":
    run()
