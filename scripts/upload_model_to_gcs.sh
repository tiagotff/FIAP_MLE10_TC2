#!/usr/bin/env bash
# Sobe os artefatos do modelo treinado (models/) para o bucket GCS usado
# pela API como model registry (ver src/recommender/api/model_registry.py).
#
# Uso: ./scripts/upload_model_to_gcs.sh SEU_BUCKET
#
# Depois de rodar, reinicie o serviço no Cloud Run para que a API baixe
# a versão nova (sem precisar rebuildar/redeployar a imagem):
#   gcloud run services update recommender-api --region us-central1

set -euo pipefail

BUCKET="${1:?Uso: ./scripts/upload_model_to_gcs.sh SEU_BUCKET}"
MODELS_DIR="models"

REQUIRED_FILES=(model.pt user_encoder.joblib product_encoder.joblib vocab_sizes.json)
for file in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${MODELS_DIR}/${file}" ]]; then
    echo "ERRO: ${MODELS_DIR}/${file} não existe. Rode o pipeline (dvc repro) antes." >&2
    exit 1
  fi
  gcloud storage cp "${MODELS_DIR}/${file}" "gs://${BUCKET}/${file}"
done

# Métricas são opcionais (usadas só para enriquecer o /metadata da API)
if [[ -f "data/metrics.json" ]]; then
  gcloud storage cp "data/metrics.json" "gs://${BUCKET}/model_metrics.json"
fi

echo "Artefatos publicados em gs://${BUCKET}/"
