#!/usr/bin/env bash
# Builda e implanta o dashboard Streamlit no Cloud Run, como um serviço
# independente da API, apontando para a URL pública da API já implantada.
#
# Uso: ./scripts/deploy_streamlit_to_cloud_run.sh SEU_PROJETO URL_DA_API
#
# Exemplo:
#   ./scripts/deploy_streamlit_to_cloud_run.sh instacart-recommender-tc2 \
#     https://recommender-api-XXXXX.us-central1.run.app

set -euo pipefail

PROJECT="${1:?Uso: ./scripts/deploy_streamlit_to_cloud_run.sh SEU_PROJETO URL_DA_API}"
API_URL="${2:?Uso: ./scripts/deploy_streamlit_to_cloud_run.sh SEU_PROJETO URL_DA_API}"

gcloud config set project "${PROJECT}"
gcloud builds submit --config cloudbuild.streamlit.yaml .
gcloud run deploy recommender-dashboard \
  --image "gcr.io/${PROJECT}/recommender-dashboard" \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "RECOMMENDER_API_URL=${API_URL}" \
  --memory 512Mi --cpu 1 --port 8080

echo "Dashboard implantado — a URL pública apareceu no output acima."
