# ---- Stage 1: builder ----
# Instala as dependências via Poetry num venv isolado, sem levar
# ferramentas de build para a imagem final. Instala os grupos "main"
# (torch/sklearn/pandas/etc.) e "training" (mlflow) — este último fica
# de fora da imagem da API (Dockerfile.api), que nunca importa mlflow.
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PIP_NO_CACHE_DIR=1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main,training --no-root --no-interaction

COPY src ./src
RUN poetry install --only main,training --no-interaction

# ---- Stage 2: runtime ----
# Imagem enxuta: só o venv já resolvido + código-fonte, sem Poetry
# nem cache de build.
FROM python:3.11-slim AS runtime

# git: só o binário (não copiamos .git/ para a imagem — isso infla a
# imagem à toa e não é necessário em produção). Instalá-lo evita o
# warning do MLflow sobre "Git executable not found"; sem um repositório
# .git presente, o MLflow simplesmente não anexa o SHA do commit às
# runs, sem erro.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY configs ./configs
COPY scripts ./scripts

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

USER app

ENTRYPOINT ["python", "-m"]
CMD ["recommender.pipeline.train"]
