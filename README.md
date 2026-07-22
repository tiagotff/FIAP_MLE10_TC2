# 🛒 Tech Challenge 2 — Sistema de Recomendação de Produtos com Pipeline de ML Completo

[![Status](https://img.shields.io/badge/status-em%20andamento-yellow)]()
[![Etapa](https://img.shields.io/badge/etapa%20atual-3%20de%204-blue)]()

Projeto da Fase 2 da Pós Tech (FIAP) — **MLE10 / Tech Challenge 2**.

Construção, do zero, de um sistema de recomendação de produtos de
e-commerce, usando uma rede neural híbrida (embeddings + MLP) em PyTorch
sobre o dataset **Instacart Market Basket Analysis**, com pipeline de
dados reprodutível via DVC, rastreamento de experimentos via MLflow e
empacotamento em containers Docker.

---

## Quickstart

Sequência única, do zero, para quem acabou de clonar o repositório e
ainda não tem nenhum terminal aberto. Cada comando assume que você está
na pasta raiz do projeto (`cd FIAP_MLE10_TC2`) e usa **Git Bash**
(Linux/macOS: mesmos comandos; Windows PowerShell/cmd: ver variações em
[Setup do ambiente](#setup-do-ambiente)).

> Você vai precisar de **2 terminais abertos ao mesmo tempo** — um para o
> servidor MLflow, outro para rodar o pipeline. Abra o segundo terminal
> só quando o passo indicar.

**Terminal 1** — do clone até o MLflow rodando:

```bash
# 1. Clonar e entrar na pasta
git clone https://github.com/tiagotff/FIAP_MLE10_TC2.git
cd FIAP_MLE10_TC2

# 2. Instalar o projeto e todas as dependências (Python 3.11 ou 3.12 — ver nota abaixo)
poetry install

# 3. Copiar o template de variáveis de ambiente
cp .env.example .env

# 4. Confirmar que o ambiente está pronto (Python, libs, Settings, diretórios)
poetry run python scripts/validate_env.py

# 5. Subir o servidor MLflow — deixe este terminal aberto e rodando
poetry run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5001
```

Espere aparecer a confirmação de que o servidor subiu antes de seguir —
**não digite mais nada neste terminal**, ele precisa continuar rodando.
A UI fica em `http://localhost:5001`.

**Terminal 2** — novo terminal, com o MLflow do Terminal 1 ainda rodando:

```bash
# 1. Entrar na pasta
cd FIAP_MLE10_TC2

# 2. Baixar o dataset do Kaggle e colocar em data/raw/ (a pasta já existe
#    no repositório, com um .gitkeep — só falta colocar os CSVs dentro)
#    https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
#    orders.csv, order_products__prior.csv, order_products__train.csv,
#    products.csv, aisles.csv, departments.csv

# 3. Rodar o pipeline completo (preprocess → feature_eng → train → evaluate)
poetry run dvc repro

# 4. Ver as métricas do modelo treinado
poetry run dvc metrics show
```

> ⚠️ Se `poetry` não for reconhecido, ou se aparecer erro de versão de
> Python, veja [Setup do ambiente](#setup-do-ambiente) — instalar a
> versão certa é o único pré-requisito que pode exigir um passo extra
> antes de começar.
>
> O dataset completo tem ~32,4 milhões de linhas; o estágio `train` pode
> levar alguns minutos dependendo da sua máquina — ver
> [Troubleshooting](#troubleshooting) se parecer travado.

---

## Sumário

- [Quickstart](#quickstart)
- [Contexto do problema](#contexto-do-problema)
- [Status do projeto](#status-do-projeto)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Setup do ambiente](#setup-do-ambiente)
- [Como executar](#como-executar)
- [Troubleshooting](#troubleshooting)
- [Pipeline de dados e treino (DVC)](#pipeline-de-dados-e-treino-dvc)
- [Rodando via Docker](#rodando-via-docker)
- [API de inferência](#api-de-inferência)
- [Testes automatizados](#testes-automatizados)
- [Dataset](#dataset)
- [Arquitetura do modelo](#arquitetura-do-modelo)
- [Design patterns aplicados](#design-patterns-aplicados)
- [Resultados](#resultados)
- [MLflow Model Registry](#mlflow-model-registry)
- [Deploy em nuvem (bônus)](#deploy-em-nuvem-bônus)
- [Model Card](MODEL_CARD.md)
- [Licença](#licença)
- [Equipe](#equipe)

---

## Contexto do problema

Uma empresa de e-commerce precisa de um sistema de recomendação de
produtos baseado no comportamento de navegação e compra dos seus
usuários. O projeto cobre o ciclo completo de um sistema de ML em
produção: engenharia de features, treino de uma rede neural híbrida em
PyTorch, rastreamento de experimentos no MLflow, versionamento de dados
com DVC e empacotamento em containers Docker — com boas práticas de
clean code, reprodutibilidade e testes.

O sinal de recomendação usado é a probabilidade de **recompra**
(`reordered`): para cada combinação usuário-produto observada no
histórico, o modelo prevê a chance desse produto ser pedido de novo,
alimentando um ranking de sugestões no momento da compra.

## Status do projeto

| Etapa | Descrição | Status |
|---|---|---|
| **1** | Clean Code e Estrutura (SOLID, design patterns, linting) | ✅ Concluída |
| **2** | Ambiente e Dependências (Poetry, lock file, `.env`, validação) | ✅ Concluída |
| **3** | Containerização e Versionamento (Docker, DVC, MLflow tracking) | ✅ Concluída |
| **4** | Rede Neural, Registry e Entrega (baselines, Model Registry, Model Card, vídeo STAR) | ⏳ Em andamento — baseline ✅, Model Registry ✅, Model Card ✅, API + deploy ✅, README ✅, vídeo STAR ⏳ |

## Estrutura do repositório

```
.
├── configs/
│   ├── model.yaml               # Hiperparâmetros de arquitetura do modelo
│   └── training.yaml            # Hiperparâmetros de treino (batch, épocas, early stopping)
├── data/
│   ├── raw/                     # CSVs brutos do Instacart (não versionado em git — via DVC)
│   └── processed/                # Parquets gerados pelo pipeline (não versionado em git — via DVC)
├── models/                       # Artefatos de modelo treinado (não versionado em git — via DVC)
├── scripts/
│   ├── validate_env.py           # Validação do ambiente local
│   └── upload_model_to_gcs.sh    # Publica os artefatos do modelo no bucket GCS (deploy)
├── src/
│   └── recommender/
│       ├── __init__.py
│       ├── api/
│       │   ├── main.py             # App FastAPI (/, /health, /ready, /predict, /predict/batch, /metadata, /metrics)
│       │   ├── schemas.py          # Schemas Pydantic (request/response)
│       │   ├── inference.py        # Carrega modelo/encoders e executa predições
│       │   ├── model_registry.py   # Baixa artefatos do modelo via Cloud Storage (deploy)
│       │   ├── metrics.py          # Métricas operacionais (Prometheus) para /metrics
│       │   └── logging_config.py   # Logging estruturado (JSON), sem print()
│       ├── config/
│       │   ├── model_config.py   # Dataclasses de config (modelo, treino)
│       │   └── settings.py       # Settings via Pydantic (.env)
│       ├── data/
│       │   └── dataset.py        # InstacartReorderDataset (torch.utils.data.Dataset)
│       ├── models/
│       │   ├── base.py           # RecommenderModel (ABC)
│       │   ├── hybrid_mlp.py     # HybridMlpRecommender (embeddings + MLP)
│       │   └── factory.py        # ModelFactory (Factory pattern)
│       ├── preprocessing/
│       │   ├── base.py           # PreprocessingStrategy (ABC)
│       │   ├── strategies.py     # Estratégias concretas de features
│       │   └── pipeline.py       # FeaturePipeline (orquestra as estratégias)
│       ├── pipeline/
│       │   ├── preprocess.py     # Estágio 1 do DVC
│       │   ├── feature_eng.py    # Estágio 2 do DVC
│       │   ├── train.py          # Estágio 3 do DVC (com MLflow tracking + Registry)
│       │   ├── evaluate.py       # Estágio 4 do DVC
│       │   ├── baseline.py       # Estágio DVC: baseline Scikit-Learn para comparação
│       │   ├── registry.py       # Registro/promoção no MLflow Model Registry
│       │   └── common.py         # Funções compartilhadas entre train/evaluate
│       └── training/             # Reservado para consolidação do loop de treino
├── tests/
│   ├── test_model_factory.py     # Testes do ModelFactory
│   ├── test_preprocessing_strategies.py  # Testes das estratégias + integração
│   ├── test_settings.py          # Testes das Settings (Pydantic)
│   ├── test_hybrid_mlp.py        # Forward pass do HybridMlpRecommender
│   ├── test_dataset.py           # InstacartReorderDataset
│   ├── test_baseline.py          # Baseline Scikit-Learn
│   ├── test_registry.py          # Lógica de promoção do Model Registry (MLflow)
│   ├── test_api.py               # Endpoints da API (FastAPI TestClient)
│   └── test_model_registry.py    # Download de artefatos via GCS (mockado)
├── Dockerfile                    # Build multi-stage (builder + runtime) — treino
├── Dockerfile.api                # Build multi-stage — API de inferência
├── docker-compose.yml            # Serviço MLflow + serviço de treino
├── dvc.yaml                      # Pipeline DVC (preprocess → feature_eng → train/baseline → evaluate)
├── .dvc/config                   # Configuração do remote do DVC
├── .pre-commit-config.yaml       # Hooks de lint automático (ruff)
├── .github/workflows/ci.yml      # CI: lint + testes a cada push
├── .gitignore
├── .dockerignore
├── .env.example
├── LICENSE                       # Licença MIT
├── MODEL_CARD.md                 # Performance, limitações e vieses do modelo
├── pyproject.toml                # Poetry: deps de prod/dev, config do ruff e pytest
└── README.md
```

## Setup do ambiente

Pré-requisitos: **Python 3.11 ou 3.12**, [Poetry](https://python-poetry.org/),
**Git**, conta no [Kaggle](https://www.kaggle.com/) (para baixar o
dataset), [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
autenticado (`gcloud auth application-default login`) para sincronizar
dados com o remote do DVC no GCS. Docker + Docker Compose são opcionais,
apenas para rodar containerizado.

### 1. Clonar o repositório

```bash
git clone https://github.com/tiagotff/FIAP_MLE10_TC2.git
cd FIAP_MLE10_TC2
```

### 2. Instalar as dependências

```bash
poetry install
```

Isso cria um ambiente virtual isolado (`.venv/` dentro do projeto) e
instala tudo declarado no `pyproject.toml`: PyTorch, Scikit-Learn,
MLflow, DVC, Pandas, Pydantic Settings, e as ferramentas de
desenvolvimento (`pytest`, `ruff`, `pre-commit`).

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

As configurações (URI do MLflow, diretórios de dados, seed, device) são
carregadas via `Settings` (Pydantic Settings) a partir desse arquivo —
nenhum módulo do projeto lê `os.environ` diretamente.

### 4. Validar o ambiente

```bash
poetry run python scripts/validate_env.py
```

Confere versão do Python, se todas as bibliotecas obrigatórias estão
instaladas, se as `Settings` carregam corretamente, e se os diretórios
de dados/modelos existem.

## Como executar

### 1. Baixar o dataset

Baixe o [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)
no Kaggle e coloque os CSVs em `data/raw/` — a pasta já existe no
repositório (com um `.gitkeep`, já que os dados em si não são
versionados em git), só falta colocar os arquivos dentro dela (ver
[Dataset](#dataset) para a lista completa de arquivos esperados).

### 2. Subir o MLflow

```bash
poetry run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5001
```

Deixe rodando em um terminal separado. UI em `http://localhost:5001`.

### 3. Rodar o pipeline

```bash
poetry run dvc repro          # roda os 4 estágios (só re-executa o que mudou)
poetry run dvc dag            # visualiza o grafo de dependências
poetry run dvc metrics show   # mostra as métricas gravadas em data/metrics.json
```

### 4. Lint

```bash
poetry run ruff check .
```

## Troubleshooting

**`ConnectionRefusedError` / `Failed to establish a new connection` ao
rodar `dvc repro` no estágio `train`.**
O servidor MLflow não está rodando. Veja [Como executar](#como-executar)
— ele precisa estar ativo, em um terminal separado, **antes** de rodar
o `train`.

**`WARNING: ... RWLock-file ... had been killed. Auto removed it from
the lock file` ao rodar `dvc repro`.**
Não é um erro — o DVC detectou que uma execução anterior foi
interrompida (ex.: fechamento abrupto do terminal) e limpou
automaticamente o lock órfão. É seguro rodar `dvc repro` de novo.

**O treino demora muito / parece travado.**
O dataset completo tem ~32,4 milhões de linhas brutas (~26M exemplos de
treino). Em CPU, o gargalo é o número de iterações por época — ajuste
`batch_size` em `configs/training.yaml` para um valor maior (o projeto
usa `131072` por padrão) para reduzir drasticamente esse número. O
`early_stopping_patience` também interrompe o treino automaticamente
assim que o AUC de validação para de melhorar de forma significativa.

**`The Poetry configuration is invalid: ... required in package mode`
ao rodar `poetry install`.**
Sua versão do Poetry é anterior à 2.0 e não lê o formato PEP 621
(`[project]`). O `pyproject.toml` deste projeto já usa o formato
clássico (`[tool.poetry]`), compatível com qualquer versão — se ainda
assim ocorrer, rode `poetry --version` e atualize com `pip install -U
poetry`.

**`MemoryError` / processo morto (`Killed`) durante `feature_eng` ou
`train`.**
O pipeline processa o dataset completo do Instacart em memória; máquinas
com pouca RAM disponível (< 4GB livres) podem não suportar o volume
completo. O código já usa dtypes reduzidos (`int32`/`int8`/`float32`) e
`pd.factorize` em vez de `LabelEncoder` para minimizar o uso de memória,
mas datasets desse porte exigem alguns GB livres.

**`Unknown project id` ao rodar `gcloud storage buckets create`.**
O projeto GCP ainda não existe — `gcloud config set project` só troca o
contexto padrão, não cria o projeto. Rode primeiro
`gcloud projects create instacart-recommender-tc2 --name="Instacart
Recommender TC2"`, depois `gcloud config set project
instacart-recommender-tc2`.

**`dvc push`/`dvc pull` falham com erro de permissão no GCS.**
A autenticação expirou ou nunca foi feita nesta máquina. Rode
`gcloud auth application-default login` e tente de novo. Se o projeto
usar uma service account (ex.: em CI), configure com `dvc remote modify
gcpremote credentialpath /caminho/para/chave.json` em vez disso.

## Pipeline de dados e treino (DVC)

```
preprocess → feature_eng → train → evaluate
                              └→ baseline
```

| Estágio | O que faz | Saídas |
|---|---|---|
| `preprocess` | Junta `orders.csv` + `order_products__prior.csv` pelo `order_id` | `data/processed/orders_merged.parquet` |
| `feature_eng` | Codifica `user_id`/`product_id` como inteiros contíguos (`pd.factorize`), roda o `FeaturePipeline` (Strategy pattern), separa treino/validação (80/20) | `features_train.parquet`, `features_val.parquet`, encoders, `vocab_sizes.json` |
| `train` | Treina o `HybridMlpRecommender` com early stopping; loga params/métricas/artefato no MLflow; registra e promove a versão no Model Registry | `models/model.pt` |
| `evaluate` | Calcula AUC-ROC, recall, precision e F1 no conjunto de validação, loga no MLflow | `data/metrics.json` |
| `baseline` | Treina uma Regressão Logística (Scikit-Learn) só com as features tabulares, para comparação com o modelo neural | `data/metrics_baseline.json` |

Os hiperparâmetros de treino ficam em `configs/training.yaml` —
`batch_size` alto reduz drasticamente o número de iterações por época em
CPU, e `early_stopping_patience` exige uma melhora mínima de AUC (não
qualquer variação de ruído) para não parar o treino cedo demais nem
tarde demais.

**Versionamento de dados:** o remote do DVC está configurado no
**Google Cloud Storage** (`gs://instacart-recommender-tc2-dvc`, projeto
`instacart-recommender-tc2`).

> O enunciado do desafio pede remote "local ou S3"; o uso de GCP como
> alternativa foi autorizado pelo coordenador do curso.

```bash
poetry run dvc push   # sincroniza .dvc/cache com o bucket no GCS
poetry run dvc pull   # traz os dados versionados de volta (em uma máquina nova)
```

Requer autenticação prévia via `gcloud auth application-default login`
(ou uma service account, configurada com `dvc remote modify gcpremote
credentialpath ...`). Durante o desenvolvimento, o projeto também foi
validado com um remote local (`../dvc-storage`) — útil como alternativa
caso o acesso ao GCP não esteja disponível.

## Rodando via Docker

```bash
docker compose up --build
```

Sobe o servidor MLflow (`http://localhost:5001`, backend SQLite) e o
serviço de treino containerizado, com `data/` e `models/` montados como
volumes. O `Dockerfile` usa build multi-stage: um estágio `builder`
resolve as dependências via Poetry, e o estágio `runtime` final carrega
só o ambiente virtual já resolvido e o código-fonte, sem ferramentas de
build, rodando como usuário não-root.

**Otimização de tamanho da imagem:** a imagem final passou por duas
otimizações que juntas reduziram o tamanho em ~75% (de 8,83 GB para
2,18 GB):

- **PyTorch CPU-only**: o `pyproject.toml` aponta o `torch` para o
  índice `https://download.pytorch.org/whl/cpu` — sem isso, o Poetry
  resolve a build com CUDA completo, que embute bibliotecas NVIDIA
  (`nvidia-cublas`, `nvidia-cudnn`, etc.) somando vários GB, inúteis
  aqui já que o projeto roda em CPU (`device: cpu` nas Settings).
- **DVC fora da imagem**: `dvc` mora num grupo Poetry separado
  (`[tool.poetry.group.cli]`), não no grupo `main`. Nenhum script em
  `src/recommender` importa `dvc` — é só a ferramenta de linha de
  comando que orquestra o pipeline, e o `Dockerfile` instala apenas
  `--only main`, então ela nunca entra na imagem. Continua instalada
  normalmente no ambiente local (`poetry install`, sem `--only`,
  instala todos os grupos), então `poetry run dvc repro` funciona
  igual.

## API de inferência

O modelo em **Production** no MLflow Model Registry é servido via uma
API FastAPI (`src/recommender/api/`), seguindo a convenção adotada por
plataformas de model serving em produção (KServe, Seldon, BentoML,
MLflow Serving): endpoints distintos para liveness, readiness,
inferência, metadados e métricas operacionais.

### 1. Ter um modelo treinado

A API precisa dos artefatos em `models/` (`model.pt`,
`user_encoder.joblib`, `product_encoder.joblib`, `vocab_sizes.json`) —
gerados pelo pipeline (`poetry run dvc repro`), ou baixados de um bucket
GCS (ver [Deploy em nuvem](#deploy-em-nuvem-bônus)).

### 2. Iniciar a API

```bash
poetry run uvicorn recommender.api.main:app --reload
```

A API fica disponível em `http://127.0.0.1:8000`. Documentação
interativa (Swagger UI) em `http://127.0.0.1:8000/docs` — dá para testar
todos os endpoints abaixo direto no navegador.

### Endpoints

| Endpoint | Método | Propósito |
|---|---|---|
| `/` | GET | Informações básicas e links para os demais endpoints |
| `/health` | GET | **Liveness** — o processo está vivo? (não depende do modelo) |
| `/ready` | GET | **Readiness** — o modelo está carregado e pronto para inferência? |
| `/predict` | POST | Predição de recompra para **um** par usuário-produto |
| `/predict/batch` | POST | Predição para **até 500** pares em uma chamada (um único forward pass) |
| `/metadata` | GET | Versão/métricas do modelo + JSON Schema de entrada/saída de `/predict` |
| `/metrics` | GET | Métricas operacionais da API no formato Prometheus |

**`GET /health`** — liveness probe simples, sempre rápida:

```bash
curl http://127.0.0.1:8000/health
```
```json
{"status": "ok"}
```

**`GET /ready`** — readiness probe: confirma que o modelo está
carregado. Diferente de `/health` — a API pode estar viva mas ainda não
pronta (ex.: durante o carregamento do modelo):

```bash
curl http://127.0.0.1:8000/ready
```
```json
{"status": "ready", "model_loaded": true}
```

**`GET /metadata`** — versão/métricas do modelo em produção + o JSON
Schema esperado por `/predict` (útil para descobrir o contrato da API
programaticamente):

```bash
curl http://127.0.0.1:8000/metadata
```
```json
{
  "model_info": {
    "model_version": "1", "num_users": 206209, "num_products": 49677,
    "embedding_dim": 32, "mlp_hidden_dims": [128, 64, 32],
    "metrics": {"auc_roc": 0.9045, "recall": 0.9876, "precision": 0.7879, "f1": 0.8765}
  },
  "input_schema": { "...": "JSON Schema completo dos campos aceitos por /predict" },
  "output_schema": { "...": "JSON Schema da resposta de /predict" }
}
```

**`GET /metrics`** — métricas operacionais (contagem de requisições,
latência por rota, predições por faixa de probabilidade), no formato
texto do Prometheus:

```bash
curl http://127.0.0.1:8000/metrics
```
```
recommender_api_requests_total{method="POST",path="/predict",status_code="200"} 12.0
recommender_api_request_latency_seconds_sum{method="POST",path="/predict"} 0.58
recommender_predictions_total{reorder_likelihood="high"} 7.0
recommender_predictions_total{reorder_likelihood="low"} 5.0
```

**`POST /predict`** — recebe um par usuário-produto e retorna a
probabilidade de recompra:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1, "product_id": 196, "purchase_count": 3,
    "days_since_last_order": 7, "order_hour_of_day": 10,
    "order_dow": 2, "basket_size": 8
  }'
```
```json
{"reorder_probability": 0.73, "model_version": "1"}
```

`user_id`/`product_id` fora do vocabulário de treino (cold-start)
retornam **HTTP 422**, com uma mensagem clara, antes de chegar à lógica
de inferência.

**`POST /predict/batch`** — mesma predição, para até 500 pares em uma
única chamada; itens cold-start retornam `reorder_probability: null`
em vez de derrubar a requisição inteira:

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{ "...": "mesmo formato do /predict, um objeto por item" }]}'
```
```json
{"results": [{"reorder_probability": 0.73, "model_version": "1"}]}
```

### Outros detalhes da API

- Toda requisição é logada em formato estruturado (JSON,
  `src/recommender/api/logging_config.py` — nenhum módulo de produção
  usa `print()`), registrada nas métricas Prometheus, e recebe os
  headers `X-Request-ID` e `X-Process-Time-Ms` via middleware.
- **CORS** habilitado — a API pode ser consumida diretamente de um
  frontend/navegador.
- Qualquer erro inesperado (não relacionado à validação de entrada) é
  capturado por um handler genérico e retorna **HTTP 500** com uma
  mensagem padrão — detalhes internos nunca são expostos na resposta,
  apenas registrados no log estruturado.
- **Model registry via GCS** (`src/recommender/api/model_registry.py`):
  se a variável de ambiente `MODEL_BUCKET` estiver definida, a API baixa
  os artefatos do modelo de um bucket GCS na subida, em vez de exigir
  que estejam embutidos na imagem — promover um modelo novo em produção
  é só atualizar o bucket (`scripts/upload_model_to_gcs.sh`) e reiniciar
  o serviço, sem rebuild/redeploy da imagem Docker.

## Testes automatizados

```bash
poetry run pytest -q
```

Para incluir relatório de cobertura, adicione
`--cov=src/recommender --cov-report=term-missing`.

| Arquivo | O que valida |
|---|---|
| `tests/test_model_factory.py` | `ModelFactory` instancia o modelo correto a partir da config, levanta erro para `model_type` desconhecido, e aceita registro de novos modelos (Open/Closed) |
| `tests/test_preprocessing_strategies.py` | Cada `PreprocessingStrategy` isoladamente, **e** o `FeaturePipeline` com as três estratégias combinadas — teste de integração que garante que o número de linhas se preserva ao concatenar features de estratégias diferentes |
| `tests/test_settings.py` | `Settings` carrega defaults corretos e respeita override por variável de ambiente |
| `tests/test_hybrid_mlp.py` | Forward pass do `HybridMlpRecommender`: shape de saída, valores finitos, respeita dimensão de features configurada |
| `tests/test_dataset.py` | `InstacartReorderDataset`: `__len__`, `__getitem__`, construção via `from_dataframe` e via arrays numpy |
| `tests/test_baseline.py` | Treino e cálculo de métricas do baseline (Regressão Logística) |
| `tests/test_registry.py` | Lógica de promoção do Model Registry (Staging sempre, Production só se melhor que a atual) — via `MlflowClient` simulado |
| `tests/test_api.py` | Endpoints `/`, `/health`, `/ready`, `/metadata`, `/metrics`, `/predict` e `/predict/batch` (válido, cold-start, limite de 500 itens), headers de observabilidade, CORS |
| `tests/test_model_registry.py` | Download dos artefatos do modelo via Cloud Storage (mockado) — inclusive o caso de arquivo opcional ausente |

## Dataset

**Instacart Market Basket Analysis** — dataset público de comportamento
de compra em e-commerce, disponibilizado pela Instacart.

- **Fonte**: [Kaggle — instacart-market-basket-analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)
- **Volume**: ~3,4 milhões de pedidos, ~206 mil usuários, ~50 mil
  produtos, ~32,4 milhões de linhas produto-pedido em
  `order_products__prior.csv`
- **Licença/uso**: dataset público de competição, amplamente utilizado
  para fins educacionais e de pesquisa
- **Onde colocar**: a pasta `data/raw/` já existe no repositório (com um
  `.gitkeep` — os CSVs em si não são versionados em git, só via DVC).
  Baixe o `.zip` do Kaggle, extraia, e coloque estes arquivos dentro
  dela: `orders.csv`, `order_products__prior.csv`,
  `order_products__train.csv`, `products.csv`, `aisles.csv`,
  `departments.csv`

## Arquitetura do modelo

Modelo híbrido que combina **embeddings aprendidos** com **features
tabulares**, satisfazendo o requisito do desafio de forma dupla (o
enunciado pede "MLP ou embedding-based" — este modelo é as duas coisas
ao mesmo tempo):

```
user_id ──► nn.Embedding(dim=32) ─┐
                                    ├─► concat ──► MLP [128 → 64 → 32 → 1] ──► logit (sigmoid = P(reorder))
product_id ─► nn.Embedding(dim=32) ┘
                                    │
features tabulares ─────────────────┘
  (recência, frequência de compra,
   hora/dia do pedido, tamanho do carrinho)
```

- **Embeddings**: `user_id` e `product_id`, dimensão 32 cada
- **MLP**: camadas `[128, 64, 32]`, com `BatchNorm1d`, `ReLU` e
  `Dropout(0.3)` entre elas
- **Loss**: `BCEWithLogitsLoss`
- **Features tabulares** (via Strategy pattern): recência e frequência
  de compra por par usuário-produto, padrão temporal do pedido (dia da
  semana, hora do dia), tamanho do carrinho

## Design patterns aplicados

- **Factory** (`models/factory.py`) — `ModelFactory.create(config)`
  decide qual arquitetura instanciar a partir de `config.model_type`,
  sem acoplar o código de treino a uma classe concreta. Novas
  arquiteturas são adicionadas via `ModelFactory.register(...)`, sem
  alterar código existente (Open/Closed).
- **Strategy** (`preprocessing/`) — cada `PreprocessingStrategy`
  encapsula uma família de features; o `FeaturePipeline` as orquestra
  sem conhecer os detalhes de cada uma (Single Responsibility).
- **Dependency Inversion** — o pipeline de treino depende das
  abstrações `RecommenderModel` e `PreprocessingStrategy`, nunca de
  implementações concretas importadas diretamente.

## Resultados

Métricas no conjunto de validação (20% dos dados, split aleatório),
modelo treinado com o dataset completo (~32,4M linhas brutas → ~26M
exemplos de treino após o split):

| Métrica | Modelo neural (híbrido) | Baseline (Regressão Logística) |
|---|---|---|
| AUC-ROC | **0,9045** | 0,8964 |
| Recall | **0,9876** | 0,7698 |
| Precision | 0,7879 | **0,8777** |
| F1-score | **0,8765** | 0,8202 |

O baseline (`src/recommender/pipeline/baseline.py`, estágio `baseline`
no `dvc.yaml`) usa só as features tabulares, sem os embeddings de
usuário/produto — a diferença isola o ganho específico de aprender
representações latentes. O recall bem mais alto do modelo neural
reflete que ele captura quase todos os casos reais de recompra, ao
custo de uma precision um pouco menor.

Essas métricas são gravadas em `data/metrics.json` (modelo neural) e
`data/metrics_baseline.json` (baseline) a cada execução do pipeline, e
podem ser inspecionadas com `dvc metrics show`. Análise completa de
limitações e vieses está no [Model Card](MODEL_CARD.md).

## MLflow Model Registry

Ao final de cada treino, `src/recommender/pipeline/registry.py`
registra automaticamente uma nova versão do modelo
(`instacart-recommender`) no MLflow Model Registry e a promove:

1. Sempre para **Staging**.
2. Para **Production** apenas se o AUC de validação superar a versão
   atualmente em Production (ou se for a primeira versão) — protege
   contra um treino pior sobrescrever um modelo melhor já em uso.

Consulte as versões registradas na UI do MLflow
(`http://localhost:5001` → aba **Models**).

---

## Deploy em nuvem (bônus)

A API pode ser implantada no **Google Cloud Platform**, via **Cloud
Run**, com o modelo carregado dinamicamente de um bucket do **Cloud
Storage** (model registry — ver [API de inferência](#api-de-inferência)).

### Componentes do deploy

- **Imagem Docker** ([`Dockerfile.api`](Dockerfile.api)): build
  multi-stage, sem o modelo embutido — só código-fonte e dependências
  de runtime (PyTorch CPU-only).
- **Model registry** (`src/recommender/api/model_registry.py`): a API
  baixa `model.pt`, os encoders e `vocab_sizes.json` de um bucket GCS na
  inicialização, configurado via `MODEL_BUCKET` — promover um modelo
  novo é só atualizar o bucket, sem rebuild/redeploy da imagem.
- **Cloud Build**: builda e publica a imagem a partir do código-fonte.
- **Cloud Run**: serviço serverless, escala a zero quando sem tráfego.

### Reproduzindo o deploy

```bash
# 1. Criar o bucket de modelos e subir os artefatos treinados
#    (treine antes com: poetry run dvc repro)
gcloud storage buckets create gs://instacart-recommender-tc2-models --location=us-central1
./scripts/upload_model_to_gcs.sh instacart-recommender-tc2-models

# 2. Build e deploy da API
gcloud builds submit --tag gcr.io/instacart-recommender-tc2/recommender-api -f Dockerfile.api
gcloud run deploy recommender-api \
  --image gcr.io/instacart-recommender-tc2/recommender-api \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_BUCKET=instacart-recommender-tc2-models,MODEL_VERSION=1 \
  --memory 1Gi --cpu 1 --port 8080
```

O comando de deploy imprime a URL pública do serviço ao final —
documentação interativa em `<URL>/docs`.

## Licença

O código-fonte e a documentação autoral deste repositório estão
licenciados sob a **Licença MIT** — ver [`LICENSE`](LICENSE).

O enunciado oficial do desafio é de autoria da **FIAP / POS TECH** e
**não está** coberto pela licença MIT acima — mencionado apenas para
fins de referência e contexto acadêmico.

---

## Equipe

Tech Challenge 2 — FIAP MLE10 - Machine Learning Engineering

By Tiago de Freitas Faustino
