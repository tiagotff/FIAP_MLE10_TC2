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
- [Testes automatizados](#testes-automatizados)
- [Dataset](#dataset)
- [Arquitetura do modelo](#arquitetura-do-modelo)
- [Design patterns aplicados](#design-patterns-aplicados)
- [Resultados](#resultados)
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
| **4** | Rede Neural, Registry e Entrega (baselines, Model Registry, Model Card, vídeo STAR) | ⏳ Em andamento |

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
│   └── validate_env.py           # Validação do ambiente local
├── src/
│   └── recommender/
│       ├── __init__.py
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
│       │   ├── train.py          # Estágio 3 do DVC (com MLflow tracking)
│       │   ├── evaluate.py       # Estágio 4 do DVC
│       │   └── common.py         # Funções compartilhadas entre train/evaluate
│       └── training/             # Reservado para consolidação do loop de treino (Etapa 4)
├── tests/
│   ├── test_model_factory.py     # Testes do ModelFactory
│   ├── test_preprocessing_strategies.py  # Testes das estratégias + integração
│   └── test_settings.py          # Testes das Settings (Pydantic)
├── Dockerfile                    # Build multi-stage (builder + runtime)
├── docker-compose.yml            # Serviço MLflow + serviço de treino
├── dvc.yaml                      # Pipeline DVC (preprocess → feature_eng → train → evaluate)
├── .dvc/config                   # Configuração do remote do DVC
├── .pre-commit-config.yaml       # Hooks de lint automático (ruff)
├── .github/workflows/ci.yml      # CI: lint + testes a cada push
├── .gitignore
├── .dockerignore
├── .env.example
├── LICENSE                       # Licença MIT
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
```

| Estágio | O que faz | Saídas |
|---|---|---|
| `preprocess` | Junta `orders.csv` + `order_products__prior.csv` pelo `order_id` | `data/processed/orders_merged.parquet` |
| `feature_eng` | Codifica `user_id`/`product_id` como inteiros contíguos (`pd.factorize`), roda o `FeaturePipeline` (Strategy pattern), separa treino/validação (80/20) | `features_train.parquet`, `features_val.parquet`, encoders, `vocab_sizes.json` |
| `train` | Treina o `HybridMlpRecommender` com early stopping, loga params/métricas/artefato no MLflow a cada run | `models/model.pt` |
| `evaluate` | Calcula AUC-ROC, recall, precision e F1 no conjunto de validação, loga no MLflow | `data/metrics.json` |

Os hiperparâmetros de treino ficam em `configs/training.yaml` —
`batch_size` alto reduz drasticamente o número de iterações por época em
CPU, e `early_stopping_patience` exige uma melhora mínima de AUC (não
qualquer variação de ruído) para não parar o treino cedo demais nem
tarde demais.

**Versionamento de dados:** o remote do DVC está configurado no
**Google Cloud Storage** (`gs://instacart-recommender-tc2-dvc`, projeto
`instacart-recommender-tc2`) — mesmo padrão usado no Tech Challenge 1
(projeto de churn):

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

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,9045 |
| Recall | 0,9876 |
| Precision | 0,7879 |
| F1-score | 0,8765 |

O recall alto reflete que o modelo captura quase todos os casos reais de
recompra; a precision mais moderada é esperada dado o desbalanceamento
natural da tarefa (a maioria dos pares usuário-produto não é recomprada
em um pedido específico). Comparação formal com baselines do
Scikit-Learn (≥4 métricas) fica para a Etapa 4.

Essas métricas são gravadas em `data/metrics.json` a cada execução do
pipeline e podem ser inspecionadas com `dvc metrics show`.

---

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
