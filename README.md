# Sistema de Recomendação de Produtos — FIAP MLE10 Tech Challenge (Fase 02)

Sistema de recomendação de produtos de e-commerce baseado no comportamento de
navegação dos usuários, usando o dataset **Instacart Market Basket**. O
modelo central é uma rede neural híbrida (embeddings + MLP) em PyTorch, com
pipeline reprodutível via Docker, DVC e MLflow.

## Arquitetura do modelo

Modelo híbrido que combina embeddings aprendidos com features tabulares:

- `nn.Embedding` para `user_id` (dim 32) e `product_id` (dim 32)
- Concatenação com features tabulares (recência, frequência de compra,
  padrão temporal, tamanho médio do carrinho)
- MLP por cima: `[128, 64, 32] → 1` (logit de probabilidade de reorder),
  com BatchNorm, ReLU e Dropout 0.3
- Satisfaz o requisito "MLP ou embedding-based" ao ser as duas coisas: usa
  embeddings como entrada de uma MLP

## Estrutura do projeto

```
src/recommender/
  models/         # RecommenderModel (ABC), HybridMlpRecommender, ModelFactory
  preprocessing/  # PreprocessingStrategy (ABC), estratégias concretas, FeaturePipeline
  data/           # InstacartReorderDataset (torch.utils.data.Dataset)
  config/         # dataclasses de config (modelo/treino) + Settings (Pydantic, .env)
  training/       # loop de treino, métricas, early stopping (Etapa 4)
  pipeline/       # scripts do DVC: preprocess, feature_eng, train, evaluate
tests/            # testes unitários (pytest)
scripts/          # validate_env.py — validação do ambiente local
configs/          # model.yaml, training.yaml
Dockerfile        # build multi-stage (builder + runtime)
docker-compose.yml # serviço MLflow + serviço de treino
dvc.yaml          # pipeline DVC (preprocess → feature_eng → train → evaluate)
data/             # raw/ e processed/ (versionados via DVC, não via git)
models/           # artefatos de modelo treinado (não versionados via git)
```

## Design patterns aplicados

- **Factory** (`models/factory.py`): `ModelFactory.create(config)` decide
  qual arquitetura instanciar a partir de `config.model_type`, sem acoplar
  o código de treino a uma classe concreta.
- **Strategy** (`preprocessing/`): cada `PreprocessingStrategy` encapsula
  uma família de features (recência/frequência, temporal, tamanho de
  carrinho); o `FeaturePipeline` as orquestra sem conhecer os detalhes de
  cada uma.

## Status por etapa

- [x] **Etapa 1 — Clean Code e Estrutura**: estrutura `src/`, `tests/`,
      `data/`, `models/`, `configs/`; SOLID; Factory + Strategy; type hints e
      docstrings Google style em todas as funções públicas; `ruff check`
      sem erros; pre-commit hooks configurados.
- [x] **Etapa 2 — Ambiente e Dependências**: `pyproject.toml` com Poetry
      (dependências de prod/dev separadas), lock file commitado, Settings
      via Pydantic (`.env`), script de validação de ambiente.
- [x] **Etapa 3 — Containerização e Versionamento**: Dockerfile
      multi-stage (builder + runtime), `docker-compose.yml` (MLflow +
      treino), pipeline DVC com 4 estágios (`preprocess → feature_eng →
      train → evaluate`), MLflow tracking (params/métricas/artefatos por
      run), remote local do DVC configurado.
- [ ] **Etapa 4 — Rede Neural, Registry e Entrega**: comparação com
      baselines Scikit-Learn, MLflow Model Registry, Model Card, vídeo
      STAR.

## Pipeline de dados e treino (DVC)

```bash
# Coloque os CSVs do Instacart em data/raw/ antes de rodar:
# orders.csv, order_products__prior.csv, order_products__train.csv,
# products.csv, aisles.csv, departments.csv

dvc repro          # roda os 4 estágios (preprocess → feature_eng → train → evaluate)
dvc dag            # visualiza o grafo de dependências
dvc metrics show   # mostra as métricas gravadas em data/metrics.json
```

## Rodando via Docker

```bash
docker compose up --build
```
Sobe o servidor MLflow (`http://localhost:5001`) e roda o treino
containerizado, com `data/` e `models/` montados como volumes.

## Instalação e configuração do ambiente

```bash
# Instala as dependências (prod + dev) a partir do lock file
poetry install

# Copia o template de variáveis de ambiente
cp .env.example .env

# Valida se o ambiente está pronto
poetry run python scripts/validate_env.py
```

## Rodando os testes e o lint localmente

```bash
poetry run ruff check .
poetry run pytest -q
```
