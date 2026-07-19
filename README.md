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
  config/         # dataclasses de configuração (modelo e treino)
  training/       # loop de treino, métricas, early stopping (Etapa 4)
tests/            # testes unitários (pytest)
configs/          # model.yaml, training.yaml
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
- [ ] **Etapa 2 — Ambiente e Dependências**: Poetry, lock file, `.env` +
      Pydantic Settings, script de validação de ambiente.
- [ ] **Etapa 3 — Containerização e Versionamento**: Dockerfile multi-stage,
      docker-compose (treino + MLflow), DVC (pipeline com ≥3 stages).
- [ ] **Etapa 4 — Rede Neural, Registry e Entrega**: treino completo,
      comparação com baselines Scikit-Learn, MLflow Model Registry, Model
      Card, vídeo STAR.

## Rodando os testes e o lint localmente

```bash
pip install ruff pytest torch pandas numpy
ruff check .
python -m pytest -q
```

## Aprendizados aplicados do TC1 (churn)

- Aplicação mais rigorosa de SOLID desde a primeira linha (feedback do TC1
  apontou uso raso dos princípios)
- Nenhum script solto gerando notebook — projeto todo em módulos `src/`
- Vídeo STAR será hospedado em YouTube/Vimeo/Drive e apenas linkado no
  README, não commitado no repositório
