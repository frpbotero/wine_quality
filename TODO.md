# TODO — Pipeline End-to-End Wine Classifier

---

## 🔧 Fase 1 — Setup & Ambiente

- [ ] Criar `.env` na raiz do `wine_project/` com:
  - `SUPABASE_URL`, `SUPABASE_KEY`
  - `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `DAGSHUB_REPO_NAME`
  - `MLFLOW_EXPERIMENT`, `MLFLOW_MODEL_NAME`
  - `DATABASE_URL` (opcional, apenas para histórico local)
- [ ] Adicionar `.env` ao `.gitignore`
- [ ] Criar repositório no GitHub e conectar ao DagsHub
- [ ] Rodar `dvc init` na raiz do projeto
- [ ] Configurar DVC remote apontando para DagsHub storage

---

## 🗃️ Fase 2 — Dados (Supabase + DuckDB)

- [ ] Criar projeto no Supabase e tabela `wine_quality`
- [ ] Criar `src/ingestion.py`:
  - Conecta ao Supabase via `supabase-py`
  - Faz upsert inicial do `data/winequality-red.csv`
  - Lê de volta e salva em `data/raw/winequality-red.csv`
- [ ] Criar `src/preprocessing.py` com DuckDB:
  - Lê o raw via `duckdb.read_csv_auto()`
  - SQL: remove duplicatas, cria `quality_1` (`quality >= 7 → 1`), aplica `log1p` em `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `sulphates`
  - Seleciona as 7 features do `minimal` + `quality_1`
  - Salva `data/processed/wine_processed.parquet`
- [ ] Rodar `dvc add data/raw/ data/processed/`
- [ ] Criar `dvc.yaml` com stages:
  ```
  ingestion → preprocessing → training
  ```

---

## 🧪 Fase 3 — Notebook v3 (Refatoração)

Abrir `notebooks/wine_quality_improved_v3.ipynb` e aplicar:

- [ ] Trocar target por `quality_1` (binário: `quality >= 7 → 1`)
- [ ] Substituir feature set pelos 7 do `minimal`:
  - `alcohol`, `volatile acidity`, `citric acid`, `density`
  - `sulphates_log`, `chlorides_log`, `residual sugar_log`
- [ ] Remover coluna `wine_type`, `OneHotEncoder` e `LabelEncoder`
- [ ] Trocar SMOTENC por `SMOTE` simples (apenas numérico)
- [ ] Reduzir `EXPERIMENTS` para 3 modelos:
  - **LogisticRegression**
  - **RandomForest**
  - **XGBoost**
- [ ] Ajustar métricas para binário: `accuracy`, `f1`, `roc_auc`
- [ ] Manter toda a estrutura `run_experiment()` + MLflow tracking no DagsHub
- [ ] Adicionar na célula de promoção:
  ```python
  joblib.dump(best_pipeline, '../app/best_model.pkl')
  ```

---

## 🏋️ Fase 4 — Script de Treinamento

- [ ] Criar `src/train.py`:
  - Lê `data/processed/wine_processed.parquet`
  - Executa o loop dos 3 modelos com MLflow no DagsHub
  - Promove melhor modelo para `Production` no Model Registry
  - Exporta `best_model.pkl` em `app/`

---

## 🌐 Fase 5 — App Streamlit Unificado

Reescrever `streamlit_app/app.py`:

- [ ] Remover toda lógica de chamada à API FastAPI (`requests`)
- [ ] Embutir função `predict(features)` que:
  - Tenta carregar modelo do MLflow Registry (DagsHub)
  - Fallback: carrega `best_model.pkl` via `joblib`
  - Aplica `log1p` nas 5 colunas skewed inline
  - Retorna predição binária + probabilidade
- [ ] UI com sliders para os 7 inputs:
  - `alcohol`, `volatile acidity`, `citric acid`, `density`, `sulphates`, `chlorides`, `residual sugar`
- [ ] Exibir resultado: **🍷 Boa Qualidade** / **❌ Não-Boa** + barra de probabilidade
- [ ] Aba extra: gráficos EDA (distribuições, correlação) com `wine_processed.parquet`
- [ ] Histórico de predições com `st.session_state`

---

## 🐳 Fase 6 — Docker

- [ ] Atualizar `streamlit_app/requirements.txt`:
  - Adicionar: `mlflow`, `scikit-learn`, `xgboost`, `joblib`, `python-dotenv`, `dagshub`
  - Remover: `requests`
- [ ] Atualizar `streamlit_app/Dockerfile`:
  - Garantir cópia do `best_model.pkl` como fallback
  - Expor porta `8501`
- [ ] Simplificar `docker-compose.yml`:
  - Remover serviço `api`
  - Manter apenas `streamlit` (+ `postgres` opcional para dev local)
- [ ] Testar localmente:
  ```bash
  docker build -t wine-app ./streamlit_app
  docker run -p 8501:8501 --env-file .env wine-app
  ```

---

## 🚀 Fase 7 — Deploy (Render)

- [ ] Conectar repositório GitHub ao Render
- [ ] Criar Web Service com deploy via Docker (apontar para `streamlit_app/Dockerfile`)
- [ ] Configurar variáveis de ambiente no Render:
  - `DAGSHUB_USERNAME`
  - `DAGSHUB_TOKEN`
  - `DAGSHUB_REPO_NAME`
  - `MLFLOW_TRACKING_URI`
  - `MLFLOW_MODEL_NAME`
- [ ] Confirmar URL pública funcionando

---

## 📋 Fase 8 — Entregáveis Finais

- [ ] Atualizar `README.md` com:
  - Problema e variável alvo (`quality_1`, binário: vinho bom vs não-bom)
  - Arquitetura do pipeline: `Supabase → DuckDB → DVC → MLflow/DagsHub → Docker → Render`
  - Como reproduzir: `dvc repro`
  - Comparação dos 3 modelos com métricas (F1, ROC-AUC)
  - Link da aplicação no Render
- [ ] Marcar todos os itens do `Todo` original ✅
- [ ] Push final no GitHub + sincronização DagsHub
- [ ] Confirmar no DagsHub: 3 experimentos MLflow registrados + melhor modelo em `Production`

---

## 📌 Ordem de Execução Recomendada

```
Fase 1 → Fase 2 → Fase 3 → Fase 4 → Fase 5 → Fase 6 → Fase 7 → Fase 8
```

## 📐 Decisões Técnicas Fixadas

| Decisão | Escolha |
|---|---|
| Target | Binário — `quality_1` (`quality >= 7 → 1`) |
| Feature set | `minimal` — 7 features |
| Modelos comparados | LogisticRegression, RandomForest, XGBoost |
| Balanceamento | `class_weight='balanced'` + SMOTE simples |
| App | Streamlit unificado (sem FastAPI separada) |
| Deploy | Render via Docker (única imagem) |
| Modelo em produção | MLflow Registry no DagsHub (fallback: `best_model.pkl`) |
