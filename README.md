# Wine Quality Classifier

Projeto de MLOps para classificação de qualidade de vinhos com:
- Pipeline de dados e treinamento (`src/*.py` + `dvc.yaml`)
- UI Streamlit (`streamlit_ui.py`)
- Integração com Supabase e MLflow/DagsHub

## Estrutura

### EDA (notebook)
- `notebooks/wine_quality.ipynb`
- Principais etapas do EDA no notebook:
  - Carga do dataset
  - Distribuição da variável alvo (`quality`)
  - Histogramas das features
  - Análise de outliers (IQR 1.5x)
  - Fusão de classes para problema binário:
    - `0` = qualidade `< 7`
    - `1` = qualidade `>= 7`
  - Split treino/validação/teste (`60/20/20`)

### Scripts de pipeline (`src/`)
- `src/ingestion.py`: coleta dados do Supabase; se necessário tenta criar tabela e fazer seed (Kaggle/local fallback).
- `src/preprocessing.py`: normaliza schema e gera `data/processed/wine_processed.parquet`.
- `src/prepare_data.py`: one-hot em `type`, split estratificado 60/20/20, salva `train/val/test.parquet`.
- `src/train.py`: treina modelos (SMOTE e Tomek), registra métricas, salva modelos e `models/best_model.pkl`.
- `src/evaluate.py`: avalia modelos no conjunto de teste e gera `reports/evaluation_report.json` + gráficos.
- `src/prepare_supabase_predictions_table.py`: garante tabela de predições no Supabase.

### App/UI e suporte
- `streamlit_ui.py`: interface para simulação e visualização de métricas.
- `database.py`: conexão SQLAlchemy com `DATABASE_URL`.
- `models.py`: modelo SQLAlchemy para logs locais (`simulation_logs`).
- `supabase_logger.py`: grava/consulta histórico de predições no Supabase.
- `dvc.yaml`: definição oficial dos estágios reproduzíveis com DVC.

## Como subir o projeto (passo a passo)

### 1) Pré-requisitos
- Python 3.10+
- `pip`
- (Opcional) Docker
- (Opcional) DVC
- (Opcional) Kaggle CLI (`pip install kaggle`) se quiser seed automático via Kaggle

### 2) Configurar ambiente
```bash
cp .env.example .env
```
Preencha os valores reais no `.env`.

### 3) Instalar dependências
```bash
pip install -r requirements.txt
```

### 4) Rodar pipeline completo
```bash
dvc repro
```

### 5) Subir API + UI
```bash
streamlit run streamlit_ui.py
```
- Streamlit: `http://localhost:8501`

## Deploy

### Render (Docker)
O projeto já inclui:
- `Dockerfile`
- `render.yaml`

No Render, configure as variáveis de ambiente (mesmas do `.env`) e use health check:
- `/health`

## Variáveis de ambiente

As variáveis abaixo são utilizadas no código:

### Supabase (Banco de dados remoto + API)
- `SUPABASE_URL`: URL da instância Supabase (ex: `https://xyz.supabase.co`)
- `SUPABASE_KEY`: API key do Supabase para autenticação
- `SUPABASE_TABLE`: Nome da tabela com dados brutos (default: `wine_quality`)
- `SUPABASE_PREDICTIONS_TABLE`: Tabela para armazenar histórico de predições (default: `wine_predictions`)
- `SUPABASE_DATABASE_URL` (ou `SUPABASE_DB_URL`): Connection string PostgreSQL para operações DDL e seed SQL direto

### Banco local/UI (Postgres local para logs)
- `DATABASE_URL`: Connection string PostgreSQL local para armazenar `simulation_logs` (ex: `postgresql://user:pass@localhost:5432/wine_db`)
- `MODEL_PATH`: Caminho do modelo treinado para uso em produção (default: `models/best_model.pkl`)

### MLflow / DagsHub (Rastreamento de experimentos)
- `DAGSHUB_USERNAME`: Seu username no DagsHub
- `DAGSHUB_REPO_NAME`: Nome do repositório no DagsHub (ex: `wine-quality`)
- `DAGSHUB_TOKEN`: Token de autenticação do DagsHub
- `MLFLOW_EXPERIMENT`: Nome do experimento MLflow (default: `wine-quality`) — usado para organizar runs
- `MLFLOW_MODEL_NAME`: Nome do modelo no MLflow Registry (default: `wine-quality-binary`)
- `MLFLOW_TRACKING_URI`: URI do servidor MLflow (opcional; se vazio, usa fallback automático via DagsHub)

### Ingestão de dados
- `ALLOW_LOCAL_FALLBACK`: Se `true`, permite usar arquivo local quando Supabase indisponível (default: `false`)
- `KAGGLE_USERNAME`: Username do Kaggle para download automático do dataset (opcional)
- `KAGGLE_KEY`: API key do Kaggle para autenticação (opcional)

### Deploy & Ambiente
- `PORT`: Porta para o Streamlit (default: `8501`)
- `ENVIRONMENT`: Ambiente de execução (`development`, `staging`, `production`)

**Dica**: Copie `.env.example` e preencha com seus valores reais.

## Artefatos gerados
- `data/raw/wine_quality.csv`
- `data/processed/wine_processed.parquet`
- `data/processed/splits/{train,val,test}.parquet`
- `models/trained/*.joblib`
- `models/best_model.pkl`
- `reports/training_report.json`
- `reports/evaluation_report.json`

