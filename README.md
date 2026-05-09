# 🍷 Wine Quality Classifier

Projeto de **MLOps** para classificação de qualidade de vinhos com pipeline completo de ingestão, pré-processamento, treinamento e avaliação de modelos. Integração com **Supabase**, **MLflow/DagsHub**, **DVC** e **Streamlit**.

## 📋 Visão Geral

- **Objetivo**: Classificar a qualidade de vinhos em 2 classes (Ruim/Bom) usando dados físico-químicos
- **Modelos**: Logistic Regression, Random Forest, KNN, SVM, XGBoost, MLP
- **Técnicas de balanceamento**: SMOTE (over-sampling) + Tomek Links (under-sampling)
- **Métrica principal**: F1-Score (validação e teste)
- **Mecanismo de seleção**: Champion (melhor modelo promovido no MLflow Registry)

## 🗂️ Estrutura do Projeto

```
wine_project/
├── README.md
├── Dockerfile              # Build da imagem Docker
├── dvc.yaml               # Pipeline reproduzível com DVC
├── requirements.txt       # Dependências Python
├── .env.example           # Template de variáveis de ambiente
│
├── src/                   # Scripts de pipeline
│   ├── ingestion.py       # Coleta dados (Supabase/Kaggle)
│   ├── preprocessing.py   # Normalização e transformação
│   ├── prepare_data.py    # One-hot encoding + split 60/20/20
│   ├── train.py           # Treinamento com SMOTE/Tomek
│   ├── evaluate.py        # Avaliação no conjunto de teste
│   └── prepare_supabase_predictions_table.py  # Setup tabela de predições
│
├── notebooks/
│   └── wine_quality.ipynb # EDA e exploração
│
├── streamlit_ui.py        # Interface Streamlit (predição + métricas)
├── database.py            # Modelos SQLAlchemy
├── models.py              # ORM para simulation_logs
├── supabase_logger.py     # Logger para histórico de predições
│
├── data/
│   ├── raw/               # Dados originais
│   └── processed/         # Dados processados e splits
│
├── models/
│   ├── trained/           # Modelos .joblib (SMOTE + Tomek)
│   └── best_model.pkl     # Modelo champion em produção
│
├── reports/               # Artefatos e relatórios
│   ├── training_report.json
│   ├── evaluation_report.json
│   └── model_comparison.json
│
└── mlruns/                # MLflow tracking local
```

## 🔄 Pipeline de Dados (DVC)

O pipeline é orquestrado por `dvc.yaml` com os seguintes estágios:

1. **prepare_supabase_predictions_table** 
   - Garante tabela `wine_predictions` no Supabase
   - Output: marker file

2. **ingest**
   - Busca dados do Supabase ou Kaggle
   - Cria tabela `wine_quality` se não existir
   - Output: `data/raw/wine_quality.csv`

3. **preprocess**
   - Normaliza schema, remove outliers (IQR 1.5x)
   - Codifica tipo de vinho (red/white)
   - Output: `data/processed/wine_processed.parquet`

4. **prepare**
   - One-hot encoding + split estratificado 60/20/20
   - Output: `train.parquet`, `val.parquet`, `test.parquet`

5. **train** ⭐
   - Treina 12 modelos (6 algoritmos × 2 estratégias de balanceamento)
   - Registra no MLflow/DagsHub
   - Promove melhor modelo como `@champion`
   - Outputs:
     - `models/trained/SMOTE_*.joblib` (6 modelos)
     - `models/trained/Tomek_*.joblib` (6 modelos)
     - `models/best_model.pkl` (champion)
     - `reports/training_report.json` (métricas de validação)

6. **evaluate**
   - Avalia todos os modelos no conjunto de teste
   - Output: `reports/evaluation_report.json`

## 🚀 Começar Rápido

### Pré-requisitos
- Python 3.10+
- Docker (opcional, para containerização)
- DVC (instalado via `requirements.txt`)

### Setup Local

#### 1. Clonar e configurar ambiente
```bash
git clone <repo>
cd wine_project

# Criar virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou: .venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

#### 2. Configurar variáveis de ambiente
```bash
cp .env.example .env
# Editar .env com seus valores
```

**Variáveis essenciais:**
- `SUPABASE_URL`, `SUPABASE_KEY`: Acesso ao banco de dados
- `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `DAGSHUB_REPO_NAME`: MLflow remoto
- `DATABASE_URL`: Postgres local para logs (opcional)

#### 3. Rodar pipeline completo
```bash
dvc repro -f
```

#### 4. Executar Streamlit
```bash
streamlit run streamlit_ui.py
```
- Acesse: `http://localhost:8501`
- Abas: **Predição** (simular) + **Histórico** (logs)

### 🐳 Deploy com Docker

```bash
# Build da imagem
docker build -t wine-quality .

# Rodar localmente
docker run --env-file .env -p 8501:8501 wine-quality

# No Render: conectar repositório, usar PORT=8501
```

## 📊 Métricas e Monitoramento

### Validação (treinamento)
O arquivo `reports/training_report.json` contém:
- Acurácia, Precisão, Recall, F1-Score para cada modelo

### Teste (avaliação)
O arquivo `reports/evaluation_report.json` contém:
- Mesmas métricas aplicadas ao conjunto de teste (20% dos dados)

### MLflow/DagsHub
- Todos os runs registrados em tempo real
- Modelo champion marcado com alias `@champion`
- Disponível em: `https://dagshub.com/{user}/{repo}.mlflow`

## 🔧 Configuração de Ambiente

### Supabase (Banco de Dados)
```env
SUPABASE_URL=https://xyz.supabase.co
SUPABASE_KEY=<anon_key>
SUPABASE_DB_URL=postgresql://postgres:<pass>@xyz.supabase.co:5432/postgres
SUPABASE_TABLE=wine_quality
SUPABASE_PREDICTIONS_TABLE=wine_predictions
```

### MLflow/DagsHub (Tracking de Experimentos)
```env
DAGSHUB_USERNAME=frpbotero
DAGSHUB_REPO_NAME=wine-quality
DAGSHUB_TOKEN=<token>
MLFLOW_EXPERIMENT=wine-quality
MLFLOW_TRACKING_URI=https://dagshub.com/frpbotero/wine-quality.mlflow
```

### Banco Local (Logs de Predições)
```env
DATABASE_URL=postgresql://user:pass@localhost:5432/wine_db
```

### Kaggle (Dataset Automático)
```env
KAGGLE_USERNAME=<username>
KAGGLE_KEY=<key>
```

## 📈 Exemplo de Uso

### Predição via UI
1. Acesse `http://localhost:8501`
2. Selecione tipo de vinho (Tinto/Branco)
3. Ajuste parâmetros físico-químicos com sliders
4. Clique em **Classificar**
5. Veja predição (Bom/Ruim) e confiança (0-100%)

### Histórico
- Todas as predições são registradas no Supabase
- Aba **Histórico** mostra últimas 100 consultas
- Exportar para CSV disponível

## 🛠️ Tecnologias

| Componente | Tecnologia |
|---|---|
| **Ingestão** | Supabase, Kaggle API |
| **Processamento** | Pandas, NumPy, Scikit-learn |
| **ML/Balanceamento** | SMOTE, Tomek Links, Scikit-learn |
| **Tracking** | MLflow, DagsHub |
| **Reprodução** | DVC |
| **UI** | Streamlit |
| **Banco de Dados** | PostgreSQL (Supabase + local) |
| **Container** | Docker |
| **CI/CD** | Render |

## 📝 Fluxo de Desenvolvimento

```
1. EDA (notebook)
   ↓
2. Ajustar preprocessing (src/preprocessing.py)
   ↓
3. Treinar novos modelos (src/train.py)
   ↓
4. Avaliar (src/evaluate.py)
   ↓
5. Testar UI (streamlit_ui.py)
   ↓
6. Commitar + Push
   ↓
7. Render redeploy automático
```

## 🤔 Troubleshooting

### DVC error: arquivo já rastreado pelo Git
```bash
git rm -r --cached <arquivo>
git commit -m "stop tracking <arquivo>"
```

### MLflow: experimento deletado
- Restaure via: `client.restore_experiment(exp_id)`
- Ou crie novo: `mlflow.set_experiment("novo_nome")`

### Supabase connection timeout
- Verifique IP whitelist
- Confirm `SUPABASE_URL` e `SUPABASE_KEY`

### Render deployment fails
- Cheque `PORT=8501` em variáveis de ambiente
- Verifique logs: `Build Logs` na dashboard Render

## 📄 Licença

MIT

## 👨‍💻 Autor

**Felipe Botero** - `frpbotero`


