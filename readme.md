# 🧪 Análise de Qualidade de Vinhos com Machine Learning

Classificador de qualidade de vinhos com **pipeline reproduzível**, **DVC orquestração**, **MLflow rastreamento** e **Streamlit UI interativa**.

**Acurácia atual: ~69%** → Objetivo: Melhorar com feature engineering e ensemble methods.

---

## 📊 Arquitetura ML (Padrão Rotatividade)

Pipeline estruturado em 5 etapas **reproduzíveis** com DVC:

```
ingest → preprocess → prepare → train → evaluate
```

| Etapa | Entrada | Saída | Propósito |
|-------|---------|-------|-----------|
| **ingest** | CSV local / Supabase | `data/raw/wine_quality.csv` | Busca dados |
| **preprocess** | CSV bruto | `data/processed/wine_processed.parquet` | Feature engineering (DuckDB SQL) |
| **prepare** | Processado | `data/processed/splits/{train,val,test}.parquet` + `preprocessor.joblib` | Split 60/20/20, fit preprocessor em train only |
| **train** | Splits | 12 modelos (`models/trained/*.joblib`) | Treina SMOTE+Tomek, 6 classificadores cada |
| **evaluate** | Test set + modelos | Plots + reports | Avalia em test, gera ROC/confusion matrix |

---

## ⚙️ Tecnologias

- **ML**: scikit-learn, XGBoost, imbalanced-learn (SMOTE/Tomek)
- **Pipeline**: DVC (orquestração), MLflow (rastreamento)
- **Dados**: DuckDB (SQL), Parquet (storage)
- **UI**: Streamlit (predições + comparação)
- **API**: FastAPI (REST endpoint)

---

## 🗂️ Estrutura do Projeto

```
wine_project/
├── src/
│   ├── ingestion.py        # Busca dados (Supabase/CSV fallback)
│   ├── preprocessing.py    # Feature engineering em DuckDB SQL
│   ├── prepare_data.py     # Split 60/20/20 + preprocessor serializado
│   ├── train.py            # Treina 12 modelos (SMOTE/Tomek × 6 classif.)
│   ├── evaluate.py         # Avalia em test set, plots ROC + confusion matrix
│   └── __init__.py
├── data/
│   ├── raw/
│   │   └── wine_quality.csv
│   └── processed/
│       ├── wine_processed.parquet
│       └── splits/
│           ├── train.parquet
│           ├── val.parquet
│           └── test.parquet
├── models/
│   ├── trained/
│   │   ├── SMOTE_logistic_regression.joblib
│   │   ├── SMOTE_random_forest.joblib
│   │   ├── ... (12 total)
│   │   └── Tomek_mlp.joblib
│   └── preprocessors/
│       └── preprocessor.joblib
├── app/
│   └── best_model.pkl  # Champion model
├── reports/
│   ├── training_report.json
│   ├── evaluation_report.json
│   ├── confusion_matrix_*.png
│   └── roc_curve_*.png
├── notebooks/
│   └── wine_quality.ipynb
├── streamlit_ui.py     # UI interativa
├── main.py             # FastAPI + Supabase logger
├── dvc.yaml            # Pipeline orquestração
├── requirements.txt
└── .env.example
```

---

## 🚀 Como Executar

### 1. Setup

```bash
# Clone ou navegue ao diretório
cd wine_project

# Instale dependências
pip install -r requirements.txt

# Configure .env
cp .env.example .env
# Edite com suas credenciais Supabase + DagsHub + MLflow
```

### 2. Executar Pipeline Completo (DVC)

```bash
# Reproduz todo pipeline em ordem: ingest → preprocess → prepare → train → evaluate
dvc repro
```

**Ou manualmente:**

#### Etapa 1: Ingestion
```bash
python src/ingestion.py
# Output: data/raw/wine_quality.csv
```

#### Etapa 2: Preprocessing (Feature Engineering)
```bash
python src/preprocessing.py
# Output: data/processed/wine_processed.parquet
# Features adicionadas: log transforms, ratios, flags
```

#### Etapa 3: Prepare Data (Split + Preprocessor)
```bash
python src/prepare_data.py
# Output: 
#   - data/processed/splits/{train,val,test}.parquet
#   - models/preprocessors/preprocessor.joblib
```

#### Etapa 4: Treinamento
```bash
python src/train.py
# Output:
#   - models/trained/{SMOTE,Tomek}_{classifier}.joblib (12 modelos)
#   - app/best_model.pkl (champion)
#   - reports/training_report.json
# MLflow: Registra todos os modelos em DagsHub
```

#### Etapa 5: Avaliação
```bash
python src/evaluate.py
# Output:
#   - reports/evaluation_report.json
#   - reports/{confusion_matrix,roc_curve}_*.png
# MLflow: Logs métricas de teste em runs separadas
```

### 3. Streamlit UI (Predições + Comparação)

```bash
streamlit run streamlit_ui.py
# Abre em http://localhost:8501
# Abas:
#   - Predição: Ajuste features e veja predição em tempo real
#   - Comparação: Métricas de validação vs teste
#   - Histórico: Simulações anteriores
```

### 4. FastAPI (REST Endpoint)

```bash
python main.py
# API em http://localhost:8000
# POST /predict → predição de qualidade
# GET /simulations → histórico
```

---

## 📈 Melhorias Implementadas vs Original

| Aspecto | Original | Novo | Benefício |
|---------|----------|------|-----------|
| **Reprodutibilidade** | Split em train.py | prepare_data.py separado + DVC | ✅ Não há data leakage, caching DVC |
| **Avaliação** | Test no train.py | evaluate.py separado | ✅ Separação concerns, test set protegido |
| **Feature Eng.** | Mínima | Log transforms + ratios + flags (DuckDB SQL) | ✅ Potencial 69%→75%+ F1 |
| **Orquestração** | Manual | DVC stages (5 etapas) | ✅ Reproduzibilidade, tracking |
| **Preprocessor** | Não salvo | joblib artefato MLflow | ✅ Reutilizável em produção |
| **Streamlit** | 2 abas | 3 abas + sidebar com métricas | ✅ Melhor UX, visibilidade de performance |
| **Modelos Salvos** | MLflow only | MLflow + joblib local | ✅ Fallback local, rápido |

---

## 🎯 Próximos Passos

1. **Feature Engineering Avançado**
   - Análise de interações (PCA, correlação)
   - Domain knowledge: química enológica (pH, acidez)

2. **Otimização de Hiperparâmetros**
   - GridSearchCV / Optuna para cada modelo
   - Early stopping em XGBoost

3. **Ensemble Methods**
   - Stacking (meta-learner)
   - Voting classifier com weights

4. **Monitoramento em Produção**
   - MLflow Model Registry promotions
   - Data drift detection
   - Retraining pipelines automáticas

---

## 📝 Exemplos de Uso

### Python (Programático)
```python
import joblib

# Carregar melhor modelo
model = joblib.load("models/trained/SMOTE_xgboost.joblib")

# Carregar preprocessor
preprocessor = joblib.load("models/preprocessors/preprocessor.joblib")

# Predição
X = [[7.4, 0.5, 0.3, 2.0, 0.08, 15, 46, 0.996, 3.3, 0.6, 10.0]]
X_processed = preprocessor.transform(X)
prediction = model.predict(X_processed)  # → 1 (Médio)
```

### Streamlit (UI)
- Ajuste sliders das 11 features
- Veja probabilidade por classe em tempo real
- Histórico de simulações

### FastAPI (REST)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.5,
    ...
    "alcohol": 10.0
  }'
```

---

## 🤝 Referência: Projeto Rotatividade

Este projeto adota a arquitetura modular do [Rotatividade](../) como padrão:
- **Pipeline DVC** reproduzível
- **Preparação de dados** separada (sem data leakage)
- **Avaliação isolada** em test set
- **MLflow tracking** com aliases
- **Modelos salvos** em múltiplos formatos

**Melhorias do wine_project sobre Rotatividade:**
- ✅ UI Streamlit mais elaborada (3 abas + sidebar)
- ✅ Balanceamento agressivo (SMOTE + Tomek)
- ✅ Mais classificadores testados (6 vs 3)
- ✅ FastAPI para integração

---

## 📦 Dependências

Ver [requirements.txt](requirements.txt) para versões exatas.

**Principais:**
- pandas, numpy, scikit-learn, xgboost
- imbalanced-learn (SMOTE, TomekLinks)
- duckdb (SQL queries)
- mlflow, dagshub (rastreamento)
- streamlit, fastapi (UI/API)
- python-dotenv, supabase (configuração)

---

## 📜 Licença

[Especifique sua licença]

---

## 👤 Autor

Projeto de demonstração de best practices em ML - Estrutura Rotatividade adaptada para wine quality.

### 2. Criar e Ativar o Ambiente Virtual (venv)

É uma boa prática criar um ambiente virtual para isolar as dependências do projeto.

**No Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**No macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Instalar as Dependências

Com o ambiente virtual ativado, instale todas as bibliotecas listadas no arquivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

### 4. Treinar o Modelo

Antes de iniciar a aplicação, você precisa treinar o modelo de Machine Learning.  
Para isso, execute o notebook `wine_quality.ipynb`.  
A execução completa deste notebook irá gerar o arquivo `mlp_wine.pkl` na raiz do projeto.

Abra o Jupyter Notebook ou Jupyter Lab e execute todas as células do arquivo:

```bash
wine_quality.ipynb
```

---

### 5. Iniciar a Aplicação Streamlit

Após a geração do arquivo `mlp_wine.pkl`, você pode iniciar a aplicação web.  
No seu terminal (com o ambiente virtual ainda ativado), execute o seguinte comando:

```bash
streamlit run app.py
```

O comando irá iniciar um servidor local e abrir a aplicação no seu navegador padrão.  
Agora você pode interagir com a interface para prever a qualidade de novos vinhos! 🍷
