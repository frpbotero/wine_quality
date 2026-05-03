# 🍷 Wine Quality Classifier

**ML Pipeline + Streamlit UI + FastAPI Backend** - Classificação de qualidade de vinho com balanceamento de dados (SMOTE/Tomek)

## 📊 Projeto

- **Modelo Vencedor**: SMOTE + Random Forest (Test F1: 0.69)
- **Estratégias**: SMOTE (over-sampling) e Tomek Links (under-sampling)
- **Classificação**: 3 classes (Ruim/Médio/Bom)
- **Features**: 11 numéricas + 1 categórica (red/white)
- **Dataset**: Wine Quality (UCI ML Repository)

## 🚀 Deploy no Render

### Pré-requisitos

1. Conta no [Render.com](https://render.com)
2. Token do GitHub pessoal
3. Variáveis de ambiente (Supabase, DagsHub, etc.)

### Configuração

1. **Criar Secret File no Render**
   - Nome: `.env`
   - Conteúdo (copiar de `.env.example` e preencher valores):
     ```
     SUPABASE_URL=https://your-project.supabase.co
     SUPABASE_KEY=your_anon_key
     DAGSHUB_USERNAME=seu_usuario
     DAGSHUB_REPO_NAME=wine-quality
     DAGSHUB_TOKEN=seu_token
     API_SECRET_KEY=sua_chave_secreta
     ```

2. **Conectar GitHub**
   - Autorizar Render a acessar seu repositório
   - Branch: `master`

3. **Deploy automático**
   - Selecionar `Docker` como runtime
   - Render detecta `Dockerfile` automaticamente
   - Build e deploy com cada push

### Volumes & Persistence (Opcional)

Para manter modelos treinados entre deploys:
```yaml
mounts:
  - name: models
    mountPath: /app/models
    path: /app/models
```

## 🛠️ Desenvolvimento Local

### Setup

```bash
# 1. Clonar repositório
git clone https://github.com/frpbotero/wine_quality.git
cd wine_quality

# 2. Criar virtualenv
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar .env
cp .env.example .env
# Editar .env com seus valores reais
```

### Estrutura de Diretórios

```
wine_project/
├── main.py              # FastAPI app
├── streamlit_ui.py      # Streamlit UI
├── database.py          # Supabase connection
├── models.py            # ML utilities
├── supabase_logger.py   # Logging
│
├── src/                 # ML Pipeline
│   ├── ingestion.py     # Carregar dados
│   ├── preprocessing.py # Processar features
│   ├── prepare_data.py  # Split train/val/test
│   ├── train.py         # Treinar modelos
│   └── evaluate.py      # Avaliar modelos
│
├── data/
│   ├── raw/             # CSV original (6497 linhas)
│   ├── processed/       # Parquet processado
│   └── splits/          # Train/val/test splits
│
├── models/
│   ├── trained/         # 12 modelos joblib (SMOTE + Tomek)
│   ├── preprocessors/   # ColumnTransformers
│   └── best_model.pkl   # Modelo campeão
│
├── reports/             # Métricas e plots
│
├── dvc.yaml             # DVC pipeline
├── requirements-api.txt # Dependências Python
└── Dockerfile           # Build container
```

### DVC Pipeline

```bash
# Ativar pipeline DVC (ingest → preprocess → prepare → train → evaluate)
cd wine_project
dvc repro

# Resultados:
# ✅ 12 modelos treinados
# ✅ Métricas de validação/teste
# ✅ Confusion matrices e ROC curves
```

### Streamlit Local

```bash
cd wine_project
streamlit run streamlit_ui.py --server.port=8501
```

Acesse em: **http://localhost:8501**

### FastAPI Local

```bash
cd wine_project
python main.py
```

Acesse em: **http://localhost:8000/docs**

## 📈 Resultados

### Modelos Treinados

| Modelo | Val F1 | Test F1 | Estratégia |
|--------|--------|---------|-----------|
| SMOTE_random_forest | 0.6845 | **0.6933** | Over-sampling |
| Tomek_xgboost | 0.6774 | 0.6952 | Under-sampling |
| Tomek_random_forest | 0.6749 | 0.6974 | Under-sampling |
| SMOTE_xgboost | 0.6733 | 0.6749 | Over-sampling |

**🏆 Vencedor**: SMOTE_random_forest (Test F1 ≈ 0.69)

### Distribuição de Classes

```
Ruim (0):   38%  (2384 amostras)
Médio (1):  44%  (2836 amostras)  
Bom (2):    20%  (1277 amostras)
```

## 🔧 Endpoints FastAPI

```bash
POST /predict
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "ph": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "type": "red"
}

Response:
{
  "quality": 2,
  "probabilities": {
    "Ruim": 0.15,
    "Médio": 0.25,
    "Bom": 0.60
  },
  "model": "wine_SMOTE_random_forest"
}
```

## 📚 Tecnologias

- **ML**: scikit-learn, xgboost, imblearn (SMOTE/Tomek)
- **Data**: pandas, duckdb, numpy
- **Tracking**: MLflow, DagsHub
- **API**: FastAPI, Uvicorn
- **UI**: Streamlit
- **Database**: Supabase (PostgreSQL)
- **Deployment**: Docker, Render.com
- **Version Control**: DVC (Data Version Control)

## 📝 Notas

- Dataset contém 6.497 amostras com 11 features numéricas + 1 categórica
- Balanceamento de dados necessário (distribuição desigual)
- Modelos salvos como joblib para inference rápida
- Suporte a múltiplas estratégias de balanceamento

## 👤 Autor

Fernando Botero - [@frpbotero](https://github.com/frpbotero)

## 📄 Licença

MIT License
