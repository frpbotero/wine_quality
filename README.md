# 🍷 Wine Quality Classifier

**ML Pipeline + Streamlit UI** - Classificação de qualidade de vinho com balanceamento de dados (SMOTE/Tomek) e rastreamento remoto em DagsHub

## 📊 Projeto

- **Modelo Vencedor**: SMOTE + Random Forest (Val F1: 0.6845)
- **Estratégias**: SMOTE (over-sampling) e Tomek Links (under-sampling)
- **Classificação**: 3 classes (Ruim ≤5 / Médio 6 / Bom ≥7)
- **Features**: 11 químicas + 1 categórica (red/white) one-hot encoded = **13 features totais**
- **Dataset**: Wine Quality (UCI ML Repository) - 6.497 amostras
- **Logging**: Cada predição salva em Supabase + MLflow remoto em DagsHub

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

### Streamlit UI (Interface Principal)

```bash
cd wine_project
streamlit run streamlit_ui.py --server.port=8501
```

Acesse em: **http://localhost:8501**

**Layout da Interface:**

🔙 **Sidebar - Modelo em Uso**
```
📛 Nome: SMOTE_random_forest
⚙️ Estratégia: SMOTE
🧠 Algoritmo: random_forest

📊 Métricas de Validação:
  ✅ Acurácia: 0.6854
  🎯 F1-Score: 0.6845
  📈 Recall: 0.6800
  🔍 Precisão: 0.6890

🏆 Métricas de Teste:
  ✅ Acurácia: 0.6850
  🎯 F1-Score: 0.6933
  📈 Recall: 0.6900
  🔍 Precisão: 0.6970
```

📱 **Aba 1 - 🔮 Predição**
- Seletor de tipo: Red 🍷 / White 🥂
- 11 sliders para features químicas
- Botão "Classificar" → Predição em tempo real
- Gráfico de probabilidades por classe
- **Auto-save**: Predição registrada em Supabase com timestamp

📜 **Aba 2 - Histórico**
- Últimas 20 predições (sessão atual)
- Sincronização com API (se disponível)
- Visualização de tendências

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

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│              Streamlit UI (Port 8501)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SIDEBAR: Model Metadata + Performance Metrics      │   │
│  │  • Model: SMOTE_random_forest                       │   │
│  │  • Strategy: SMOTE (Over-sampling)                  │   │
│  │  • Val F1: 0.6845 | Test F1: 0.6933                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ 🔮 Predição     │  │ 📜 Histórico     │                │
│  │ • Type Selector │  │ • Last 20 Preds  │                │
│  │ • 11 Sliders    │  │ • Timestamps     │                │
│  │ • Probability   │  │ • Trends         │                │
│  │   Chart         │  │                  │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
         ↓ (Auto-save each prediction)
┌─────────────────────────────────────────────────────────────┐
│  Local Model Inference (joblib)                             │
│  ├─ One-hot Encoding: type → [type_red, type_white]       │
│  ├─ Log Transforms: sulphates, chlorides, residual_sugar  │
│  ├─ StandardScaler: All features                           │
│  └─ Best Model: SMOTE_random_forest.pkl                    │
└─────────────────────────────────────────────────────────────┘
         ↓ Predictions logged        ↑ Training data
┌─────────────────────────────────────────────────────────────┐
│  Supabase PostgreSQL (Auditoria)                            │
│  └─ wine_predictions table: features, quality, probs, ts   │
└─────────────────────────────────────────────────────────────┘

🔄 MLflow Remote Tracking (DagsHub)
└─ 12 models registered | Best promoted as "champion"
```

## 📚 Tecnologias

- **ML**: scikit-learn, xgboost, imblearn (SMOTE/Tomek)
- **Data**: pandas, numpy
- **Tracking**: MLflow (Remote to DagsHub), DVC
- **UI**: Streamlit (Local Inference)
- **Database**: Supabase PostgreSQL (Predictions Audit)
- **Deployment**: Docker, Render.com
- **Version Control**: Git, DVC (Data Version Control)

## 📝 Notas

- **Features**: 11 chemical properties + 1 categorical (red/white) one-hot encoded = **13 features totais**
- **Preprocessing**: Log transforms (sulphates, chlorides, residual_sugar) + StandardScaler normalization
- **Balanceamento**: SMOTE vs Tomek Links; SMOTE + Random Forest é o melhor performer
- **Inferência**: Local via joblib (não via API HTTP)
- **Auditoria**: Todas as predições registradas em Supabase com features, timestamp e probabilidades
- **MLflow Registry**: 12 modelos treinados; melhor modelo promovido com alias "champion"
- **DVC + DagsHub**: Versionamento de dados e modelos com integração MLflow remota

## 👤 Autor

Fernando Botero - [@frpbotero](https://github.com/frpbotero)

## 📄 Licença

MIT License
