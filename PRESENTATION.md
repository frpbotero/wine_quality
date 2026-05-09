# Wine Quality Classifier

Classificação Inteligente de Vinhos com Machine Learning e MLOps

Equipe 
Felipe Botero
Hanna Souza
José Henrique

---

# Visão Geral do Projeto

## 🎯 Objetivo

Classificar vinhos em **"Bom"** ou **"Ruim"** com acurácia superior a 85%, utilizando 11 atributos físico-químicos e o tipo do vinho (tinto ou branco).

## ⚠️ O Desafio

Superar o desbalanceamento da classificação original (notas 3-9), onde ~70% dos dados concentravam-se nas notas intermediárias (5-7), limitando a acurácia a ~69%.

## 💡 A Solução

Abordagem binária com threshold fixo em 7 (0=Ruim, 1=Bom), combinada com duas estratégias de balanceamento (SMOTE e Tomek Links) e 6 algoritmos distintos — totalizando 12 modelos treinados e comparados automaticamente.

---

# Stack Tecnológico

## 🔀 Pipeline & Tracking

**DVC** para controle de versão de dados e reprodutibilidade do pipeline. **MLflow** com **DagsHub** para rastreamento rigoroso de experimentos e registro de modelos.

## 🗄️ Processamento & Armazenamento

**DuckDB** para ingestão e transformação de dados em alta performance — com acesso direto ao Supabase via extensão Postgres. **Supabase** como fonte de verdade centralizada dos dados de vinho.

## 🧠 Modelagem

**scikit-learn** (Logistic Regression, Random Forest, KNN, SVM, MLP) e **XGBoost** — 6 algoritmos treinados com 2 estratégias de balanceamento cada.

## 📦 Deploy

**Streamlit** para interface interativa com predição embutida e **Docker** para entrega portável e isolada.

---

# EDA: Insights dos Dados

## Dataset

~6.497 amostras combinando vinhos tintos (~1.599) e brancos (~4.898), com 11 atributos físico-químicos como acidez, teor alcoólico, sulfatos e cloretos.

## Desbalanceamento Crítico

Notas 5, 6 e 7 concentravam ~85% dos dados. A binarização (qualidade ≥ 7 → "Bom") gerou classes mais equilibradas e aumentou a acurácia em ~25 pontos percentuais.

## Variabilidade por Tipo

Vinhos brancos possuem maior teor de açúcar residual e dióxido de enxofre total, enquanto tintos apresentam maior acidez volátil e sulfatos — exigindo One-Hot Encoding para capturar essa diferença.

## Correlações Relevantes

O teor alcoólico é o maior preditor positivo de qualidade, enquanto a acidez volátil e a densidade apresentam correlação negativa com notas mais altas.

---

# Estratégia de Classificação Binária

## O Problema Original: Multiclasse Desbalanceada

A classificação com notas de 3 a 9 produzia F1 ≈ 69% — o modelo falhava especialmente nas classes extremas (notas 3, 4, 8, 9) por escassez de exemplos.

## A Transformação: Threshold Fixo em 7

O `preprocessing.py` aplica via **DuckDB SQL**:
```sql
CASE WHEN CAST(quality AS INTEGER) < 7 THEN 0 ELSE 1 END AS quality
```
Eliminando multiclasse e reduzindo ruído nas fronteiras de decisão.

## O Resultado

F1-Score passou de ~69% (multiclasse) para **95%** (binário com Tomek + Random Forest) — um ganho de **+26 pontos percentuais**.

## Classes Finais

- `0 = Ruim` — vinhos com nota original abaixo de 7
- `1 = Bom` — vinhos com nota original ≥ 7

---

# Multiclasse vs Binário: Comparativo Real

> Ambas as abordagens foram treinadas com os **mesmos algoritmos, estratégias de balanceamento e dados** — a única diferença é a definição do target.

## 📊 Comparativo por Modelo (Test F1)

| Modelo | Multiclasse (3 classes) | Binário (0/1) | Ganho |
|--------|------------------------|--------------|-------|
| **Tomek_random_forest** | 62.30% | **95.03%** | **+32.73pp** |
| Tomek_svm | 63.35% | — | — |
| Tomek_xgboost | 59.71% | 94.90% | +35.19pp |
| SMOTE_random_forest | 58.80% | 93.97% | +35.17pp |
| SMOTE_xgboost | 57.69% | 94.52% | +36.83pp |
| Tomek_logistic_regression | 58.98% | — | — |
| SMOTE_svm | 56.15% | — | — |
| SMOTE_logistic_regression | 55.42% | — | — |

## ⚠️ Por que a Multiclasse Falha?

O dataset possui **5 classes efetivas** (notas 4, 5, 6, 7, 8) com distribuição muito desigual — a nota 4 tem apenas 49 amostras no teste contra 567 da nota 6. Isso gera F1 de apenas **32%** para a classe mais rara.

## ✅ Por que o Binário Funciona?

A binarização agrupa as notas em dois blocos coesos, reduz o ruído nas fronteiras de decisão e cria classes com volume suficiente para aprendizado robusto — resultando em **+35pp de ganho médio no F1**.

---

# Engenharia de Atributos & Pré-processamento

## 🦆 DuckDB SQL Inline

Toda a transformação de dados ocorre em SQL diretamente sobre o Supabase Postgres ou parquets locais — sem carga desnecessária em memória.

## 🔄 Detecção Automática de Schema

O `preprocessing.py` detecta automaticamente se o CSV segue o schema UCI original (colunas com espaços) ou o schema normalizado do Supabase (underscores) e aplica o SQL correto.

## 🏷️ Encoding Categórico

O atributo `type` (tinto/branco) é convertido via **One-Hot Encoding** (`type_red`, `type_white`), gerando 13 features totais para o modelo.

## ⚖️ Normalização no Pipeline

**StandardScaler** aplicado via `ColumnTransformer` dentro de um `Pipeline` do scikit-learn, garantindo que o scaler seja fitado apenas no treino e aplicado consistentemente na inferência.

---

# Pipeline de Dados (DVC)

## 📥 INGESTION
Coleta dados do Supabase (via API ou PostgreSQL direto) ou Kaggle como fallback. Normaliza colunas e salva em `data/raw/wine_quality.csv`.

↓

## 🛠️ PREPROCESSING
DuckDB transforma e binariza a variável alvo. Salva `data/processed/wine_processed.parquet`.

↓

## 🧪 PREPARE DATA
Split estratificado 60/20/20 (treino/validação/teste) com `random_state=42`. Gera `train.parquet`, `val.parquet` e `test.parquet`.

↓

## 🖥️ TRAIN
Treina 12 modelos (6 algoritmos × SMOTE + Tomek). Registra no MLflow. Promove o melhor como `@champion`. Salva `models/best_model.pkl`.

↓

## 📊 EVALUATE
Avalia todos os 12 modelos no test set puro. Gera `evaluation_report.json`, matrizes de confusão e curvas ROC.

---

## ✅ Benefícios do DVC

**Reprodutibilidade total:** `dvc repro` reconstrói todo o pipeline do zero, detectando mudanças em cada stage pelo hash dos inputs.

**Versionamento de dados:** Parquets e modelos rastreados — qualquer estado anterior pode ser auditado e restaurado.

**Integração com Supabase:** A fonte de dados primária é sempre o Supabase, com fallback automático para parquets locais.

---

# Estratégias de Balanceamento

## ⬆️ SMOTE — Over-sampling

Cria amostras sintéticas da classe minoritária via interpolação entre vizinhos, aumentando o conjunto de treino para equilibrar as classes.

## ⬇️ Tomek Links — Under-sampling

Remove amostras da classe majoritária que estão próximas da fronteira de decisão, "limpando" a separação entre classes e reduzindo ruído.

## 🏆 Resultado do Embate

Tomek Links + Random Forest venceu com **F1 de Validação = 87.43%** e **F1 de Teste = 95.03%**, sendo promovido automaticamente como modelo `@champion` no MLflow.

---

# Treinamento e Validação

## 🧠 6 Algoritmos Comparados

| Algoritmo | Tipo |
|-----------|------|
| Logistic Regression | Linear |
| Random Forest | Ensemble (Bagging) |
| K-Nearest Neighbors | Instance-based |
| SVM | Margem de decisão |
| XGBoost | Ensemble (Boosting) |
| MLP | Rede Neural |

## 📐 Split Estratificado 60/20/20

O split preserva a proporção das classes em treino, validação e teste — eliminando data leakage entre os conjuntos.

## 🎯 Métricas Monitoradas

F1-Score Ponderado (principal), Acurácia, Precisão e Recall — todas registradas no MLflow por run.

---

# Resultados de Validação

| Modelo | Val Accuracy | Val F1 |
|--------|-------------|--------|
| **Tomek_random_forest** | **88.09%** | **87.43%** |
| Tomek_xgboost | 86.00% | 85.57% |
| SMOTE_random_forest | 86.39% | 86.75% |
| SMOTE_xgboost | 85.54% | 85.84% |
| SMOTE_mlp | 84.07% | 84.29% |
| Tomek_mlp | 83.91% | 84.05% |
| Tomek_knn | 83.29% | 82.89% |
| Tomek_svm | 83.60% | 81.56% |
| SMOTE_knn | 77.11% | 79.01% |
| SMOTE_svm | 75.72% | 77.90% |
| Tomek_logistic_regression | 81.75% | 79.93% |
| SMOTE_logistic_regression | 71.23% | 74.05% |

---

# Resultados no Teste (Dados Não Vistos)

Todos os **12 modelos** avaliados no conjunto de teste puro (20% dos dados, nunca vistos durante treino ou validação).

## 🥇 95.03% F1 — Tomek + Random Forest

Melhor modelo em produção (`@champion`). Alta robustez e generalização no conjunto de teste puro.

## 🥈 94.90% F1 — Tomek + XGBoost

Desempenho muito próximo ao campeão, com excelente capacidade de capturar interações não lineares.

## 🥉 94.60% F1 — SMOTE + XGBoost

Boosting com dados sintéticos entregou performance consistente e competitiva.

## 📈 93.97% F1 — SMOTE + Random Forest

Bagging com oversampling — combinação robusta, especialmente para dados desbalanceados.

---

| Modelo | Test Accuracy | Test F1 |
|--------|--------------|---------|
| **Tomek_random_forest** | **95.08%** | **95.03%** |
| Tomek_xgboost | 94.92% | 94.90% |
| SMOTE_xgboost | 94.54% | 94.60% |
| SMOTE_random_forest | 93.85% | 93.97% |
| SMOTE_mlp | 93.46% | 93.55% |
| Tomek_mlp | 92.92% | 93.04% |
| Tomek_knn | 86.77% | 86.36% |
| Tomek_svm | 83.69% | 81.49% |
| SMOTE_knn | 80.85% | 82.42% |
| SMOTE_svm | 76.92% | 78.96% |
| Tomek_logistic_regression | 81.69% | 79.60% |
| SMOTE_logistic_regression | 72.46% | 75.08% |

---

# Arquitetura de Deploy

## 🌐 Streamlit

Interface com **3 abas** e predição embutida diretamente via `joblib` — sem backend separado:
- **🔮 Predição** — sliders para todas as 13 features + predição em tempo real com gráfico de probabilidades
- **📊 Comparação** — métricas de validação vs teste, F1-Score charts, análise de overfitting
- **📜 Histórico** — predições anteriores registradas no Supabase via `supabase_logger`

## 📦 Docker

```dockerfile
docker run --env-file .env -p 8501:8501 wine-quality
```
Containerização completa com variáveis de ambiente injetadas — mesmo comportamento local e em produção (Render).

---

## 🗄️ Supabase como Backbone Operacional

Além de armazenar os dados brutos de treino, o Supabase registra cada predição realizada pela UI em tempo real via `supabase_logger.py` — permitindo auditoria e monitoramento de uso em produção.

---

# MLflow & DagsHub

## 📊 Rastreamento Completo

Cada um dos 12 modelos treinados gera um run no MLflow com parâmetros (estratégia, algoritmo, tamanho dos splits) e métricas (acurácia, F1, precisão, recall) registrados automaticamente.

## 🏆 Model Registry

Todos os modelos são registrados no MLflow Model Registry com nome padronizado `wine_{STRATEGY}_{algorithm}`. O campeão recebe o alias `@champion`.

## ☁️ DagsHub Integration

Quando as credenciais `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN` e `DAGSHUB_REPO_NAME` estão configuradas, o tracking ocorre remotamente — permitindo colaboração e auditoria distribuída.

## 🔄 Fallback Inteligente

Sem credenciais DagsHub → MLflow local em `./mlruns`. Sem MLflow → JSON estático em `reports/training_report.json`.

---

# Conclusões e Próximos Passos

## ✅ Meta Superada

A meta de >85% de acurácia foi amplamente superada: **95.08%** no conjunto de teste com Tomek + Random Forest — resultado obtido com dados nunca vistos durante o treinamento.

## 🔬 Rigor Metodológico

Separação estrita treino/validação/teste (60/20/20), sem data leakage. O conjunto de teste foi criado antes do treinamento e usado exclusivamente na avaliação final.

## 🚀 Reprodutibilidade Total

`dvc repro` reconstrói todo o pipeline do zero. Qualquer collaborador pode auditar, modificar e re-executar o experimento com o mesmo resultado.

## 🔭 Próximos Passos

- Implementar **monitoramento de drift** com Evidently ou Alibi Detect
- Adicionar **CI/CD** com GitHub Actions para re-treino automático ao detectar drift
- Expandir o dataset com dados de novos produtores para melhorar a generalização
- Explorar **SHAP values** para explicabilidade das predições na UI

---

# Obrigado!

Dúvidas?

**GitHub:** github.com/frpbotero/wine_quality  
**Deploy:** Render (Streamlit + Docker)  
**Tracking:** DagsHub / MLflow  
**Dados:** Supabase PostgreSQL
