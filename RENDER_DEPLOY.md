# 📦 Render Deploy Checklist

## ✅ Arquivos Criados

- ✅ `Dockerfile` (raiz) - Multi-stage build otimizado
- ✅ `requirements.txt` - Consolidado (23 dependências Python)
- ✅ `render.yaml` - Config automática
- ✅ `.env.example` - Template de variáveis
- ✅ `.dockerignore` - Otimizar build
- ✅ `README.md` - Documentação completa

## 🔧 Configuração no Render.com

### 1. Conectar Repositório
```
GitHub: https://github.com/frpbotero/wine_quality
Branch: master
```

### 2. Criar Secret File (.env)
No dashboard do Render, adicione variáveis de ambiente:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key_here
DAGSHUB_USERNAME=frpbotero
DAGSHUB_REPO_NAME=wine-quality
DAGSHUB_TOKEN=your_token_here
API_SECRET_KEY=random_secret_key
```

### 3. Build Settings
- **Dockerfile**: `./Dockerfile`
- **Port**: `8501` (Streamlit)
- **Health Check**: `/health` (se implementado)

### 4. Deploy
- Render detecta `render.yaml`
- Build automático com `docker build`
- Deploy em container isolado

## 📊 Stack Deployado

```
┌─────────────────────────────────────────┐
│     Streamlit UI (Port 8501)            │
│  ┌───────────────────────────────────┐  │
│  │  Predição | Comparação | Histórico│  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│     FastAPI Backend (Port 8000)         │
│  ┌───────────────────────────────────┐  │
│  │  /predict  /models  /health       │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│     ML Models (12 joblib)               │
│  ┌───────────────────────────────────┐  │
│  │  SMOTE_random_forest (champion)   │  │
│  │  SMOTE_xgboost, SMOTE_knn, ...    │  │
│  │  Tomek_random_forest, ...         │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│     Supabase DB (Data Source)           │
│  wine_quality table (6497 rows)         │
└─────────────────────────────────────────┘
```

## 🚨 Possíveis Erros e Soluções

### Erro: "Dockerfile not found"
✅ **Resolvido**: Dockerfile movido para raiz

### Erro: "requirements.txt not found"
✅ **Resolvido**: requirements.txt criado na raiz

### Erro: "Port already in use"
✅ **Render**: Automático (expõe 8501)

### Erro: ".env variables missing"
✅ **Setup**: Adicione variáveis via Secret File

### Erro: "Model files not found"
⚠️ **Nota**: Modelos devem estar em `wine_project/models/trained/`
- Se faltarem, treinar localmente: `cd wine_project && dvc repro`

## 🔄 CI/CD Automático

Após cada push para `master`:
1. Render clona repositório
2. Docker build a imagem
3. Deploy em container
4. Acessa em: `https://wine-quality.onrender.com`

## 📞 Debugging

```bash
# Ver logs do Render
Render Dashboard → Service → Logs

# Verificar se container está rodando
curl https://wine-quality.onrender.com/health

# Testar API
curl -X POST https://wine-quality.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"fixed_acidity": 7.4, ...}'
```

## ✨ Próximos Passos

1. ✅ Deploy no Render
2. ⏳ Testar Streamlit UI em produção
3. ⏳ Monitizar com Supabase logs
4. ⏳ Adicionar model versioning no MLflow
5. ⏳ Setup de retraining automático (cron job)

---

**Status**: ✅ Pronto para Deploy
**Último Update**: May 3, 2026
