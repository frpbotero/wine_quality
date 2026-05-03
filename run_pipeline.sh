#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Wine Quality Classifier - Pipeline Quick Start
# ════════════════════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║          🍷 Wine Quality Classifier - ML Pipeline                        ║
║                    Reproduzível com DVC + MLflow                         ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 não encontrado${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python3 detectado${NC}"

# Check requirements
if [ ! -f "$ROOT/requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt não encontrado${NC}"
    exit 1
fi

# Install dependencies
echo -e "${BLUE}📦 Instalando dependências...${NC}"
pip install -q -r "$ROOT/requirements.txt" || {
    echo -e "${RED}❌ Erro ao instalar dependências${NC}"
    exit 1
}
echo -e "${GREEN}✅ Dependências instaladas${NC}"

# Check .env
if [ ! -f "$ROOT/.env" ]; then
    if [ -f "$ROOT/.env.example" ]; then
        echo -e "${YELLOW}⚠️  .env não encontrado. Copiando .env.example...${NC}"
        cp "$ROOT/.env.example" "$ROOT/.env"
        echo -e "${YELLOW}⚠️  Configure suas credenciais em $ROOT/.env${NC}"
    else
        echo -e "${YELLOW}⚠️  .env não encontrado. Algumas funcionalidades podem não funcionar.${NC}"
    fi
else
    echo -e "${GREEN}✅ .env encontrado${NC}"
fi

# Run pipeline
echo ""
echo -e "${BLUE}🔄 Executando pipeline DVC...${NC}"
echo ""

cd "$ROOT"

stages=(
    "ingest:Ingestion (fetch dados)"
    "preprocess:Preprocessing (feature engineering)"
    "prepare:Prepare (split + preprocessor)"
    "train:Training (12 modelos)"
    "evaluate:Evaluation (test metrics + plots)"
)

for stage in "${stages[@]}"; do
    stage_name="${stage%%:*}"
    stage_desc="${stage##*:}"
    
    echo -e "${YELLOW}→ ${stage_desc}${NC}"
    
    if python3 "src/${stage_name}.py" 2>&1 | tail -5; then
        echo -e "${GREEN}✅ ${stage_desc} concluído${NC}"
    else
        echo -e "${RED}❌ ${stage_desc} falhou${NC}"
        exit 1
    fi
    echo ""
done

echo -e "${GREEN}🎉 Pipeline completo com sucesso!${NC}"
echo ""
echo -e "${BLUE}📊 Próximos passos:${NC}"
echo "  1. Streamlit UI:  streamlit run streamlit_ui.py"
echo "  2. FastAPI:       python main.py"
echo "  3. MLflow UI:     mlflow ui"
echo ""
echo -e "${GREEN}📈 Resultados:${NC}"
echo "  - Modelos treinados: models/trained/"
echo "  - Melhor modelo: app/best_model.pkl"
echo "  - Relatórios: reports/"
echo "  - Métricas: reports/{training,evaluation}_report.json"
echo ""
