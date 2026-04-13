#!/bin/bash
# ============================================================
# SCRIPT DE SETUP E EXECUÇÃO DO PROJETO
# Tech Challenge Fase 1 - Sistema de IA para Diagnóstico Médico
# ============================================================
# Uso:
#   bash run.sh          -> Setup completo + executa todos os notebooks
#   bash run.sh setup    -> Apenas instala dependências
#   bash run.sh execute  -> Apenas executa os notebooks (requer setup prévio)
# ============================================================

set -e  # Parar em caso de erro

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Sem cor

echo ""
echo "=========================================================="
echo "  Tech Challenge Fase 1 - Setup e Execução"
echo "  Sistema Inteligente de Suporte ao Diagnóstico"
echo "=========================================================="
echo ""

# Diretório do projeto (onde este script está)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ============================================================
# FUNÇÃO: SETUP (criar venv e instalar dependências)
# ============================================================
setup() {
    echo -e "${YELLOW}[1/3] Criando ambiente virtual Python...${NC}"
    if [ ! -d "venv" ]; then
        python -m venv venv
        echo -e "${GREEN}  Ambiente virtual criado!${NC}"
    else
        echo -e "${GREEN}  Ambiente virtual já existe, reutilizando.${NC}"
    fi

    # Ativar ambiente virtual
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate  # Windows (Git Bash)
    else
        source venv/bin/activate      # Linux/Mac
    fi

    echo -e "${YELLOW}[2/3] Instalando dependências...${NC}"
    python -m pip install --quiet --upgrade pip 2>/dev/null || true
    pip install --quiet -r requirements.txt
    echo -e "${GREEN}  Dependências instaladas!${NC}"

    echo -e "${YELLOW}[3/3] Criando pastas necessárias...${NC}"
    mkdir -p data/images/breast_cancer/{train,val,test}/{benign,malignant}
    mkdir -p models
    echo -e "${GREEN}  Estrutura de pastas criada!${NC}"

    echo ""
    echo -e "${GREEN}Setup concluído com sucesso!${NC}"
    echo ""
}

# ============================================================
# FUNÇÃO: EXECUTAR NOTEBOOKS
# ============================================================
execute() {
    # Ativar ambiente virtual
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi

    echo -e "${YELLOW}Executando notebooks...${NC}"
    echo "(Os gráficos serão salvos na pasta data/)"
    echo ""

    # Notebook 01 - EDA
    echo -e "${YELLOW}[1/4] Executando 01_eda_tabular.ipynb...${NC}"
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=300 \
        notebooks/01_eda_tabular.ipynb \
        --output 01_eda_tabular_executed.ipynb 2>/dev/null
    echo -e "${GREEN}  Notebook 01 executado com sucesso!${NC}"

    # Notebook 02 - Modelagem
    echo -e "${YELLOW}[2/4] Executando 02_modelagem_tabular.ipynb...${NC}"
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=600 \
        notebooks/02_modelagem_tabular.ipynb \
        --output 02_modelagem_tabular_executed.ipynb 2>/dev/null
    echo -e "${GREEN}  Notebook 02 executado com sucesso!${NC}"

    # Notebook 03 - CNN (EXTRA) - só executa se tiver imagens
    if [ -f "data/images/breast_cancer/train/benign/"*.png ] 2>/dev/null || \
       [ -f "data/images/breast_cancer/train/benign/"*.jpg ] 2>/dev/null; then
        echo -e "${YELLOW}[3/4] Executando 03_modelagem_imagem.ipynb (EXTRA)...${NC}"
        jupyter nbconvert --to notebook --execute \
            --ExecutePreprocessor.timeout=1200 \
            notebooks/03_modelagem_imagem.ipynb \
            --output 03_modelagem_imagem_executed.ipynb 2>/dev/null
        echo -e "${GREEN}  Notebook 03 executado com sucesso!${NC}"
    else
        echo -e "${YELLOW}[3/4] Pulando 03 (dataset de imagens não encontrado)${NC}"
        echo "  Para executar, baixe o dataset do Kaggle em data/images/breast_cancer/"
    fi

    # Notebook 04 - Caso Clínico
    echo -e "${YELLOW}[4/4] Executando 04_caso_clinico.ipynb...${NC}"
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=600 \
        notebooks/04_caso_clinico.ipynb \
        --output 04_caso_clinico_executed.ipynb 2>/dev/null
    echo -e "${GREEN}  Notebook 04 executado com sucesso!${NC}"

    echo ""
    echo -e "${GREEN}=========================================================${NC}"
    echo -e "${GREEN}  Todos os notebooks executados com sucesso!${NC}"
    echo -e "${GREEN}=========================================================${NC}"
    echo ""
    echo "Arquivos gerados:"
    echo "  - notebooks/*_executed.ipynb  (notebooks com outputs)"
    echo "  - data/*.png                  (gráficos salvos)"
    echo "  - models/*.pkl                (modelos treinados)"
    echo ""
    echo "Para abrir no VS Code:"
    echo "  code notebooks/01_eda_tabular_executed.ipynb"
    echo ""
}

# ============================================================
# MAIN - Processar argumentos
# ============================================================
open_jupyter() {
    # Ativar ambiente virtual
    if [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi

    echo -e "${GREEN}Abrindo Jupyter no navegador...${NC}"
    echo "Acesse: http://localhost:8888"
    echo "Pressione Ctrl+C para encerrar."
    echo ""
    jupyter notebook --notebook-dir=notebooks --no-browser --port=8888 2>&1 &
    sleep 2
    # Abrir no navegador
    if command -v start &> /dev/null; then
        start http://localhost:8888  # Windows
    elif command -v xdg-open &> /dev/null; then
        xdg-open http://localhost:8888  # Linux
    elif command -v open &> /dev/null; then
        open http://localhost:8888  # Mac
    fi
    wait
}

case "${1:-all}" in
    setup)
        setup
        ;;
    execute)
        execute
        ;;
    open|jupyter|web)
        open_jupyter
        ;;
    all|"")
        setup
        execute
        echo ""
        echo -e "${YELLOW}Para abrir no navegador, execute:${NC}"
        echo "  bash run.sh open"
        ;;
    *)
        echo "Uso: bash run.sh [setup|execute|open|all]"
        echo "  setup   - Apenas instala dependências"
        echo "  execute - Apenas executa notebooks"
        echo "  open    - Abre Jupyter no navegador"
        echo "  all     - Setup + execução (padrão)"
        exit 1
        ;;
esac
