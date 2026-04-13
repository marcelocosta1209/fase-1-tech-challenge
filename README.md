# Tech Challenge - Fase 1
# Sistema Inteligente de Suporte ao Diagnostico Medico

## Sobre o Projeto

<!-- Este projeto implementa um sistema de IA para apoio ao diagnostico medico,
     focado em classificacao de cancer de mama usando Machine Learning (dados tabulares)
     e Visao Computacional com CNN (imagens de mamografia).
     O objetivo e acelerar a triagem e apoiar decisoes medicas, sem substituir o profissional. -->

Um grande hospital universitario busca implementar um sistema inteligente de suporte
ao diagnostico, capaz de ajudar medicos e equipes clinicas na analise inicial de exames.

Este projeto aplica **Machine Learning** e **Visao Computacional** para:
- Classificar resultados de exames tabulares (FNA) como benigno ou maligno
- Analisar imagens de mamografia com CNN e identificar regioes suspeitas (EXTRA)
- Explicar as decisoes do modelo com SHAP e Grad-CAM
- Simular um caso clinico real integrando ambas as analises

## Equipe

<!-- Cada membro e responsavel por uma parte especifica do projeto,
     conforme as historias definidas no backlog agil -->

| Membro | Responsabilidade | Historia(s) |
|--------|-----------------|-------------|
| Antonio | EDA e Exploracao de Dados | Historia 1 |
| Renato | Pre-processamento de Dados | Historia 2 |
| Vinicius Geisler | Modelagem Tabular e Avaliacao | Historias 3 e 4 |
| Marcelo | Modelo de Imagem CNN (EXTRA) | Historias 5, 6 e 7 |
| Vinicius Blasque | Integracao, Caso Clinico e Entrega | Historias 8, 9 e 10 |

## Estrutura do Projeto

<!-- Organizacao dos arquivos e pastas do projeto -->

```
tech-challenge/
├── notebooks/                            # Notebooks Jupyter (codigo principal)
│   ├── 01_eda_tabular.ipynb             # Exploracao e analise de dados
│   ├── 02_modelagem_tabular.ipynb       # Pre-processamento + modelos ML + SHAP
│   ├── 03_modelagem_imagem.ipynb        # CNN para mamografia + Grad-CAM (EXTRA)
│   └── 04_caso_clinico.ipynb            # Simulacao de caso clinico real
├── data/                                 # Dados e graficos gerados
│   └── images/                          # Dataset de imagens (baixar do Kaggle)
│       └── breast_cancer/               # Imagens de mamografia organizadas
│           ├── train/                   # Dados de treino
│           │   ├── benign/              # Imagens de casos benignos
│           │   └── malignant/           # Imagens de casos malignos
│           ├── val/                     # Dados de validacao
│           └── test/                    # Dados de teste
├── models/                               # Modelos treinados salvos (.pkl, .keras)
├── instrucoes/                           # PDF com instrucoes do desafio
├── requirements.txt                      # Dependencias Python do projeto
├── Dockerfile                            # Container Docker para execucao
└── README.md                             # Este arquivo
```

## Datasets Utilizados

### Dados Tabulares (Obrigatorio)
<!-- Dataset do scikit-learn - nao precisa baixar, ja vem incluso -->
- **Breast Cancer Wisconsin (Diagnostic)**
- Fonte: incluso no scikit-learn (`sklearn.datasets.load_breast_cancer`)
- 569 amostras, 30 features numericas
- Classificacao binaria: Maligno (0) ou Benigno (1)
- **Nao precisa baixar nenhum arquivo CSV**

### Dados de Imagem (EXTRA)
<!-- Dataset do Kaggle - precisa baixar manualmente -->
- **CBIS-DDSM - Breast Cancer Image Dataset** (mamografias)
- Fonte: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
- Apos baixar, organizar em pastas `train/val/test` com subpastas `benign/malignant`
- Se nao baixar, o Notebook 03 gera imagens sinteticas de demonstracao

## Como Executar

### Pre-requisitos
<!-- Ferramentas necessarias antes de comecar -->
- Python 3.11 ou superior
- VS Code com extensoes **Python** e **Jupyter** (recomendado)
- Ou: Jupyter Notebook / JupyterLab

### Opcao 1: VS Code (Recomendado)

<!-- Passo a passo para rodar no VS Code -->

```bash
# 1. Criar ambiente virtual (isola as dependencias do projeto)
python -m venv venv

# 2. Ativar o ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalar todas as dependencias
pip install -r requirements.txt

# 4. Abrir o VS Code na pasta do projeto
code .
```

<!-- No VS Code:
     - Abrir cada notebook (.ipynb)
     - Selecionar o kernel do venv (canto superior direito)
     - Executar celula por celula com Shift+Enter
     - Seguir a ordem: 01 -> 02 -> 03 -> 04 -->

### Opcao 2: Docker

<!-- Para rodar em container sem instalar nada localmente -->

```bash
# Construir a imagem Docker
docker build -t tech-challenge .

# Rodar o container (mapeando a porta do Jupyter)
docker run -p 8888:8888 -v $(pwd)/data:/app/data tech-challenge
```

<!-- Acessar http://localhost:8888 no navegador -->

### Opcao 3: Jupyter Notebook direto

```bash
pip install -r requirements.txt
jupyter notebook
```

## Ordem de Execucao dos Notebooks

<!-- IMPORTANTE: Seguir esta ordem! O notebook 04 depende dos modelos
     salvos pelos notebooks 02 e 03 -->

| Ordem | Notebook | Descricao | Dependencia |
|-------|----------|-----------|-------------|
| 1 | `01_eda_tabular.ipynb` | Explorar dados, graficos, correlacao | Nenhuma |
| 2 | `02_modelagem_tabular.ipynb` | Treinar modelos, SHAP, salvar modelo | Nenhuma |
| 3 | `03_modelagem_imagem.ipynb` | CNN + Grad-CAM (EXTRA) | Dataset imagens |
| 4 | `04_caso_clinico.ipynb` | Caso clinico integrado | Notebooks 02 e 03 |

## Roteiro de Apresentacao (por Historia)

<!-- Use este roteiro para o video de demonstracao de ate 15 minutos -->

### Historia 1 - Ingestao e Exploracao de Dados (Antonio)
**Notebook: 01_eda_tabular.ipynb**
1. Mostrar o carregamento do dataset (569 amostras, 30 features)
2. Apresentar as estatisticas descritivas
3. Explicar a distribuicao da variavel alvo (63% benigno, 37% maligno)
4. Mostrar os histogramas separados por classe
5. Mostrar os boxplots e destacar a separacao entre classes
6. Explicar a matriz de correlacao e as features mais relevantes
7. Concluir com os insights principais da EDA

### Historia 2 - Pre-processamento (Renato)
**Notebook: 02_modelagem_tabular.ipynb (Secoes 3-6)**
1. Mostrar a verificacao de dados nulos e inconsistentes
2. Explicar a separacao features (X) vs target (y)
3. Demonstrar o split estratificado (60/20/20)
4. Explicar o StandardScaler e por que fit apenas no treino (data leakage)
5. Mostrar que a media ficou ~0 e desvio padrao ~1

### Historia 3 - Treinamento de Modelos (Vinicius Geisler)
**Notebook: 02_modelagem_tabular.ipynb (Secoes 7-10)**
1. Apresentar os 4 modelos escolhidos e justificar cada um
2. Mostrar a tabela comparativa de metricas
3. Destacar o Recall como metrica principal (contexto medico)
4. Mostrar as matrizes de confusao
5. Explicar as curvas ROC e o significado do AUC

### Historia 4 - Avaliacao e Interpretacao (Vinicius Geisler)
**Notebook: 02_modelagem_tabular.ipynb (Secoes 11-15)**
1. Mostrar a avaliacao final no conjunto de teste
2. Apresentar o feature importance do Random Forest
3. Explicar o SHAP Summary Plot (como cada feature influencia)
4. Discutir se o modelo pode ser usado na pratica
5. Mencionar limitacoes e a importancia do medico na decisao final

### Historia 5 - Preparacao de Imagens (Marcelo) [EXTRA]
**Notebook: 03_modelagem_imagem.ipynb (Secoes 1-4)**
1. Explicar o dataset de mamografia (CBIS-DDSM)
2. Mostrar a organizacao das pastas (train/val/test)
3. Demonstrar o data augmentation (rotacao, zoom, flip)
4. Mostrar exemplos visuais das imagens

### Historia 6 - Treinamento CNN (Marcelo) [EXTRA]
**Notebook: 03_modelagem_imagem.ipynb (Secoes 5-7)**
1. Explicar o transfer learning com MobileNetV2
2. Mostrar a arquitetura do modelo (camadas congeladas + novas)
3. Apresentar as curvas de treinamento (loss e accuracy)
4. Explicar os callbacks (EarlyStopping, ReduceLROnPlateau)

### Historia 7 - Avaliacao CNN e Grad-CAM (Marcelo) [EXTRA]
**Notebook: 03_modelagem_imagem.ipynb (Secoes 8-11)**
1. Mostrar metricas no conjunto de teste
2. Apresentar a matriz de confusao da CNN
3. Demonstrar o Grad-CAM: como o modelo identifica regioes suspeitas
4. Mostrar a classificacao BI-RADS simplificada
5. Explicar a importancia da explicabilidade visual em radiologia

### Historia 8 - Caso Clinico (Vinicius Blasque)
**Notebook: 04_caso_clinico.ipynb**
1. Apresentar o cenario clinico (paciente 52 anos, nodulo suspeito)
2. Mostrar a predicao do modelo tabular com probabilidades
3. Explicar a escala de risco (baixo/moderado/alto/muito alto)
4. Demonstrar o SHAP Waterfall para o caso individual
5. Mostrar a analise de imagem com Grad-CAM (se disponivel)
6. Apresentar o relatorio integrado

### Historia 9 - Interpretacao Critica (Vinicius Blasque)
**Notebook: 04_caso_clinico.ipynb (Secao 9)**
1. Discutir: o modelo pode ser usado na pratica?
2. Explicar por que Recall e a metrica mais importante
3. Apresentar as limitacoes do sistema
4. Mencionar consideracoes eticas (LGPD, transparencia, responsabilidade)
5. Reforcar: o medico SEMPRE tem a palavra final

### Historia 10 - Entregaveis Finais (Vinicius Blasque)
1. Mostrar a estrutura do repositorio no GitHub
2. Demonstrar o Dockerfile funcionando
3. Apresentar este README com instrucoes de execucao
4. Mostrar os graficos e resultados salvos na pasta data/

## Tecnologias Utilizadas

<!-- Bibliotecas e ferramentas do projeto -->

| Tecnologia | Uso no Projeto |
|-----------|----------------|
| Python 3.11 | Linguagem principal |
| pandas | Manipulacao de dados tabulares |
| numpy | Operacoes numericas |
| matplotlib | Graficos e visualizacoes |
| seaborn | Graficos estatisticos |
| scikit-learn | Modelos de ML (LogReg, RF, KNN, SVM) |
| TensorFlow/Keras | CNN com MobileNetV2 (transfer learning) |
| SHAP | Explicabilidade do modelo tabular |
| Grad-CAM | Explicabilidade visual da CNN |
| OpenCV | Processamento de imagens |
| Docker | Containerizacao do projeto |

## Modelos Implementados

### Dados Tabulares
<!-- 4 algoritmos de classificacao comparados -->
- **Regressao Logistica**: modelo linear, interpretavel, baseline
- **Random Forest**: ensemble de arvores, robusto a outliers
- **KNN (K-Nearest Neighbors)**: classificacao por vizinhanca
- **SVM (Support Vector Machine)**: maximiza margem de separacao

### Dados de Imagem (EXTRA)
<!-- Transfer learning com rede pre-treinada -->
- **MobileNetV2** (transfer learning do ImageNet)
- Camadas de classificacao customizadas
- Data augmentation para reducao de overfitting

## Metricas de Avaliacao

<!-- Metricas escolhidas considerando o contexto medico -->

- **Accuracy**: proporcao de acertos gerais
- **Recall (Sensibilidade)**: capacidade de detectar casos malignos - **METRICA PRINCIPAL**
- **Precision**: proporcao de malignos preditos que sao realmente malignos
- **F1-Score**: media harmonica entre precision e recall
- **AUC-ROC**: capacidade de discriminacao do modelo

**Por que Recall e a metrica mais importante?**
Em diagnostico de cancer, um falso negativo (cancer nao detectado) e muito mais
grave que um falso positivo (alarme falso). O paciente com cancer nao detectado
pode perder a janela de tratamento.

## Resultados Esperados

<!-- Resultados tipicos ao executar o projeto -->
- Accuracy acima de 95% nos modelos tabulares
- Recall para classe maligno acima de 90%
- AUC-ROC acima de 0.98
- Graficos SHAP mostrando worst concave points e worst radius como features principais
- Grad-CAM identificando regioes de nodulos em mamografias

## Consideracoes Eticas

<!-- Aspectos eticos importantes do projeto -->
- O sistema e uma **ferramenta de apoio**, nao substitui o medico
- O diagnostico definitivo SEMPRE deve ser feito por profissional qualificado
- Dados medicos devem seguir LGPD (Lei Geral de Protecao de Dados)
- Pacientes devem ser informados sobre o uso de IA no processo
- O modelo pode ter vieses relacionados ao dataset de treinamento
