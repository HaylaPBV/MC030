# MC030 — Análise Comparativa de Arquiteturas de Deep Learning para Predição de Tráfego em Redes 5G

Projeto final de graduação do curso de Engenharia de Computação da Universidade Estadual de Campinas (UNICAMP).

O objetivo é comparar modelos de aprendizado de máquina para a **previsão de séries temporais multivariadas** de tráfego em redes 5G com *network slicing*, avaliando Transformers, redes neurais clássicas e baselines estatísticos.

---

## Contexto

Redes 5G utilizam o conceito de *network slicing* para dividir a infraestrutura física em fatias lógicas independentes, cada uma otimizada para um perfil de uso:

| Slice | Descrição |
|-------|-----------|
| **eMBB** | Enhanced Mobile Broadband — alto throughput (ex.: streaming, AR/VR) |
| **URLLC** | Ultra-Reliable Low-Latency Communications — latência mínima (ex.: veículos autônomos, cirurgias remotas) |
| **mMTC** | massive Machine-Type Communications — grande volume de dispositivos IoT |

Prever a demanda futura de cada slice permite que operadoras aloquem recursos com antecedência, reduzindo desperdício e garantindo qualidade de serviço.

---

## Dataset

### Origem

Os dados provêm do dataset público **CRAWDAD umkc/networkslicing5g**, disponível no [IEEE DataPort](https://ieee-dataport.org/open-access/crawdad-umkcnetworkslicing5g) (DOI: [10.15783/k0w0-js18](https://dx.doi.org/10.15783/k0w0-js18)).

O dataset foi gerado pelo framework **DeepSlice**, cuja concepção é apresentada no artigo:

> Thantharate, A., Beard, C., Paropkari, R., Walunj, V. **"DeepSlice: A Deep Learning Approach towards an Efficient and Reliable Network Slicing in 5G Networks."** IEEE, 2019. [https://ieeexplore.ieee.org/document/8993066](https://ieeexplore.ieee.org/document/8993066)

### Arquivos em `data/`

| Arquivo | Descrição |
|---------|-----------|
| `5G_Dataset_Network_Slicing_CRAWDAD_Shared.xlsx` | Arquivo original do dataset, conforme disponibilizado pelos autores |
| `model_inputs.csv` | Derivado do original — contém todas as colunas (inputs e output do DeepSlice), usadas pelo pipeline de pré-processamento |

### Colunas de `model_inputs.csv`

| Coluna | Descrição |
|--------|-----------|
| `Use CaseType (Input 1)` | Tipo de dispositivo (ex.: Smartphone) |
| `LTE/5G UE Category (Input 2)` | Categoria do equipamento de usuário |
| `Technology Supported (Input 3)` | Tecnologia de acesso (LTE/5G) |
| `Day (Input4)` | Dia da semana (Monday … Sunday) |
| `Time (Input 5)` | Hora do dia (1 … 24) |
| `QCI (Input 6)` | QoS Class Identifier |
| `Packet Loss Rate (Reliability)` | Taxa de perda de pacotes |
| `Packet Delay Budget (Latency)` | Orçamento de atraso |
| `Slice Type (Output)` | Tipo de slice (eMBB, URLLC, mMTC) |

O pré-processamento agrega as requisições por hora e por slice, gerando uma série temporal multivariada com **168 timestamps** (7 dias × 24 horas) e 3 variáveis numéricas (contagem de requisições por slice).

### Citação do dataset

```bibtex
@data{k0w0-js18-22,
  doi       = {10.15783/k0w0-js18},
  url       = {https://dx.doi.org/10.15783/k0w0-js18},
  author    = {Anurag Thantharate and Cory Beard and Rahul Paropkari and Vijay Walunj},
  publisher = {IEEE Dataport},
  title     = {CRAWDAD umkc/networkslicing5g},
  year      = {2022}
}
```

---

## Pipeline

```
model_inputs.csv
      │
      ▼
1. Pré-processamento & Agregação   → série temporal (168 × 3)
2. Divisão Treino/Teste (70/30)    → split sequencial, sem shuffle
3. Normalização (Z-score)          → StandardScaler fitado só no treino
4. Janelamento deslizante          → look_back=48h, horizontes H∈{12, 24}
5. Modelagem
   ├── PatchTST (+ Estudo de Ablação + SSL)
   ├── CNN 1D (channel-independent)
   ├── LSTM
   ├── DLinear
   ├── SARIMA (sazonal, m=24)
   └── Baselines triviais (Naive, Seasonal-Naive, Ensemble CNN+LSTM)
6. Avaliação                       → MSE e MAE (global e por canal)
```

### Modelos avaliados

| Modelo | Família |
|--------|---------|
| `PatchTST_Original` | Transformer (Hugging Face) |
| `PatchTST_A_small` | Transformer — Ablação A (capacidade reduzida) |
| `PatchTST_AB_revin` | Transformer — Ablação A+B (+ RevIN) |
| `PatchTST_ABC_sincos` | Transformer — Ablação A+B+C (+ PE sincos) |
| `PatchTST_SSL` | Transformer — pré-treino auto-supervisionado |
| `CNN1D` | Rede neural convolucional |
| `LSTM` | Rede neural recorrente |
| `DLinear` | Decomposição linear (tendência + sazonalidade) |
| `Ensemble_CNN_LSTM` | Ensemble — média de CNN1D e LSTM |
| `SARIMA` | Estatístico sazonal (m=24) |
| `Naive` | Baseline trivial — último valor |
| `SNaive` | Baseline trivial — mesmo horário do dia anterior |

---

## Estrutura do repositório

```
MC030/
├── data/
│   ├── 5G_Dataset_Network_Slicing_CRAWDAD_Shared.xlsx   ← dataset original (não versionado)
│   └── model_inputs.csv                                 ← derivado do original (não versionado)
├── notebook/
│   └── benchmarking.ipynb    ← versão notebook (alternativa ao script)
├── src/
│   └── benchmarking.py       ← script principal
├── img/                      ← gerado automaticamente ao executar o script
├── metrics/                  ← gerado automaticamente ao executar o script
├── requirements.txt
└── README.md
```

> Os arquivos da pasta `data/` **não estão versionados** por serem grandes. Baixe o dataset original em [ieee-dataport.org](https://ieee-dataport.org/open-access/crawdad-umkcnetworkslicing5g) e coloque os dois arquivos em `data/` antes de executar.

---

## Pré-requisitos

- Python 3.10 ou superior
- pip

Para treino com **GPU** (recomendado para o PatchTST), instale o PyTorch com suporte a CUDA seguindo as instruções em [pytorch.org](https://pytorch.org/get-started/locally/) antes de executar o passo de instalação abaixo. Sem GPU, o notebook roda normalmente na CPU, porém mais lento.

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositório>
cd MC030

# 2. (Opcional) Crie e ative um ambiente virtual
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Execução

### Script Python (recomendado)

Execute a partir da **raiz do repositório** (não de dentro de `src/`):

```bash
python src/benchmarking.py
```

O script cria as pastas `img/` e `metrics/` automaticamente e salva todas as saídas nelas:

| Pasta | Conteúdo |
|-------|----------|
| `img/` | `series_overview.png`, `train_test_split.png`, `sliding_windows_sample.png`, `ablation_H{12,24}.png`, `neural_baselines_H{12,24}.png`, `statistical_baselines_H{12,24}.png` |
| `metrics/` | `metrics_global.csv`, `metrics_per_channel.csv`, `ablation_study.csv` |

### Notebook (alternativa)

Abra `notebook/benchmarking.ipynb` no VSCode, Cursor ou Jupyter. Selecione o kernel Python do ambiente onde as dependências foram instaladas e execute as células em ordem com **Run All** ou `Shift+Enter`.

```bash
jupyter notebook notebook/benchmarking.ipynb
```

---

## Métricas reportadas

- **MSE global** e **MAE global** — média sobre todas as janelas de teste, passos do horizonte e canais.
- **MSE / MAE por canal** — desempenho individual para eMBB, URLLC e mMTC.
