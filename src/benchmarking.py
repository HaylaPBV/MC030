# =============================================================================
# PFG - Network Slicing 5G (LTSF) — Benchmarking
#
# Pipeline completo: pré-processamento → normalização → janelamento →
# treinamento (PatchTST + ablação + SSL, CNN1D, LSTM, DLinear, SARIMA,
# baselines ingênuos, Ensemble) → avaliação (MSE/MAE global e por canal).
#
# Saídas geradas automaticamente:
#   img/      — figuras das séries, split, janelas e predições por grupo
#   metrics/  — tabelas CSV com MSE/MAE global, por canal e estudo de ablação
#
# Execução (a partir da raiz do repositório):
#   python src/benchmarking.py
# =============================================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # backend sem janela — necessário fora do notebook
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import PatchTSTConfig, PatchTSTForPrediction, PatchTSTForPretraining

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------------------------------------------------------
# Pastas de saída
# -----------------------------------------------------------------------------
IMG_DIR     = "img"
METRICS_DIR = "metrics"
os.makedirs(IMG_DIR,     exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Caminhos
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "model_inputs.csv")


# =============================================================================
# Etapa 1 — Pré-processamento e Agregação
# =============================================================================

# 1.1 Leitura do CSV
df = pd.read_csv(DATA_PATH, index_col=0, low_memory=False)
print("Shape bruto:", df.shape)

# Verificações rápidas
print("Colunas:", list(df.columns))
print("\nValores únicos de Day:", df["Day (Input4)"].unique())
print("Valores únicos de Time:", sorted(df["Time (Input 5)"].unique()))
print("Valores únicos de Slice Type:", df["Slice Type (Output)"].unique())
print("\nNulos por coluna:")
print(df.isna().sum())

# 1.2 Construção de um índice temporal contínuo
# hour_index = day_of_week * 24 + (time - 1)  →  168 timestamps cobrindo 1 semana
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

df["day_of_week"] = df["Day (Input4)"].map(day_map)
df["hour_of_day"] = df["Time (Input 5)"].astype(int) - 1   # 0..23
df["hour_index"]  = df["day_of_week"] * 24 + df["hour_of_day"]

assert df["day_of_week"].notna().all(), "Algum nome de dia não foi mapeado."
assert df["hour_of_day"].between(0, 23).all(), "Hora fora do intervalo 0..23."
print("\nhour_index range:", df["hour_index"].min(), "->", df["hour_index"].max())

# 1.3 Agregação: contagem de requisições por hora e por slice
agg = (
    df.groupby(["hour_index", "Slice Type (Output)"])
      .size()
      .unstack(fill_value=0)
      .rename_axis(columns=None)
)

# Garante todas as 168 horas da semana presentes
full_index = pd.RangeIndex(start=0, stop=7 * 24, name="hour_index")
agg = agg.reindex(full_index, fill_value=0)

# Garante as três colunas mesmo se alguma faltar
for col in ["eMBB", "URLLC", "mMTC"]:
    if col not in agg.columns:
        agg[col] = 0
agg = agg[["eMBB", "URLLC", "mMTC"]].astype(np.int64)

print("\nShape agregado:", agg.shape)
print(agg.describe().round(2))

# 1.4 Visualização das três séries
fig, ax = plt.subplots(figsize=(14, 5))
agg.plot(ax=ax)
ax.set_title("Requisições por hora por Slice Type (1 semana)")
ax.set_xlabel("hour_index (0 = Seg 0h, 167 = Dom 23h)")
ax.set_ylabel("Nº de requisições")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "series_overview.png"), dpi=150)
plt.close()
print(f"\n[img] series_overview.png salvo.")


# =============================================================================
# Etapa 2 — Divisão Treino/Teste (Sequencial, 70/30)
# =============================================================================
TRAIN_RATIO = 0.70

n_total = len(agg)
n_train = int(np.floor(n_total * TRAIN_RATIO))
n_test  = n_total - n_train

train_df = agg.iloc[:n_train].copy()
test_df  = agg.iloc[n_train:].copy()

print(f"\nTotal de horas:  {n_total}")
print(f"Treino (70%):    {len(train_df)}  -> hour_index {train_df.index.min()}..{train_df.index.max()}")
print(f"Teste  (30%):    {len(test_df)}   -> hour_index {test_df.index.min()}..{test_df.index.max()}")

assert len(train_df) + len(test_df) == n_total
assert train_df.index.max() < test_df.index.min(), "Sobreposição temporal entre treino e teste!"

fig, ax = plt.subplots(figsize=(14, 5))
train_df.plot(ax=ax, alpha=0.9)
test_df.plot(ax=ax, linestyle="--", alpha=0.9)
ax.axvline(test_df.index.min(), color="k", linestyle=":", label="split treino/teste")
ax.set_title("Split sequencial 70/30")
ax.set_xlabel("hour_index")
ax.set_ylabel("Nº de requisições")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "train_test_split.png"), dpi=150)
plt.close()
print("[img] train_test_split.png salvo.")


# =============================================================================
# Etapa 3 — Normalização (Z-score)
# =============================================================================
# fit SOMENTE no treino para evitar data leakage
FEATURES = ["eMBB", "URLLC", "mMTC"]

scaler = StandardScaler()
scaler.fit(train_df[FEATURES].values)

train_scaled = scaler.transform(train_df[FEATURES].values)
test_scaled  = scaler.transform(test_df[FEATURES].values)

train_scaled_df = pd.DataFrame(train_scaled, index=train_df.index, columns=FEATURES)
test_scaled_df  = pd.DataFrame(test_scaled,  index=test_df.index,  columns=FEATURES)

print("\nMédias aprendidas (treino):", dict(zip(FEATURES, scaler.mean_.round(4))))
print("Desvios aprendidos (treino):", dict(zip(FEATURES, scaler.scale_.round(4))))
print("\nShape treino normalizado:", train_scaled.shape)
print("Shape teste  normalizado:", test_scaled.shape)

# Validação: no treino, média ~0 e desvio ~1 por coluna
print("\nTreino  -> média:", train_scaled.mean(axis=0).round(4),
      "| desvio:", train_scaled.std(axis=0).round(4))
print("Teste   -> média:", test_scaled.mean(axis=0).round(4),
      "| desvio:", test_scaled.std(axis=0).round(4),
      "   (não precisa ser 0/1 — esperado)")

recon = scaler.inverse_transform(train_scaled)
assert np.allclose(recon, train_df[FEATURES].values), "inverse_transform falhou no treino"
print("OK: inverse_transform reconstrói os valores originais.")


# =============================================================================
# Etapa 4 — Janelamento Deslizante (Sliding Windows)
# =============================================================================
def make_sliding_windows(series: np.ndarray, look_back: int, horizon: int, stride: int = 1):
    """
    Gera janelas deslizantes para previsão multivariada multi-step (DMS).

    Parâmetros
    ----------
    series : np.ndarray, shape (T, C)
        Série temporal já normalizada. T = nº de passos, C = nº de variáveis.
    look_back : int
        Tamanho da janela de histórico (L).
    horizon : int
        Tamanho do horizonte de previsão (H).
    stride : int, default=1
        Passo entre janelas consecutivas.

    Retorna
    -------
    X : np.ndarray, shape (N, L, C) — históricos.
    Y : np.ndarray, shape (N, H, C) — futuros (alvos).
    """
    if series.ndim != 2:
        raise ValueError(f"series deve ter shape (T, C); recebido {series.shape}")
    T, C = series.shape
    max_start = T - look_back - horizon + 1
    if max_start <= 0:
        raise ValueError(
            f"Série muito curta: T={T}, look_back={look_back}, horizon={horizon} "
            f"-> nenhuma janela possível."
        )
    starts = np.arange(0, max_start, stride)
    N = len(starts)
    X = np.empty((N, look_back, C), dtype=np.float32)
    Y = np.empty((N, horizon,   C), dtype=np.float32)
    for i, s in enumerate(starts):
        X[i] = series[s : s + look_back]
        Y[i] = series[s + look_back : s + look_back + horizon]
    return X, Y


# Hiperparâmetros globais (compartilhados pelos experimentos)
LOOK_BACK  = 48
STRIDE     = 1
BATCH_SIZE = 32
HORIZONS   = [12, 24]   # Teste 1 e Teste 2


def build_experiment(horizon: int,
                     train_series: np.ndarray,
                     test_series: np.ndarray,
                     look_back: int = LOOK_BACK,
                     stride: int = STRIDE,
                     batch_size: int = BATCH_SIZE) -> dict:
    """
    Monta janelas e DataLoaders para um dado horizonte de previsão.

    - Treino: janelas geradas apenas com `train_series`.
    - Teste:  prepende as últimas `look_back` horas do treino ao teste, para que
              a 1ª predição comece no primeiro passo do conjunto de teste.
              Não há leakage (esses pontos só entram como X, nunca como Y).
    """
    X_tr, Y_tr = make_sliding_windows(train_series, look_back, horizon, stride)

    test_input = np.concatenate([train_series[-look_back:], test_series], axis=0)
    X_te, Y_te = make_sliding_windows(test_input, look_back, horizon, stride)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    test_ds  = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(Y_te))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return {
        "horizon":      horizon,
        "look_back":    look_back,
        "X_train":      X_tr, "Y_train": Y_tr,
        "X_test":       X_te, "Y_test":  Y_te,
        "train_loader": train_loader,
        "test_loader":  test_loader,
    }


# Materializa os dois experimentos
EXPERIMENTS = {h: build_experiment(h, train_scaled, test_scaled) for h in HORIZONS}

for h, exp in EXPERIMENTS.items():
    print(f"\n[H={h:>2}] X_train={exp['X_train'].shape}  Y_train={exp['Y_train'].shape}"
          f"   |   X_test={exp['X_test'].shape}  Y_test={exp['Y_test'].shape}")

# Sanity check: inspeciona um batch de cada experimento
for h, exp in EXPERIMENTS.items():
    xb, yb = next(iter(exp["train_loader"]))
    print(f"[H={h:>2}] batch X: {tuple(xb.shape)} {xb.dtype}   "
          f"batch Y: {tuple(yb.shape)} {yb.dtype}")

# Visualização de uma janela (índice 0) de cada experimento — desnormalizada
sample_idx = 0
fig, axes = plt.subplots(len(FEATURES), len(EXPERIMENTS),
                         figsize=(7 * len(EXPERIMENTS), 7),
                         sharex="col")

for col, (h, exp) in enumerate(EXPERIMENTS.items()):
    L, H = exp["look_back"], exp["horizon"]
    x_sample = scaler.inverse_transform(exp["X_train"][sample_idx])  # (L, C)
    y_sample = scaler.inverse_transform(exp["Y_train"][sample_idx])  # (H, C)
    t_hist = np.arange(L)
    t_fut  = np.arange(L, L + H)

    for i, feat in enumerate(FEATURES):
        ax = axes[i, col] if len(EXPERIMENTS) > 1 else axes[i]
        ax.plot(t_hist, x_sample[:, i], label="histórico (X)", color="tab:blue")
        ax.plot(t_fut,  y_sample[:, i], label="futuro (Y)",    color="tab:orange")
        ax.axvline(L - 0.5, color="k", linestyle=":")
        if col == 0:
            ax.set_ylabel(feat)
        ax.grid(True, alpha=0.3)

    top_ax = axes[0, col] if len(EXPERIMENTS) > 1 else axes[0]
    top_ax.set_title(f"Teste {col+1}: look_back={L}, horizon={H}")
    bot_ax = axes[-1, col] if len(EXPERIMENTS) > 1 else axes[-1]
    bot_ax.set_xlabel("passo (h)")

(axes[0, 0] if len(EXPERIMENTS) > 1 else axes[0]).legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "sliding_windows_sample.png"), dpi=150)
plt.close()
print("[img] sliding_windows_sample.png salvo.")


# =============================================================================
# Etapa 5 — Modelagem
# =============================================================================

# Seed e device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CHANNELS = len(FEATURES)
print(f"\nDevice: {DEVICE}")


# ---------------------------------------------------------------------------
# 5.1 PatchTST — wrapper parametrizável para o Estudo de Ablação
# ---------------------------------------------------------------------------
class PatchTST(nn.Module):
    """Wrapper sobre PatchTSTForPrediction parametrizável para o Estudo de Ablação.

    Os hiperparâmetros que afetam capacidade (`d_model`, `n_layers`, `n_heads`,
    `dropout`), normalização interna (`scaling` — RevIN quando `'std'`) e o
    Positional Encoding temporal (`positional_encoding_type`) são expostos como
    argumentos, permitindo testar uma variação por vez sem reescrever a rede.

    Forward: (B, L, C) -> (B, H, C).
    """
    def __init__(self, look_back: int, horizon: int, num_channels: int,
                 patch_length: int = 8, patch_stride: int = 8,
                 # --- capacidade arquitetural (Ablação A) ---
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ffn_dim: int = 128, dropout: float = 0.1,
                 # None = sem RevIN (assume Z-score externo); 'std' = ativa RevIN
                 scaling=None,
                 # None = default do HF; 'sincos'/'random'/'no' força explicitamente
                 positional_encoding_type: str = None):
        super().__init__()
        cfg_kwargs = dict(
            num_input_channels      = num_channels,
            context_length          = look_back,
            prediction_length       = horizon,
            patch_length            = patch_length,
            patch_stride            = patch_stride,
            d_model                 = d_model,
            num_attention_heads     = n_heads,
            num_hidden_layers       = n_layers,
            ffn_dim                 = ffn_dim,
            dropout                 = dropout,
            head_dropout            = dropout,
            scaling                 = scaling,
            do_mask_input           = False,   # treino supervisionado direto (DMS)
        )
        if positional_encoding_type is not None:
            cfg_kwargs["positional_encoding_type"] = positional_encoding_type
        self.config = PatchTSTConfig(**cfg_kwargs)
        self.model  = PatchTSTForPrediction(self.config)

    def forward(self, x):  # x: (B, L, C)
        out = self.model(past_values=x)
        return out.prediction_outputs  # (B, H, C)


# ---------------------------------------------------------------------------
# 5.2 Loop de treino genérico
# ---------------------------------------------------------------------------
def train_model(model: nn.Module, train_loader, num_epochs: int = 100,
                lr: float = 1e-3, device=DEVICE, verbose_every: int = 20) -> nn.Module:
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(1, num_epochs + 1):
        total, n = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0); n += xb.size(0)
        if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == num_epochs):
            print(f"  epoch {epoch:>3}/{num_epochs}  train MSE (norm): {total/n:.4f}")
    return model


@torch.no_grad()
def predict(model: nn.Module, loader, device=DEVICE):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        p = model(xb).cpu().numpy()
        preds.append(p); trues.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


# ---------------------------------------------------------------------------
# 5.3 Baseline: CNN 1D channel-independent
# ---------------------------------------------------------------------------
class CNN1D(nn.Module):
    """CNN 1D channel-independent para previsão multi-step direta (DMS).

    Forward: (B, L, C) -> (B, H, C).
    Cada canal é processado independentemente com convoluções 1D compartilhadas,
    e uma cabeça linear projeta o histórico para o horizonte completo.
    """
    def __init__(self, look_back: int, horizon: int, num_channels: int,
                 hidden_channels: int = 32, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.look_back    = look_back
        self.horizon      = horizon
        self.num_channels = num_channels

        pad = kernel_size // 2  # 'same' padding (kernel ímpar)
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_channels * look_back, horizon)

    def forward(self, x):  # x: (B, L, C)
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, 1, L)   # (B*C, 1, L)
        x = self.conv(x)                               # (B*C, hidden, L)
        x = x.reshape(B * C, -1)                       # (B*C, hidden*L)
        x = self.head(x)                               # (B*C, H)
        x = x.reshape(B, C, self.horizon).permute(0, 2, 1)  # (B, H, C)
        return x


# ---------------------------------------------------------------------------
# 5.3.1 LSTM e DLinear
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """LSTM multivariada para previsão multi-step direta (DMS).

    Forward: (B, L, C) -> (B, H, C).
    Estratégia: empilha 2 camadas LSTM com input_size=C (multivariada conjunta).
    A saída do ÚLTIMO passo de tempo (h_t do topo) é projetada por uma única
    camada linear para H*C, e remodelada para (B, H, C).
    """
    def __init__(self, look_back: int, horizon: int, num_channels: int,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.horizon      = horizon
        self.num_channels = num_channels
        self.lstm = nn.LSTM(
            input_size  = num_channels,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon * num_channels)

    def forward(self, x):                          # x: (B, L, C)
        out, _ = self.lstm(x)                      # (B, L, hidden)
        last   = out[:, -1, :]                     # (B, hidden) — último passo
        proj   = self.head(last)                   # (B, H*C)
        return proj.view(-1, self.horizon, self.num_channels)  # (B, H, C)


class _MovingAvg(nn.Module):
    """Média móvel 1D com padding por replicação dos extremos para preservar L.

    Forward: (B, L, C) -> (B, L, C). Saída é a TENDÊNCIA da série.
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size deve ser ímpar para padding simétrico.")
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):                           # x: (B, L, C)
        front = x[:, :1, :].repeat(1, self.pad, 1)  # replica o 1º passo
        end   = x[:, -1:, :].repeat(1, self.pad, 1) # replica o último passo
        x = torch.cat([front, x, end], dim=1)       # (B, L+2*pad, C)
        x = x.permute(0, 2, 1)                      # AvgPool1d espera (B, C, L)
        x = self.avg(x)                             # (B, C, L)
        return x.permute(0, 2, 1)                   # (B, L, C)


class DLinear(nn.Module):
    """DLinear (Zeng et al., 2022): decomposição por média móvel + 2 lineares.

    Forward: (B, L, C) -> (B, H, C).
    Pipeline:
      1. trend     = MovingAvg(x);  remainder = x - trend
      2. Linear(L, H) channel-independent para cada componente
      3. saída = Linear_trend(trend) + Linear_remainder(remainder)
    """
    def __init__(self, look_back: int, horizon: int, num_channels: int,
                 kernel_size: int = 25):
        super().__init__()
        self.decomp           = _MovingAvg(kernel_size=kernel_size)
        self.linear_trend     = nn.Linear(look_back, horizon)
        self.linear_remainder = nn.Linear(look_back, horizon)

    def forward(self, x):                                   # x: (B, L, C)
        trend     = self.decomp(x)                           # (B, L, C)
        remainder = x - trend                                # (B, L, C)
        trend     = trend.permute(0, 2, 1)                   # (B, C, L)
        remainder = remainder.permute(0, 2, 1)               # (B, C, L)
        out = self.linear_trend(trend) + self.linear_remainder(remainder)  # (B, C, H)
        return out.permute(0, 2, 1)                          # (B, H, C)


# ---------------------------------------------------------------------------
# 5.4 Estudo de Ablação do PatchTST + treino dos baselines neurais
# ---------------------------------------------------------------------------
NUM_EPOCHS = 100
LR         = 1e-3

# Cada entrada acumula UMA mudança nova em relação à anterior para isolar
# o efeito de cada técnica no MSE/MAE:
#   1) PatchTST_Original  → baseline com hiperparâmetros do paper
#   2) PatchTST_A_small   → + Ablação A: redução drástica de capacidade
#                             (d_model=16, n_layers=1, n_heads=2, dropout=0.3)
#   3) PatchTST_AB_revin  → + Ablação B: ativa RevIN (scaling='std')
#   4) PatchTST_ABC_sincos→ + Ablação C: força positional_encoding_type='sincos'
PATCHTST_ABLATIONS = {
    "PatchTST_Original": dict(
        d_model=64, n_heads=4, n_layers=2, dropout=0.1,
        scaling=None, positional_encoding_type=None,
    ),
    "PatchTST_A_small": dict(
        d_model=16, n_heads=2, n_layers=1, dropout=0.3,
        scaling=None, positional_encoding_type=None,
    ),
    "PatchTST_AB_revin": dict(
        d_model=16, n_heads=2, n_layers=1, dropout=0.3,
        scaling="std", positional_encoding_type=None,
    ),
    "PatchTST_ABC_sincos": dict(
        d_model=16, n_heads=2, n_layers=1, dropout=0.3,
        scaling="std", positional_encoding_type="sincos",
    ),
}

NEURAL_BASELINES = {
    "CNN1D":   lambda L, H: CNN1D(look_back=L, horizon=H, num_channels=NUM_CHANNELS),
    "LSTM":    lambda L, H: LSTMModel(look_back=L, horizon=H, num_channels=NUM_CHANNELS,
                                      hidden_size=64, num_layers=2, dropout=0.1),
    "DLinear": lambda L, H: DLinear(look_back=L, horizon=H, num_channels=NUM_CHANNELS,
                                    kernel_size=25),
}

RESULTS = {}  # (model_name, horizon) -> dict(model, preds_scaled, trues_scaled, [config])

# --- 1) Treina cada variante do PatchTST (Estudo de Ablação) ---------------
for ablation_name, cfg in PATCHTST_ABLATIONS.items():
    for h, exp in EXPERIMENTS.items():
        key = (ablation_name, h)
        print(f"\n=== {ablation_name}  |  look_back={exp['look_back']}  horizon={h} ===")
        print(f"  config: {cfg}")
        torch.manual_seed(SEED)  # mesma init para cada run → comparação justa
        model = PatchTST(
            look_back    = exp["look_back"],
            horizon      = h,
            num_channels = NUM_CHANNELS,
            **cfg,
        )
        model = train_model(model, exp["train_loader"],
                            num_epochs=NUM_EPOCHS, lr=LR, verbose_every=20)
        preds, trues = predict(model, exp["test_loader"])
        RESULTS[key] = {
            "model":        model,
            "preds_scaled": preds,    # (N, H, C) escala normalizada
            "trues_scaled": trues,    # (N, H, C) escala normalizada
            "config":       cfg,
        }
        print(f"  -> preds shape: {preds.shape}")

# --- 2) Treina baselines neurais (CNN1D, LSTM, DLinear) --------------------
for model_name, factory in NEURAL_BASELINES.items():
    for h, exp in EXPERIMENTS.items():
        key = (model_name, h)
        print(f"\n=== Treinando {model_name}  |  look_back={exp['look_back']}  horizon={h} ===")
        torch.manual_seed(SEED)
        model = factory(exp["look_back"], h)
        model = train_model(model, exp["train_loader"],
                            num_epochs=NUM_EPOCHS, lr=LR, verbose_every=20)
        preds, trues = predict(model, exp["test_loader"])
        RESULTS[key] = {
            "model":        model,
            "preds_scaled": preds,
            "trues_scaled": trues,
        }
        print(f"  -> preds shape: {preds.shape}")


# ---------------------------------------------------------------------------
# 5.4.1 PatchTST + SSL (pré-treino auto-supervisionado)
# ---------------------------------------------------------------------------
def train_patchtst_ssl(train_loader,
                       look_back: int,
                       horizon: int,
                       num_channels: int = NUM_CHANNELS,
                       d_model: int = 16, n_heads: int = 2, n_layers: int = 1,
                       ffn_dim: int = 128, dropout: float = 0.3,
                       patch_length: int = 8, patch_stride: int = 8,
                       random_mask_ratio: float = 0.5,
                       pretrain_epochs: int = 50,
                       finetune_epochs: int = NUM_EPOCHS,
                       lr: float = LR,
                       device=DEVICE,
                       seed: int = SEED) -> nn.Module:
    """Pré-treino auto-supervisionado + fine-tuning do PatchTST.

    Etapa A — Pretraining: PatchTSTForPretraining com do_mask_input=True,
              mask_type='random', random_mask_ratio=0.5. Loss interna do HF
              é a reconstrução dos patches mascarados.
    Etapa B — Fine-tuning: cria PatchTSTForPrediction (do_mask_input=False)
              e transfere o state_dict do BACKBONE pré-treinado. Treina
              supervisionado com MSE no horizonte usando train_model().

    Retorna um nn.Module com a interface (B, L, C) -> (B, H, C),
    compatível com a função predict() existente.
    """
    common_cfg = dict(
        num_input_channels       = num_channels,
        context_length           = look_back,
        prediction_length        = horizon,
        patch_length             = patch_length,
        patch_stride             = patch_stride,
        d_model                  = d_model,
        num_attention_heads      = n_heads,
        num_hidden_layers        = n_layers,
        ffn_dim                  = ffn_dim,
        dropout                  = dropout,
        head_dropout             = dropout,
        scaling                  = "std",
        positional_encoding_type = "sincos",
    )

    # ---------- Etapa A: Pretraining (mascaramento aleatório) ---------------
    torch.manual_seed(seed)
    pretrain_cfg = PatchTSTConfig(
        **common_cfg,
        do_mask_input     = True,
        mask_type         = "random",
        random_mask_ratio = random_mask_ratio,
    )
    pretrain_model = PatchTSTForPretraining(pretrain_cfg).to(device)
    optim = torch.optim.Adam(pretrain_model.parameters(), lr=lr)
    pretrain_model.train()

    print(f"\n--- SSL Pretraining ({pretrain_epochs} épocas, mask_ratio={random_mask_ratio}) ---")
    for epoch in range(1, pretrain_epochs + 1):
        total, n = 0.0, 0
        for xb, _ in train_loader:          # ignora o Y; é auto-supervisionado
            xb = xb.to(device)
            optim.zero_grad()
            out = pretrain_model(past_values=xb)
            loss = out.loss
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0); n += xb.size(0)
        if epoch == 1 or epoch % 10 == 0 or epoch == pretrain_epochs:
            print(f"  pre-epoch {epoch:>3}/{pretrain_epochs}  recon MSE: {total/n:.4f}")

    # ---------- Etapa B: Fine-tuning supervisionado -------------------------
    torch.manual_seed(seed)
    finetune_cfg = PatchTSTConfig(**common_cfg, do_mask_input=False)
    finetune_model = PatchTSTForPrediction(finetune_cfg).to(device)

    # Transferência de pesos do BACKBONE (encoder + projeções de patches).
    # A head específica de cada modelo (reconstrução vs predição) NÃO é transferida.
    backbone_state = pretrain_model.model.state_dict()
    info = finetune_model.model.load_state_dict(backbone_state, strict=False)
    print(f"  -> backbone transferido | missing={len(info.missing_keys)}  "
          f"unexpected={len(info.unexpected_keys)}")

    class _PatchTSTSSLWrapper(nn.Module):
        def __init__(self, hf_model):
            super().__init__()
            self.model = hf_model
        def forward(self, x):
            return self.model(past_values=x).prediction_outputs

    wrapped = _PatchTSTSSLWrapper(finetune_model)

    print(f"\n--- SSL Fine-tuning ({finetune_epochs} épocas, MSE no horizonte) ---")
    wrapped = train_model(wrapped, train_loader,
                          num_epochs=finetune_epochs, lr=lr,
                          device=device, verbose_every=20)
    return wrapped


for h, exp in EXPERIMENTS.items():
    key = ("PatchTST_SSL", h)
    print(f"\n=== {key[0]}  |  look_back={exp['look_back']}  horizon={h} ===")
    model = train_patchtst_ssl(
        train_loader    = exp["train_loader"],
        look_back       = exp["look_back"],
        horizon         = h,
        pretrain_epochs = 50,
        finetune_epochs = NUM_EPOCHS,
    )
    preds, trues = predict(model, exp["test_loader"])
    RESULTS[key] = {
        "model":        model,
        "preds_scaled": preds,
        "trues_scaled": trues,
    }
    print(f"  -> preds shape: {preds.shape}")


# ---------------------------------------------------------------------------
# 5.4.2 Ensemble simples (CNN1D + LSTM)
# ---------------------------------------------------------------------------
# Não treina nada — combina via média aritmética as predições já em RESULTS.
# Como o StandardScaler é linear, fazer o ensemble no espaço Z-score é seguro.
for h in HORIZONS:
    cnn_key, lstm_key = ("CNN1D", h), ("LSTM", h)
    if cnn_key not in RESULTS or lstm_key not in RESULTS:
        print(f"  Ensemble H={h} pulado: {cnn_key} ou {lstm_key} ausente em RESULTS.")
        continue
    cnn_preds  = RESULTS[cnn_key]["preds_scaled"]
    lstm_preds = RESULTS[lstm_key]["preds_scaled"]
    trues      = RESULTS[cnn_key]["trues_scaled"]

    ensemble_preds = ((cnn_preds + lstm_preds) / 2.0).astype(np.float32)
    RESULTS[("Ensemble_CNN_LSTM", h)] = {
        "model":        None,
        "preds_scaled": ensemble_preds,
        "trues_scaled": trues,
    }
    print(f"  Ensemble_CNN_LSTM H={h:>2}  ->  preds shape: {ensemble_preds.shape}")


# ---------------------------------------------------------------------------
# 5.5 Baseline: SARIMA (estatístico, univariado por canal, sazonalidade diária)
# ---------------------------------------------------------------------------
# Seleção de ordem SARIMA (p,d,q)(P,D,Q,m) por canal usando auto_arima
# no treino. m=24 = período sazonal diário (1 dia = 24h). D=1 força a
# diferenciação sazonal — necessária para capturar o ciclo diário.
SARIMA_ORDERS = {}
for c, feat in enumerate(FEATURES):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = pm.auto_arima(
            train_scaled[:, c],
            seasonal=True,
            m=24,
            D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
        )
    SARIMA_ORDERS[feat] = {
        "order":          auto.order,
        "seasonal_order": auto.seasonal_order,
    }
    print(f"  {feat:>5}: order={auto.order}  seasonal_order={auto.seasonal_order}")


def sarima_forecast_windows(X_test: np.ndarray, horizon: int,
                            orders: dict, feature_names: list) -> np.ndarray:
    """
    Para cada janela do teste e cada canal, ajusta SARIMAX com a ordem
    (p,d,q)(P,D,Q,m) pré-selecionada usando o histórico da janela e prevê
    `horizon` passos à frente.

    X_test : (N, L, C) — históricos das janelas de teste (escala normalizada).
    Retorna preds : (N, H, C) na escala normalizada.
    """
    N, L, C = X_test.shape
    preds = np.zeros((N, horizon, C), dtype=np.float32)
    for c, feat in enumerate(feature_names):
        order          = orders[feat]["order"]
        seasonal_order = orders[feat]["seasonal_order"]
        for i in range(N):
            history = X_test[i, :, c]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(history,
                                  order=order,
                                  seasonal_order=seasonal_order,
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit(disp=False)
                fc = res.forecast(steps=horizon)
            except Exception:
                # fallback: repete o último valor (naive) se SARIMAX falhar
                fc = np.repeat(history[-1], horizon)
            preds[i, :, c] = np.asarray(fc, dtype=np.float32)
    return preds


for h, exp in EXPERIMENTS.items():
    key = ("SARIMA", h)
    print(f"\n=== SARIMA  |  look_back={exp['look_back']}  horizon={h} ===")
    preds = sarima_forecast_windows(exp["X_test"], h, SARIMA_ORDERS, FEATURES)
    RESULTS[key] = {
        "model":        None,
        "preds_scaled": preds,
        "trues_scaled": exp["Y_test"],
    }
    print(f"  -> preds shape: {preds.shape}")


# ---------------------------------------------------------------------------
# 5.6 Baselines triviais (Naive e Seasonal-Naive)
# ---------------------------------------------------------------------------
SEASONAL_PERIOD = 24  # ciclo diário (horas)


def naive_forecast(X_test: np.ndarray, horizon: int) -> np.ndarray:
    """Repete o último valor da janela ao longo de todo o horizonte. (N, L, C) -> (N, H, C)."""
    last = X_test[:, -1:, :]
    return np.repeat(last, horizon, axis=1)


def seasonal_naive_forecast(X_test: np.ndarray, horizon: int,
                            period: int = SEASONAL_PERIOD) -> np.ndarray:
    """
    Para cada passo t+k do horizonte, retorna o valor observado em t+k-period.
    Requer look_back >= period.
    """
    N, L, C = X_test.shape
    if L < period:
        raise ValueError(f"look_back={L} < period={period}; SNaive não aplicável.")
    preds = np.empty((N, horizon, C), dtype=X_test.dtype)
    for k in range(horizon):
        idx = L - period + k
        if idx >= L:
            idx = L - period + (k % period)
        preds[:, k, :] = X_test[:, idx, :]
    return preds


for h, exp in EXPERIMENTS.items():
    for name, fn in [("Naive", naive_forecast), ("SNaive", seasonal_naive_forecast)]:
        preds = fn(exp["X_test"], h)
        RESULTS[(name, h)] = {
            "model":        None,
            "preds_scaled": preds.astype(np.float32),
            "trues_scaled": exp["Y_test"],
        }
        print(f"  {name:>7}  H={h:>2}  -> preds shape: {preds.shape}")


# =============================================================================
# Etapa 6 — Avaliação (desnormalização + MSE/MAE)
# =============================================================================
def inverse_scale(arr_scaled: np.ndarray) -> np.ndarray:
    """Aplica scaler.inverse_transform a um array (N, H, C) preservando o shape."""
    N, H, C = arr_scaled.shape
    flat = arr_scaled.reshape(-1, C)
    return scaler.inverse_transform(flat).reshape(N, H, C)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, feature_names: list) -> dict:
    """Calcula MSE/MAE global e por canal. Inputs em (N, H, C) na escala real."""
    err  = y_pred - y_true
    mse_global = float(np.mean(err ** 2))
    mae_global = float(np.mean(np.abs(err)))
    mse_per_channel = {feat: float(np.mean(err[..., c] ** 2))
                       for c, feat in enumerate(feature_names)}
    mae_per_channel = {feat: float(np.mean(np.abs(err[..., c])))
                       for c, feat in enumerate(feature_names)}
    return {
        "MSE_global": mse_global,
        "MAE_global": mae_global,
        "MSE_per_channel": mse_per_channel,
        "MAE_per_channel": mae_per_channel,
    }


# Desnormaliza e calcula métricas para todos os runs
for key, res in RESULTS.items():
    res["preds_real"] = inverse_scale(res["preds_scaled"])
    res["trues_real"] = inverse_scale(res["trues_scaled"])
    res["metrics"]    = compute_metrics(res["trues_real"], res["preds_real"], FEATURES)

print("\nRuns avaliados:", list(RESULTS.keys()))

# --- Tabela 1: métricas globais (MSE, MAE) por modelo × horizonte ----------
rows = []
for (model_name, h), res in RESULTS.items():
    m = res["metrics"]
    rows.append({"Modelo": model_name, "Horizon": h,
                 "MSE": m["MSE_global"], "MAE": m["MAE_global"]})

metrics_global = (pd.DataFrame(rows)
                    .sort_values(["Horizon", "MSE"])
                    .reset_index(drop=True))
print("\nMétricas globais (escala real):")
print(metrics_global.round(4).to_string(index=False))
metrics_global.to_csv(os.path.join(METRICS_DIR, "metrics_global.csv"), index=False)
print(f"[metrics] metrics_global.csv salvo.")

# --- Tabela 2: métricas por canal (MSE e MAE) ------------------------------
rows = []
for (model_name, h), res in RESULTS.items():
    m = res["metrics"]
    row = {"Modelo": model_name, "Horizon": h}
    for feat in FEATURES:
        row[f"MSE_{feat}"] = m["MSE_per_channel"][feat]
        row[f"MAE_{feat}"] = m["MAE_per_channel"][feat]
    rows.append(row)

metrics_per_channel = (pd.DataFrame(rows)
                         .sort_values(["Horizon", "Modelo"])
                         .reset_index(drop=True))
print("\nMétricas por canal (escala real):")
print(metrics_per_channel.round(4).to_string(index=False))
metrics_per_channel.to_csv(os.path.join(METRICS_DIR, "metrics_per_channel.csv"), index=False)
print(f"[metrics] metrics_per_channel.csv salvo.")

# --- Visualização: predição vs verdadeiro para a 1ª janela do teste --------
# Separamos em três grupos para não gerar grades largas demais:
#   1) Estudo de Ablação do PatchTST
#   2) Baselines neurais
#   3) Baselines estatísticos e ingênuos
sample_idx = 0
horizons_sorted = sorted({h for _, h in RESULTS.keys()})

ABLATION_GROUP           = ["PatchTST_Original", "PatchTST_A_small",
                             "PatchTST_AB_revin", "PatchTST_ABC_sincos", "PatchTST_SSL"]
NEURAL_BASELINES_GROUP   = ["CNN1D", "LSTM", "DLinear", "Ensemble_CNN_LSTM"]
STATISTICAL_BASELINES_GROUP = ["SARIMA", "Naive", "SNaive"]


def _plot_group(group_title: str, model_list: list, h: int, filename: str):
    names = [m for m in model_list if (m, h) in RESULTS]
    if not names:
        return
    n_cols = len(names)
    fig, axes = plt.subplots(len(FEATURES), n_cols,
                             figsize=(5 * n_cols, 7),
                             sharex=True, sharey="row",
                             squeeze=False)
    fig.suptitle(f"{group_title}  |  H={h}", fontsize=13, y=1.02)

    for col, model_name in enumerate(names):
        res    = RESULTS[(model_name, h)]
        y_true = res["trues_real"][sample_idx]   # (H, C)
        y_pred = res["preds_real"][sample_idx]   # (H, C)
        t = np.arange(h)
        for i, feat in enumerate(FEATURES):
            ax = axes[i, col]
            ax.plot(t, y_true[:, i], label="real",    color="tab:blue",   marker="o", ms=3)
            ax.plot(t, y_pred[:, i], label="predito", color="tab:orange", marker="x", ms=4)
            if col == 0:
                ax.set_ylabel(feat)
            ax.grid(True, alpha=0.3)
        axes[0, col].set_title(model_name)

    for ax in axes[-1, :]:
        ax.set_xlabel("passo (h)")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[img] {filename} salvo.")


for h in horizons_sorted:
    _plot_group("Estudo de Ablação do PatchTST", ABLATION_GROUP,
                h, f"ablation_H{h}.png")
    _plot_group("Baselines neurais", NEURAL_BASELINES_GROUP,
                h, f"neural_baselines_H{h}.png")
    _plot_group("Baselines estatístico e ingênuos", STATISTICAL_BASELINES_GROUP,
                h, f"statistical_baselines_H{h}.png")


# =============================================================================
# Etapa 6.1 — Tabela do Estudo de Ablação (PatchTST)
# =============================================================================
ABLATION_ORDER = [
    "PatchTST_Original",    # 1) baseline (alta capacidade, sem RevIN)
    "PatchTST_A_small",     # 2) + Ablação A
    "PatchTST_AB_revin",    # 3) + Ablação A + B
    "PatchTST_ABC_sincos",  # 4) + Ablação A + B + C
    "PatchTST_SSL",         # 5) + SSL (pré-treino mascarado + fine-tuning)
]
BASELINE_MODELS = ["SARIMA", "CNN1D", "LSTM", "DLinear",
                   "Ensemble_CNN_LSTM", "Naive", "SNaive"]

rows = []
for h in sorted({hh for _, hh in RESULTS.keys()}):
    prev_mse, prev_mae = None, None
    for name in ABLATION_ORDER:
        if (name, h) not in RESULTS:
            continue
        m = RESULTS[(name, h)]["metrics"]
        mse, mae = m["MSE_global"], m["MAE_global"]
        rows.append({
            "Grupo":   "Ablação PatchTST",
            "Modelo":  name,
            "Horizon": h,
            "MSE":     mse,
            "MAE":     mae,
            "ΔMSE_vs_anterior": None if prev_mse is None else mse - prev_mse,
            "ΔMAE_vs_anterior": None if prev_mae is None else mae - prev_mae,
        })
        prev_mse, prev_mae = mse, mae

    for name in BASELINE_MODELS:
        if (name, h) not in RESULTS:
            continue
        m = RESULTS[(name, h)]["metrics"]
        rows.append({
            "Grupo":   "Baseline",
            "Modelo":  name,
            "Horizon": h,
            "MSE":     m["MSE_global"],
            "MAE":     m["MAE_global"],
            "ΔMSE_vs_anterior": None,
            "ΔMAE_vs_anterior": None,
        })

ablation_table = pd.DataFrame(rows)
print("\nEstudo de Ablação — MSE/MAE globais (escala real):")
print(ablation_table.round(4).to_string(index=False))
ablation_table.to_csv(os.path.join(METRICS_DIR, "ablation_study.csv"), index=False)
print(f"[metrics] ablation_study.csv salvo.")

print("\n✓ Benchmarking concluído.")
print(f"  Imagens : {IMG_DIR}/")
print(f"  Métricas: {METRICS_DIR}/")
