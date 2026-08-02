"""
레벨 간 cosine similarity vs 미래 수익률 예측력 분석
- 6개 레벨 → C(6,2)=15쌍 cosine similarity (스칼라)
- 윈도우별 평균 cosine sim → gap=20 forward return

Panel 1: 각 pair × window별 Pearson r 히트맵
Panel 2: 15개 similarity를 합친 모델 R² (Ridge vs MLP)
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

EMB_DIM  = 384
LEVELS   = ["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"]
DATA_DIR = "/home/eb/LG_AI/data/fintexts"
GAP      = 20
WINDOWS  = {"1d": 1, "5d": 5, "21d": 21, "42d": 42}

PAIRS    = list(combinations(LEVELS, 2))   # 15쌍
PAIR_LABELS = [f"{a[:4]}↔{b[:4]}" for a, b in PAIRS]

files  = sorted(glob.glob(os.path.join(DATA_DIR, "*_train.parquet")))
df_ref = pd.read_parquet(files[0]).sort_values("date").reset_index(drop=True)
dates  = pd.to_datetime(df_ref["date"])
T      = len(dates)

print(f"Loading {len(files)} tickers...")

def cos_sim(a, b):
    """[384], [384] → scalar. 둘 중 하나라도 zero면 0."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def window_mean_cos(arr_a, arr_b):
    """[w,384], [w,384] → valid 날만 cosine sim 평균 (스칼라)."""
    sims  = []
    for a, b in zip(arr_a, arr_b):
        s = cos_sim(a, b)
        if np.linalg.norm(a) > 1e-6 and np.linalg.norm(b) > 1e-6:
            sims.append(s)
    return np.mean(sims) if sims else 0.0

# ── 데이터 수집: X[wname] = [N, 15], y = [N] ─────────────────
X_cos = {w: [] for w in WINDOWS}
y_all = []

for fi, f in enumerate(files):
    if fi % 20 == 0:
        print(f"  {fi}/{len(files)}...")
    df = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df_a = df_ref[["date"]].merge(df, on="date", how="left")
    if "close" not in df_a.columns:
        continue
    close = df_a["close"].values.astype(np.float32)

    lv_arrs = {}
    for lv in LEVELS:
        cols  = [f"{lv}_emb{i}" for i in range(EMB_DIM)]
        avail = [c for c in cols if c in df_a.columns]
        lv_arrs[lv] = (
            df_a[avail].fillna(0).values.astype(np.float32)
            if avail else np.zeros((T, EMB_DIM), np.float32)
        )

    max_w = max(WINDOWS.values())
    for t in range(max_w, T - GAP):
        if np.isnan(close[t]) or np.isnan(close[t + GAP]) or close[t] < 1e-6:
            continue
        ret = close[t + GAP] / close[t] - 1.0
        if abs(ret) > 2.0:
            continue
        y_all.append(ret)
        for wname, w in WINDOWS.items():
            feats = [
                window_mean_cos(lv_arrs[a][t-w+1:t+1], lv_arrs[b][t-w+1:t+1])
                for a, b in PAIRS
            ]
            X_cos[wname].append(feats)

y_arr = np.array(y_all)
print(f"Total samples: {len(y_arr)}")
for w in WINDOWS:
    X_cos[w] = np.array(X_cos[w])   # [N, 15]

# ── Panel 1: 각 pair × window Pearson r ──────────────────────
wnames = list(WINDOWS.keys())
pearson_mat = np.zeros((len(PAIRS), len(wnames)))

print("\n=== Pearson r: pair × window ===")
for j, wname in enumerate(wnames):
    for i, (pair, label) in enumerate(zip(PAIRS, PAIR_LABELS)):
        x  = X_cos[wname][:, i]
        r, p = pearsonr(x, y_arr)
        pearson_mat[i, j] = r
    print(f"  {wname}: top |r| = "
          + str(sorted([(abs(pearson_mat[i,j]), PAIR_LABELS[i])
                         for i in range(len(PAIRS))], reverse=True)[:3]))

# ── Panel 2: 15개 합쳐서 Ridge/MLP R² ─────────────────────────
def compute_r2(X, y, model_type="linear"):
    valid  = ~np.isnan(X).any(axis=1)
    X, y   = X[valid], y[valid]
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    n_tr   = int(len(y) * 0.8)
    X_tr, X_te = X_s[:n_tr], X_s[n_tr:]
    y_tr, y_te = y[:n_tr], y[n_tr:]
    if model_type == "linear":
        model = Ridge(alpha=1.0)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu",
            max_iter=300, early_stopping=True,
            validation_fraction=0.1, random_state=42,
            learning_rate_init=1e-3,
        )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

print("\n=== Combined model R² (15 cosine sims) ===")
r2_ridge, r2_mlp = [], []
for wname in wnames:
    lin = compute_r2(X_cos[wname], y_arr, "linear")
    mlp = compute_r2(X_cos[wname], y_arr, "mlp")
    r2_ridge.append(lin)
    r2_mlp.append(mlp)
    print(f"  {wname}  Ridge={lin:.4f}  MLP={mlp:.4f}")

# ── 시각화 ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 8))
gs  = fig.add_gridspec(1, 2, wspace=0.35, width_ratios=[2.2, 1])

# Panel 1: Pearson r heatmap (15 pairs × 4 windows)
ax1  = fig.add_subplot(gs[0])
vmax = max(abs(pearson_mat).max(), 0.005)
im   = ax1.imshow(pearson_mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
plt.colorbar(im, ax=ax1, shrink=0.85, label="Pearson r")
ax1.set_xticks(range(len(wnames))); ax1.set_xticklabels(wnames, fontsize=10)
ax1.set_yticks(range(len(PAIRS)));  ax1.set_yticklabels(PAIR_LABELS, fontsize=9)
ax1.set_xlabel("Window Size", fontsize=10)
ax1.set_title("Pearson r: Inter-level Cosine Similarity vs Forward Return\n(per pair × window, gap=20)",
              fontsize=11, fontweight="bold")
for i in range(len(PAIRS)):
    for j in range(len(wnames)):
        val = pearson_mat[i, j]
        color = "white" if abs(val) > vmax * 0.6 else "black"
        ax1.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7.5, color=color)

# Panel 2: Combined R² bar
ax2 = fig.add_subplot(gs[1])
x   = np.arange(len(wnames))
w   = 0.35
ax2.bar(x - w/2, r2_ridge, width=w, label="Ridge", color="#377eb8", alpha=0.85, edgecolor="black", linewidth=0.5)
ax2.bar(x + w/2, r2_mlp,   width=w, label="MLP",   color="#e41a1c", alpha=0.85, edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(wnames, fontsize=10)
ax2.set_ylabel("Out-of-sample R²", fontsize=10)
ax2.set_title("Combined 15 Cosine Sims\n→ Return Prediction R²", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
for xi, (r, m) in enumerate(zip(r2_ridge, r2_mlp)):
    ax2.text(xi - w/2, r + 0.001, f"{r:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.text(xi + w/2, m + 0.001, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Inter-level Cosine Similarity as Price Predictor (gap=20)",
             fontsize=13, fontweight="bold")
out = "/home/eb/LG_AI/FinTexTS/cosine_sim_vs_price.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
