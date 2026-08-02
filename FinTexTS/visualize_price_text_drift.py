"""
텍스트 임베딩 Drift 예측력: Linear(Ridge) vs Non-linear(MLP)
drift = mean(late_half) - mean(early_half)  → [384] 벡터
윈도우 = drift 계산에 사용한 총 기간
  2d  drift: day t  vs day t-1
  10d drift: mean(t-4~t) - mean(t-9~t-5)
  21d drift: mean(t-10~t) - mean(t-20~t-11)
  42d drift: mean(t-20~t) - mean(t-41~t-21)
정답: close[t+20] / close[t] - 1
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EMB_DIM  = 384
LEVELS   = ["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"]
DATA_DIR = "/home/eb/LG_AI/data/fintexts"
GAP      = 20

# (total_window, late_size) 쌍
DRIFT_WINDOWS = {
    "2d":  (2,  1),
    "10d": (10, 5),
    "21d": (21, 11),
    "42d": (42, 21),
}

files  = sorted(glob.glob(os.path.join(DATA_DIR, "*_train.parquet")))
df_ref = pd.read_parquet(files[0]).sort_values("date").reset_index(drop=True)
dates  = pd.to_datetime(df_ref["date"])
T      = len(dates)

print(f"Loading {len(files)} tickers...")

def nonzero_mean(arr):
    mask = np.linalg.norm(arr, axis=1) > 1e-6
    return arr[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(EMB_DIM, dtype=np.float32)

def compute_drift(lv_arr, t, total_w, late_size):
    early_size = total_w - late_size
    early = lv_arr[max(0, t - total_w + 1): t - late_size + 1]   # [early_size, 384]
    late  = lv_arr[t - late_size + 1: t + 1]                      # [late_size,  384]
    return nonzero_mean(late) - nonzero_mean(early)                # [384]

# ── 데이터 수집 ───────────────────────────────────────────────
X_drift = {lv: {w: [] for w in DRIFT_WINDOWS} for lv in LEVELS}
y_all   = []

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

    max_w = max(tw for tw, _ in DRIFT_WINDOWS.values())
    for t in range(max_w, T - GAP):
        if np.isnan(close[t]) or np.isnan(close[t + GAP]) or close[t] < 1e-6:
            continue
        ret = close[t + GAP] / close[t] - 1.0
        if abs(ret) > 2.0:
            continue
        y_all.append(ret)
        for lv in LEVELS:
            for wname, (tw, ls) in DRIFT_WINDOWS.items():
                X_drift[lv][wname].append(compute_drift(lv_arrs[lv], t, tw, ls))

y_arr = np.array(y_all)
print(f"Total samples: {len(y_arr)}")

# ── R² 계산 ────────────────────────────────────────────────────
def compute_r2(X_list, y, model_type="linear"):
    X = np.array(X_list)
    valid = ~np.isnan(X).any(axis=1)
    X, y  = X[valid], y[valid]

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    pca    = PCA(n_components=min(50, X_s.shape[1]), random_state=42)
    X_pca  = pca.fit_transform(X_s)

    n_train = int(len(y) * 0.8)
    X_tr, X_te = X_pca[:n_train], X_pca[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    if model_type == "linear":
        model = Ridge(alpha=1.0)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            learning_rate_init=1e-3,
        )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

wnames = list(DRIFT_WINDOWS.keys())
r2 = {
    "Linear (Ridge)":   {lv: {} for lv in LEVELS},
    "Non-linear (MLP)": {lv: {} for lv in LEVELS},
}

print("\n=== Drift R² ===")
for lv in LEVELS:
    for wname in wnames:
        lin = compute_r2(X_drift[lv][wname], y_arr, "linear")
        mlp = compute_r2(X_drift[lv][wname], y_arr, "mlp")
        r2["Linear (Ridge)"][lv][wname]   = lin
        r2["Non-linear (MLP)"][lv][wname] = mlp
        print(f"  {lv:20s}  {wname:4s}  Ridge={lin:.4f}  MLP={mlp:.4f}")

# ── 히트맵 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.35)

for ax, method in zip(axes, r2.keys()):
    matrix = np.array([[r2[method][lv][w] for w in wnames] for lv in LEVELS])
    vmax   = max(abs(matrix).max(), 0.01)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.85, label="R²")
    ax.set_xticks(range(len(wnames))); ax.set_xticklabels(wnames, fontsize=10)
    ax.set_yticks(range(len(LEVELS))); ax.set_yticklabels(LEVELS, fontsize=10)
    ax.set_xlabel("Drift Window Size (total)", fontsize=10)
    ax.set_title(method, fontsize=12, fontweight="bold")
    for i, lv in enumerate(LEVELS):
        for j, w in enumerate(wnames):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

plt.suptitle("Embedding Drift Predictive Power: Linear vs Non-linear\n"
             "drift = mean(late_half) − mean(early_half), gap=20",
             fontsize=12, fontweight="bold")
out = "/home/eb/LG_AI/FinTexTS/price_text_drift.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
