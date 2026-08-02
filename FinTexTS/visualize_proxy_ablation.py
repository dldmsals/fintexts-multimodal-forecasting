"""
Proxy Ablation Study
1. 레벨 조합 ablation: 어느 레벨의 proxy를 쓰는게 최적인가
2. Proxy 선택 ablation: 어느 proxy 조합이 최적인가 (macro+sector 기준)
방법: Ridge + MLP out-of-sample R² (80/20 time split)
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

EMB_DIM  = 384
LEVELS   = ["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"]
DATA_DIR = "/home/eb/LG_AI/data/fintexts"
GAP      = 20
MAX_W    = 21

PROXIES = [
    "magnitude", "shock_1d", "shock_5d",
    "novelty_5d", "novelty_21d", "persistence",
    "cum_shock_5d", "sparsity_21d", "volatility_21d",
    "macro_align", "days_since",
]

files  = sorted(glob.glob(os.path.join(DATA_DIR, "*_train.parquet")))
df_ref = pd.read_parquet(files[0]).sort_values("date").reset_index(drop=True)
T      = len(df_ref)
print(f"Loading {len(files)} tickers...")

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-6 and nb > 1e-6 else 0.0

def nonzero_mean(arr):
    mask = np.linalg.norm(arr, axis=1) > 1e-6
    return arr[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(EMB_DIM, np.float32)

# ── 데이터 수집 ───────────────────────────────────────────────
proxy_vals = {lv: {p: [] for p in PROXIES} for lv in LEVELS}
returns    = []

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

    for t in range(MAX_W + 1, T - GAP):
        if np.isnan(close[t]) or np.isnan(close[t + GAP]) or close[t] < 1e-6:
            continue
        if abs(close[t + GAP] / close[t] - 1.0) > 2.0:
            continue
        returns.append(close[t + GAP] / close[t] - 1.0)

        mac = lv_arrs["macro"][t]
        for lv in LEVELS:
            e = lv_arrs[lv]
            e_t, e_t1, e_t5 = e[t], e[t-1], e[t-5]
            win5  = e[t-5:t]
            win21 = e[t-MAX_W:t+1]
            norms21 = np.linalg.norm(win21, axis=1)

            proxy_vals[lv]["magnitude"].append(np.linalg.norm(e_t))
            proxy_vals[lv]["shock_1d"].append(np.linalg.norm(e_t - e_t1))
            proxy_vals[lv]["shock_5d"].append(np.linalg.norm(e_t - e_t5))
            proxy_vals[lv]["novelty_5d"].append(np.linalg.norm(e_t - nonzero_mean(win5)))
            proxy_vals[lv]["novelty_21d"].append(np.linalg.norm(e_t - nonzero_mean(win21)))
            proxy_vals[lv]["persistence"].append(cos_sim(e_t, e_t1))
            proxy_vals[lv]["cum_shock_5d"].append(
                sum(np.linalg.norm(e[t-k] - e[t-k-1]) for k in range(5)))
            proxy_vals[lv]["sparsity_21d"].append((norms21 > 1e-6).mean())
            proxy_vals[lv]["volatility_21d"].append(norms21.std())
            proxy_vals[lv]["macro_align"].append(cos_sim(e_t, mac))
            shocks_h = [np.linalg.norm(e[t-k] - e[t-k-1]) for k in range(MAX_W)]
            thresh = np.mean(shocks_h) + np.std(shocks_h)
            proxy_vals[lv]["days_since"].append(
                next((k for k, s in enumerate(shocks_h) if s > thresh), MAX_W))

y = np.array(returns)
print(f"Total samples: {len(y)}")

# ── R² 계산 ───────────────────────────────────────────────────
def compute_r2(X, y, model_type="mlp"):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    valid = ~np.isnan(X).any(axis=1)
    X, y_ = X[valid], y[valid]
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    n_tr   = int(len(y_) * 0.8)
    X_tr, X_te = X_s[:n_tr], X_s[n_tr:]
    y_tr, y_te = y_[:n_tr], y_[n_tr:]
    if model_type == "linear":
        m = Ridge(alpha=1.0)
    else:
        m = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                         max_iter=300, early_stopping=True,
                         validation_fraction=0.1, random_state=42,
                         learning_rate_init=1e-3)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    ss_r = np.sum((y_te - pred) ** 2)
    ss_t = np.sum((y_te - y_te.mean()) ** 2)
    return 1 - ss_r / ss_t if ss_t > 0 else 0.0

def get_X(level_list, proxy_list):
    cols = []
    for lv in level_list:
        for px in proxy_list:
            cols.append(proxy_vals[lv][px])
    return np.column_stack(cols)

# ── 1. 레벨 조합 ablation (모든 2^6 subset 중 유의미한 것) ───
print("\n=== Level Combination Ablation (all 11 proxies) ===")
level_results = []

# 단일 레벨
for lv in LEVELS:
    X = get_X([lv], PROXIES)
    r_lin = compute_r2(X, y, "linear")
    r_mlp = compute_r2(X, y, "mlp")
    level_results.append((f"{lv}", r_lin, r_mlp))
    print(f"  [{lv}]  Ridge={r_lin:.4f}  MLP={r_mlp:.4f}")

# 2레벨 조합 (macro 포함 우선)
print("  --- 2-level combos ---")
for combo in combinations(LEVELS, 2):
    X = get_X(list(combo), PROXIES)
    r_lin = compute_r2(X, y, "linear")
    r_mlp = compute_r2(X, y, "mlp")
    label = "+".join(lv[:4] for lv in combo)
    level_results.append((label, r_lin, r_mlp))
    print(f"  [{label}]  Ridge={r_lin:.4f}  MLP={r_mlp:.4f}")

# 3레벨 (macro+sector 기준 추가 레벨)
print("  --- macro+sector + 1 more ---")
for lv in [l for l in LEVELS if l not in ["macro", "sector"]]:
    combo = ["macro", "sector", lv]
    X = get_X(combo, PROXIES)
    r_lin = compute_r2(X, y, "linear")
    r_mlp = compute_r2(X, y, "mlp")
    label = "mac+sec+" + lv[:4]
    level_results.append((label, r_lin, r_mlp))
    print(f"  [{label}]  Ridge={r_lin:.4f}  MLP={r_mlp:.4f}")

# 전체 6레벨
X_all = get_X(LEVELS, PROXIES)
r_lin = compute_r2(X_all, y, "linear")
r_mlp = compute_r2(X_all, y, "mlp")
level_results.append(("ALL-6", r_lin, r_mlp))
print(f"  [ALL-6]  Ridge={r_lin:.4f}  MLP={r_mlp:.4f}")

# ── 2. Proxy 선택 ablation (macro+sector 기준) ───────────────
print("\n=== Proxy Selection Ablation (macro+sector) ===")
# 단독 proxy 중요도 순으로 정렬 (macro MLP R² 기준)
solo_scores = []
for px in PROXIES:
    X_px = get_X(["macro", "sector"], [px])
    r = compute_r2(X_px, y, "mlp")
    solo_scores.append((r, px))
    print(f"  {px:15s}  MLP={r:.4f}")

solo_scores.sort(reverse=True)
ranked_proxies = [px for _, px in solo_scores]
print(f"\nProxy ranking: {ranked_proxies}")

# Top-k forward selection
print("\n  Top-k selection:")
proxy_results = []
for k in range(1, len(PROXIES) + 1):
    top_k = ranked_proxies[:k]
    X_k   = get_X(["macro", "sector"], top_k)
    r_lin = compute_r2(X_k, y, "linear")
    r_mlp = compute_r2(X_k, y, "mlp")
    proxy_results.append((k, top_k[-1], r_lin, r_mlp))
    print(f"  top-{k:2d} (+{top_k[-1]:15s})  Ridge={r_lin:.4f}  MLP={r_mlp:.4f}")

# ── 시각화 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7))
fig.subplots_adjust(wspace=0.35)

# 패널 1: 레벨 조합 ablation
ax = axes[0]
labels  = [r[0] for r in level_results]
lin_r2  = [r[1] for r in level_results]
mlp_r2  = [r[2] for r in level_results]
x_pos   = np.arange(len(labels))
w = 0.38
b1 = ax.bar(x_pos - w/2, lin_r2, width=w, label="Ridge", color="#4477AA", alpha=0.85)
b2 = ax.bar(x_pos + w/2, mlp_r2, width=w, label="MLP",   color="#EE6677", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("Out-of-sample R²", fontsize=10)
ax.set_title("Level Combination Ablation\n(all 11 proxies per level)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
best_idx = int(np.argmax(mlp_r2))
ax.get_children()[best_idx * 2 + 1].set_edgecolor("gold")
ax.get_children()[best_idx * 2 + 1].set_linewidth(2.5)

# 패널 2: Top-k proxy ablation
ax2 = axes[1]
ks      = [r[0] for r in proxy_results]
lin_r2p = [r[2] for r in proxy_results]
mlp_r2p = [r[3] for r in proxy_results]
added   = [r[1][:8] for r in proxy_results]
ax2.plot(ks, lin_r2p, "o-", color="#4477AA", label="Ridge", linewidth=1.8, markersize=6)
ax2.plot(ks, mlp_r2p, "o-", color="#EE6677", label="MLP",   linewidth=1.8, markersize=6)
ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
for k, lin, mlp, add in zip(ks, lin_r2p, mlp_r2p, added):
    ax2.annotate(add, (k, mlp), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=6.5, color="#EE6677")
ax2.set_xticks(ks)
ax2.set_xlabel("Top-k proxies (macro+sector)", fontsize=10)
ax2.set_ylabel("Out-of-sample R²", fontsize=10)
ax2.set_title("Proxy Forward Selection\n(macro+sector, ranked by solo MLP R²)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle("Proxy Ablation Study (gap=20, 80/20 time split)", fontsize=13, fontweight="bold")
out = "/home/eb/LG_AI/FinTexTS/proxy_ablation.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
