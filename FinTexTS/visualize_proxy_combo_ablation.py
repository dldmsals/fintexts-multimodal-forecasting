"""
프록시 조합 ablation:
레벨 ablation 결과 (macro=0.2761, macro+targetCompany=0.2863) 기준
11개 프록시 중 k=1..4 subset → MLP out-of-sample R²
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
print(f"Loading {len(files)} tickers...", flush=True)

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-6 and nb > 1e-6 else 0.0

def nonzero_mean(arr):
    mask = np.linalg.norm(arr, axis=1) > 1e-6
    return arr[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(EMB_DIM, np.float32)

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
        m = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu",
                         max_iter=200, early_stopping=True,
                         validation_fraction=0.1, random_state=42,
                         learning_rate_init=1e-3, n_iter_no_change=10)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    ss_r = np.sum((y_te - pred) ** 2)
    ss_t = np.sum((y_te - y_te.mean()) ** 2)
    return 1 - ss_r / ss_t if ss_t > 0 else 0.0

def get_X(level_list, proxy_list):
    return np.column_stack([proxy_vals[lv][px]
                            for lv in level_list for px in proxy_list])

# 레벨 설정: 레벨 ablation 결과 기준 top 2
LEVEL_CONFIGS = {
    "macro":       ["macro"],
    "macro+targ":  ["macro", "targetCompany"],
}

all_results = {}   # config_name → list of (combo_tuple, k, r_lin, r_mlp)

for config_name, level_list in LEVEL_CONFIGS.items():
    print(f"\n{'='*60}")
    print(f"=== Proxy Combo Ablation: {config_name} ===")
    results = []

    for k in range(1, 4):   # k=1,2,3 (k=4: 330개 → 너무 느림)
        combos = list(combinations(PROXIES, k))
        print(f"\n  k={k} ({len(combos)} combos):", flush=True)
        k_results = []
        for i, combo in enumerate(combos):
            X    = get_X(level_list, list(combo))
            r_l  = compute_r2(X, y, "linear")
            r_m  = compute_r2(X, y, "mlp")
            k_results.append((combo, k, r_l, r_m))
            if (i + 1) % 20 == 0:
                best = max(k_results, key=lambda x: x[3])
                print(f"    ... {i+1}/{len(combos)}  curr_best={best[3]:.4f}", flush=True)

        k_results.sort(key=lambda x: -x[3])
        results.extend(k_results)

        print(f"  Top-5 (k={k}):")
        for combo, _, r_l, r_m in k_results[:5]:
            label = "+".join(px[:6] for px in combo)
            print(f"    {label:50s}  Ridge={r_l:.4f}  MLP={r_m:.4f}")
        print(f"  Bottom-3 (k={k}):")
        for combo, _, r_l, r_m in k_results[-3:]:
            label = "+".join(px[:6] for px in combo)
            print(f"    {label:50s}  Ridge={r_l:.4f}  MLP={r_m:.4f}")

    # 전체 top-10
    results_sorted = sorted(results, key=lambda x: -x[3])
    print(f"\n  Overall Top-10 ({config_name}):")
    for combo, k, r_l, r_m in results_sorted[:10]:
        label = "+".join(px[:7] for px in combo)
        print(f"    k={k}  {label:60s}  MLP={r_m:.4f}")

    all_results[config_name] = results

# ── 시각화 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(22, 14))
fig.subplots_adjust(hspace=0.4, wspace=0.35)

COLORS = {1: "#4477AA", 2: "#66CCEE", 3: "#228833", 4: "#CCBB44"}
K_LIST = [1, 2, 3, 4]

for col_idx, (config_name, results) in enumerate(all_results.items()):
    # 상단: scatter plot (k별 R² 분포)
    ax_sc = axes[0][col_idx]
    for k in K_LIST:
        r2s = [r[3] for r in results if r[1] == k]
        if not r2s:
            continue
        jitter = np.random.default_rng(k).uniform(-0.15, 0.15, len(r2s))
        ax_sc.scatter([k + j for j in jitter], r2s,
                      color=COLORS[k], alpha=0.35, s=18, label=f"k={k}")

    # 각 k별 best 표시
    for k in K_LIST:
        k_res = [(r[0], r[3]) for r in results if r[1] == k]
        if not k_res:
            continue
        best_combo, best_r2 = max(k_res, key=lambda x: x[1])
        ax_sc.scatter([k], [best_r2], color=COLORS[k], s=120, zorder=5,
                      edgecolors="black", linewidths=1.5)
        label = "+".join(px[:5] for px in best_combo)
        ax_sc.annotate(label, (k, best_r2), textcoords="offset points",
                       xytext=(6, 3), fontsize=6, color=COLORS[k],
                       fontweight="bold")

    ax_sc.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_sc.set_xticks(K_LIST)
    ax_sc.set_xlabel("# proxies in combination (k)", fontsize=10)
    ax_sc.set_ylabel("Out-of-sample R² (MLP)", fontsize=10)
    ax_sc.set_title(f"[{config_name}] Proxy combo R² distribution\n(★=best per k)",
                    fontsize=11, fontweight="bold")
    ax_sc.legend(fontsize=8); ax_sc.grid(alpha=0.3)

    # 하단: overall top-15 bar chart
    ax_bar = axes[1][col_idx]
    top15  = sorted(results, key=lambda x: -x[3])[:15]
    labels = ["+".join(px[:5] for px in r[0]) for r in top15]
    r2s    = [r[3] for r in top15]
    ks     = [r[1] for r in top15]
    x_pos  = np.arange(len(labels))
    bar_colors = [COLORS[k] for k in ks]
    bars = ax_bar.bar(x_pos, r2s, color=bar_colors, alpha=0.85, edgecolor="white")
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, rotation=50, ha="right", fontsize=6.5)
    ax_bar.set_ylabel("Out-of-sample R² (MLP)", fontsize=10)
    ax_bar.set_title(f"[{config_name}] Top-15 proxy combinations",
                     fontsize=11, fontweight="bold")
    for bar, val, k in zip(bars, r2s, ks):
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    val + 0.002, f"k={k}", ha="center", va="bottom", fontsize=6)
    # legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=COLORS[k], label=f"k={k}") for k in K_LIST]
    ax_bar.legend(handles=legend_patches, fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

plt.suptitle("Proxy Combination Ablation (k=1..4, MLP R², gap=20, 80/20 time split)",
             fontsize=13, fontweight="bold")
out = "/home/eb/LG_AI/FinTexTS/proxy_combo_ablation.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
