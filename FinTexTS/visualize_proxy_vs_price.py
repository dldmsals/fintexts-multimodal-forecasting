"""
텍스트 임베딩 Proxy 예측력 분석 (visualize_price_text_relation.py 방식)
각 proxy를 feature로 → Ridge + MLP out-of-sample R²

[단일 레벨 proxy: 레벨 × proxy 조합마다 단일 스칼라 feature로 R² 측정]
  magnitude, shock_1d, shock_5d, novelty_5d, novelty_21d,
  persistence, cum_shock_5d, sparsity_21d, volatility_21d,
  macro_align, days_since

[레벨별 전체 proxy: 한 레벨의 11개 proxy 벡터 → Ridge/MLP R²]
[크로스 레벨 proxy: 여러 레벨 결합 스칼라 → Ridge/MLP R²]
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
CROSS_PROXIES = [
    "macro_sector_sim", "macro_company_sim", "macro_lseg_sim",
    "company_lseg_sim", "macro_company_div",
    "shock_spread", "consensus", "lseg_macro_shock_diff",
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
# proxy_vals[lv][px] = list of scalar values
proxy_vals = {lv: {p: [] for p in PROXIES} for lv in LEVELS}
cross_vals = {p: [] for p in CROSS_PROXIES}
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
        ret = close[t + GAP] / close[t] - 1.0
        if abs(ret) > 2.0:
            continue
        returns.append(ret)

        mac = lv_arrs["macro"][t]

        # 단일 레벨 proxy
        for lv in LEVELS:
            e     = lv_arrs[lv]
            e_t   = e[t];  e_t1 = e[t-1];  e_t5 = e[t-5]
            win5  = e[t-5:t]
            win21 = e[t-MAX_W:t+1]

            proxy_vals[lv]["magnitude"].append(np.linalg.norm(e_t))
            proxy_vals[lv]["shock_1d"].append(np.linalg.norm(e_t - e_t1))
            proxy_vals[lv]["shock_5d"].append(np.linalg.norm(e_t - e_t5))
            proxy_vals[lv]["novelty_5d"].append(np.linalg.norm(e_t - nonzero_mean(win5)))
            proxy_vals[lv]["novelty_21d"].append(np.linalg.norm(e_t - nonzero_mean(win21)))
            proxy_vals[lv]["persistence"].append(cos_sim(e_t, e_t1))

            cs = sum(np.linalg.norm(e[t-k] - e[t-k-1]) for k in range(5))
            proxy_vals[lv]["cum_shock_5d"].append(cs)

            norms21 = np.linalg.norm(win21, axis=1)
            proxy_vals[lv]["sparsity_21d"].append((norms21 > 1e-6).mean())
            proxy_vals[lv]["volatility_21d"].append(norms21.std())
            proxy_vals[lv]["macro_align"].append(cos_sim(e_t, mac))

            shocks_hist = [np.linalg.norm(e[t-k] - e[t-k-1]) for k in range(MAX_W)]
            thresh = np.mean(shocks_hist) + np.std(shocks_hist)
            days = next((k for k, s in enumerate(shocks_hist) if s > thresh), MAX_W)
            proxy_vals[lv]["days_since"].append(days)

        # 크로스 레벨 proxy
        sec = lv_arrs["sector"][t]
        com = lv_arrs["targetCompany"][t]
        lsg = lv_arrs["lseg"][t]

        cross_vals["macro_sector_sim"].append(cos_sim(mac, sec))
        cross_vals["macro_company_sim"].append(cos_sim(mac, com))
        cross_vals["macro_lseg_sim"].append(cos_sim(mac, lsg))
        cross_vals["company_lseg_sim"].append(cos_sim(com, lsg))
        cross_vals["macro_company_div"].append(np.linalg.norm(mac - com))

        shocks_all = [np.linalg.norm(lv_arrs[lv][t] - lv_arrs[lv][t-1]) for lv in LEVELS]
        cross_vals["shock_spread"].append(np.std(shocks_all))

        all_embs = [lv_arrs[lv][t] for lv in LEVELS]
        sims = [cos_sim(all_embs[i], all_embs[j])
                for i in range(len(LEVELS)) for j in range(i+1, len(LEVELS))]
        cross_vals["consensus"].append(np.mean(sims))

        shock_lsg = np.linalg.norm(lsg - lv_arrs["lseg"][t-1])
        shock_mac = np.linalg.norm(mac - lv_arrs["macro"][t-1])
        cross_vals["lseg_macro_shock_diff"].append(abs(shock_lsg - shock_mac))

y = np.array(returns)
print(f"Total samples: {len(y)}")

# ── R² 계산 함수 ──────────────────────────────────────────────
def compute_r2(X_list, y_arr, model_type="linear"):
    X = np.array(X_list).reshape(len(X_list), -1)
    valid = ~np.isnan(X).any(axis=1)
    X, y_ = X[valid], y_arr[valid]

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    n_train = int(len(y_) * 0.8)
    X_tr, X_te = X_s[:n_train], X_s[n_train:]
    y_tr, y_te = y_[:n_train], y_[n_train:]

    if model_type == "linear":
        model = Ridge(alpha=1.0)
    else:
        model = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                             max_iter=300, early_stopping=True,
                             validation_fraction=0.1, random_state=42,
                             learning_rate_init=1e-3)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

# ── 1. 단일 proxy × 레벨 R² 행렬 ────────────────────────────
print("\n=== Single proxy R² ===")
r2_lin = np.zeros((len(LEVELS), len(PROXIES)))
r2_mlp = np.zeros((len(LEVELS), len(PROXIES)))

for li, lv in enumerate(LEVELS):
    for pi, px in enumerate(PROXIES):
        lin = compute_r2(proxy_vals[lv][px], y, "linear")
        mlp = compute_r2(proxy_vals[lv][px], y, "mlp")
        r2_lin[li, pi] = lin
        r2_mlp[li, pi] = mlp
        print(f"  {lv:20s}  {px:15s}  Ridge={lin:.4f}  MLP={mlp:.4f}")

# ── 2. 레벨별 all-proxy 묶음 R² ──────────────────────────────
print("\n=== Per-level all-proxy combined R² ===")
level_lin, level_mlp = [], []
for lv in LEVELS:
    X_lv = np.column_stack([proxy_vals[lv][px] for px in PROXIES])
    lin = compute_r2(X_lv.tolist(), y, "linear")
    mlp = compute_r2(X_lv.tolist(), y, "mlp")
    level_lin.append(lin); level_mlp.append(mlp)
    print(f"  {lv:20s}  Ridge={lin:.4f}  MLP={mlp:.4f}")

# ── 3. 크로스 레벨 proxy 묶음 R² ─────────────────────────────
print("\n=== Cross-level proxy combined R² ===")
X_cross = np.column_stack([cross_vals[px] for px in CROSS_PROXIES])
n = min(len(X_cross), len(y))
cross_lin = compute_r2(X_cross[:n].tolist(), y[:n], "linear")
cross_mlp = compute_r2(X_cross[:n].tolist(), y[:n], "mlp")
print(f"  Combined  Ridge={cross_lin:.4f}  MLP={cross_mlp:.4f}")
for px in CROSS_PROXIES:
    x = cross_vals[px][:n]
    lin = compute_r2(x, y[:n], "linear")
    mlp = compute_r2(x, y[:n], "mlp")
    print(f"  {px:25s}  Ridge={lin:.4f}  MLP={mlp:.4f}")

# ── 시각화 ────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

def draw_heatmap(ax, mat, row_labels, col_labels, title):
    vmax = max(abs(mat).max(), 0.005)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="R²")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = mat[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)

draw_heatmap(fig.add_subplot(gs[0, 0]), r2_lin, LEVELS, PROXIES,
             "Single Proxy R² — Linear (Ridge)")
draw_heatmap(fig.add_subplot(gs[0, 1]), r2_mlp, LEVELS, PROXIES,
             "Single Proxy R² — Non-linear (MLP)")

# 레벨별 all-proxy 묶음 + 크로스 proxy bar chart
ax_bar = fig.add_subplot(gs[1, :])
x_pos  = np.arange(len(LEVELS) + 1)
labels = LEVELS + ["[cross-level]"]
lin_vals = level_lin + [cross_lin]
mlp_vals = level_mlp + [cross_mlp]
w = 0.38
b1 = ax_bar.bar(x_pos - w/2, lin_vals, width=w, label="Ridge (all proxies)", color="#4477AA", alpha=0.85)
b2 = ax_bar.bar(x_pos + w/2, mlp_vals, width=w, label="MLP   (all proxies)", color="#EE6677", alpha=0.85)
ax_bar.axhline(0, color="black", linewidth=0.8)
ax_bar.axvline(len(LEVELS) - 0.5, color="gray", linestyle="--", linewidth=0.9)
ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels(labels, fontsize=10)
ax_bar.set_ylabel("Out-of-sample R²", fontsize=10)
ax_bar.set_title("All Proxies Combined per Level vs Cross-level Proxies", fontsize=11, fontweight="bold")
ax_bar.legend(fontsize=9); ax_bar.grid(axis="y", alpha=0.3)
for bar, val in zip(list(b1) + list(b2), lin_vals + mlp_vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                val + (0.001 if val >= 0 else -0.003),
                f"{val:.4f}", ha="center", va="bottom", fontsize=7.5)

plt.suptitle("Text Proxy Predictive Power: Ridge vs MLP (gap=20, 80/20 time split)",
             fontsize=13, fontweight="bold")
out = "/home/eb/LG_AI/FinTexTS/proxy_vs_price.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
