"""
텍스트 임베딩 예측력: Linear(Ridge) vs Non-linear(MLP) 비교
레벨 6개 × 윈도우 4개 × 방법 2개 → R² 히트맵
정답: close[t+20] / close[t] - 1  (gap=20 forward return)

Fix: look-ahead bias 제거
  1. Scaler/PCA를 train에만 fit → test는 transform만
  2. 날짜(시간) 기준 80/20 분리 (ticker별 연속 인덱스 분리 X)
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
WINDOWS  = {"1d": 1, "5d": 5, "21d": 21, "42d": 42}

files = sorted(glob.glob(os.path.join(DATA_DIR, "*_train.parquet")))
print(f"Loading {len(files)} tickers...")

df_ref = pd.read_parquet(files[0]).sort_values("date").reset_index(drop=True)
dates  = pd.to_datetime(df_ref["date"])
T      = len(dates)

# 날짜 기준 80/20 분리 — 모든 ticker에 동일한 시간 경계 적용
TRAIN_END_T = int(T * 0.8)  # 2019-01-01 ~ 2022-03-14
print(f"Date split: train [0, {TRAIN_END_T}) = {dates[0].date()} ~ {dates[TRAIN_END_T-1].date()}")
print(f"            test  [{TRAIN_END_T}, {T}-GAP) = {dates[TRAIN_END_T].date()} ~ {dates[T-GAP-1].date()}")

def nonzero_mean(arr):
    mask = np.linalg.norm(arr, axis=1) > 1e-6
    return arr[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(EMB_DIM, dtype=np.float32)

# ── 데이터 수집 (train/test 분리해서) ─────────────────────────
X_tr = {lv: {w: [] for w in WINDOWS} for lv in LEVELS}
X_te = {lv: {w: [] for w in WINDOWS} for lv in LEVELS}
y_tr, y_te = [], []

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

        # 날짜 기준으로 train/test 구분
        is_train = (t < TRAIN_END_T)
        target_y  = y_tr if is_train else y_te
        target_X  = X_tr if is_train else X_te

        target_y.append(ret)
        for lv in LEVELS:
            for wname, w in WINDOWS.items():
                target_X[lv][wname].append(nonzero_mean(lv_arrs[lv][t - w + 1: t + 1]))

y_tr_arr = np.array(y_tr)
y_te_arr = np.array(y_te)
print(f"Train samples: {len(y_tr_arr)},  Test samples: {len(y_te_arr)}")

# ── R² 계산 함수 (look-ahead bias 없음) ──────────────────────
def compute_r2(X_tr_list, X_te_list, y_tr, y_te, model_type="linear"):
    X_tr = np.array(X_tr_list)
    X_te = np.array(X_te_list)

    # NaN 행 제거
    valid_tr = ~np.isnan(X_tr).any(axis=1)
    valid_te = ~np.isnan(X_te).any(axis=1)
    X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
    X_te, y_te = X_te[valid_te], y_te[valid_te]

    if len(X_tr) == 0 or len(X_te) == 0:
        return 0.0

    # ✅ train에만 fit → test는 transform만
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    n_comp = min(50, X_tr_s.shape[1], X_tr_s.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)

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

    model.fit(X_tr_p, y_tr)
    y_pred = model.predict(X_te_p)

    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

# ── 모든 조합 계산 ────────────────────────────────────────────
wnames = list(WINDOWS.keys())
r2 = {
    "Linear (Ridge)":   {lv: {} for lv in LEVELS},
    "Non-linear (MLP)": {lv: {} for lv in LEVELS},
}

print("\n=== Computing R² (no look-ahead bias) ===")
for lv in LEVELS:
    for wname in wnames:
        lin = compute_r2(X_tr[lv][wname], X_te[lv][wname], y_tr_arr, y_te_arr, "linear")
        mlp = compute_r2(X_tr[lv][wname], X_te[lv][wname], y_tr_arr, y_te_arr, "mlp")
        r2["Linear (Ridge)"][lv][wname]   = lin
        r2["Non-linear (MLP)"][lv][wname] = mlp
        print(f"  {lv:20s}  {wname:4s}  Ridge={lin:.4f}  MLP={mlp:.4f}")

# ── 시각화: 히트맵 2개 나란히 ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.35)

for ax, method in zip(axes, r2.keys()):
    matrix = np.array([[r2[method][lv][w] for w in wnames] for lv in LEVELS])

    vmax = max(abs(matrix.max()), abs(matrix.min()))
    vmax = max(vmax, 0.01)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.85, label="R²")

    ax.set_xticks(range(len(wnames)))
    ax.set_xticklabels(wnames, fontsize=10)
    ax.set_yticks(range(len(LEVELS)))
    ax.set_yticklabels(LEVELS, fontsize=10)
    ax.set_xlabel("Text Window Size", fontsize=10)
    ax.set_title(method, fontsize=12, fontweight="bold")

    for i, lv in enumerate(LEVELS):
        for j, w in enumerate(wnames):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)

plt.suptitle(
    "Text Embedding Predictive Power: Linear vs Non-linear\n"
    f"(gap=20 forward return, PCA-50, date-based split: train ~2022-03-14 / test 2022-03-15~)",
    fontsize=11, fontweight="bold"
)
out = "/home/eb/LG_AI/FinTexTS/price_text_relation.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()
