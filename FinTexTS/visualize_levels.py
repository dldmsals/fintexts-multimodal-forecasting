"""
레벨 간 관계 시각화:
1. 레벨별 평균 cosine similarity 히트맵
2. PCA - 레벨별 임베딩 분포
3. 시간에 따른 레벨별 임베딩 norm (활성도)
4. 시간에 따른 레벨 간 cosine similarity
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import glob, os

EMB_DIM = 384
LEVELS  = ["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"]
COLORS  = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
DATA_DIR = "/home/eb/LG_AI/data/fintexts"
N_TICKERS = 100  # 전체 ticker

# ── 데이터 로드 ────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*_train.parquet")))[:N_TICKERS]

level_embs  = {lv: [] for lv in LEVELS}  # 전체 샘플 임베딩 모음
level_norms = {lv: [] for lv in LEVELS}  # 날짜별 norm
dates_all   = []

for f in files:
    df = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
    dates_all.append(pd.to_datetime(df["date"]))

    for lv in LEVELS:
        cols = [f"{lv}_emb{i}" for i in range(EMB_DIM)]
        arr  = df[cols].fillna(0).values.astype(np.float32)       # [T, 384]
        mask = (np.abs(arr).sum(axis=1) > 1e-6)                   # non-zero 날
        if mask.sum() > 0:
            level_embs[lv].append(arr[mask])
        level_norms[lv].append(arr)                                # 전체 (zero 포함)

# ── 레벨별 전체 임베딩 합치기 ─────────────────────────────────
for lv in LEVELS:
    level_embs[lv] = np.concatenate(level_embs[lv], axis=0) if level_embs[lv] else np.zeros((1, EMB_DIM))

# ── Figure 구성 ───────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# ── 1. 레벨별 평균 cosine similarity 히트맵 ──────────────────
ax1 = fig.add_subplot(gs[0, 0])
mean_embs = np.stack([level_embs[lv].mean(axis=0) for lv in LEVELS])  # [6, 384]
sim_matrix = cosine_similarity(mean_embs)
im = ax1.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
ax1.set_xticks(range(len(LEVELS))); ax1.set_xticklabels(LEVELS, rotation=45, ha="right", fontsize=9)
ax1.set_yticks(range(len(LEVELS))); ax1.set_yticklabels(LEVELS, fontsize=9)
for i in range(len(LEVELS)):
    for j in range(len(LEVELS)):
        ax1.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=8,
                 color="black" if abs(sim_matrix[i,j]) < 0.7 else "white")
plt.colorbar(im, ax=ax1, shrink=0.8)
ax1.set_title("Mean Embedding Cosine Similarity\n(level vs level)", fontsize=11, fontweight="bold")

# ── 2. PCA - 레벨별 임베딩 분포 ──────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
MAX_PER_LEVEL = 300
all_embs  = []
all_labels = []
for i, lv in enumerate(LEVELS):
    e = level_embs[lv]
    idx = np.random.choice(len(e), min(MAX_PER_LEVEL, len(e)), replace=False)
    all_embs.append(e[idx])
    all_labels.extend([i] * len(idx))

all_embs   = np.concatenate(all_embs, axis=0)
all_labels = np.array(all_labels)
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(all_embs)

for i, lv in enumerate(LEVELS):
    mask = all_labels == i
    ax2.scatter(proj[mask, 0], proj[mask, 1], c=COLORS[i], label=lv,
                alpha=0.4, s=15, edgecolors="none")
ax2.legend(fontsize=8, markerscale=2)
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
ax2.set_title("PCA of Level Embeddings\n(non-zero days only)", fontsize=11, fontweight="bold")

# ── 3. 시간에 따른 레벨별 임베딩 norm (100 ticker 평균) ──────
ax3 = fig.add_subplot(gs[1, 0])

# 공통 날짜 기준: AAPL 날짜 사용
df_ref  = pd.read_parquet(os.path.join(DATA_DIR, "AAPL_train.parquet")).sort_values("date").reset_index(drop=True)
dates_ref = pd.to_datetime(df_ref["date"])
T = len(df_ref)

norm_sum   = {lv: np.zeros(T) for lv in LEVELS}
norm_count = {lv: np.zeros(T) for lv in LEVELS}

for f in files:
    df = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    # 날짜 align
    merged = df_ref[["date"]].merge(df[["date"] + [f"{lv}_emb0" for lv in LEVELS]], on="date", how="left")
    for lv in LEVELS:
        cols = [f"{lv}_emb{i2}" for i2 in range(EMB_DIM)]
        available_cols = [c for c in cols if c in df.columns]
        if not available_cols:
            continue
        arr_full = df_ref[["date"]].merge(df[["date"] + available_cols], on="date", how="left").drop("date", axis=1).fillna(0).values.astype(np.float32)
        norm = np.linalg.norm(arr_full, axis=1)
        valid = norm > 1e-6
        norm_sum[lv]   += norm
        norm_count[lv] += valid.astype(float)

for i, lv in enumerate(LEVELS):
    avg_norm = norm_sum[lv] / np.maximum(norm_count[lv], 1)
    smooth   = pd.Series(avg_norm).rolling(7, min_periods=1).mean().values
    ax3.plot(dates_ref, smooth, color=COLORS[i], label=lv, alpha=0.8, linewidth=1.2)

ax3.set_xlabel("Date", fontsize=9)
ax3.set_ylabel("Mean Embedding Norm (7d avg)", fontsize=9)
ax3.set_title("Level Activity over Time (100 tickers avg)\n7-day rolling norm", fontsize=11, fontweight="bold")
ax3.legend(fontsize=8)
ax3.tick_params(axis='x', rotation=30)

# ── 4. 레벨 간 시간별 cosine similarity (100 ticker 평균) ─────
ax4 = fig.add_subplot(gs[1, 1])

pairs = [
    ("macro",         "sector",        "#e41a1c"),
    ("macro",         "targetCompany", "#377eb8"),
    ("sector",        "targetCompany", "#4daf4a"),
    ("targetCompany", "lseg",          "#984ea3"),
]
sim_sum   = {(lv1, lv2): np.zeros(T) for lv1, lv2, _ in pairs}
sim_count = {(lv1, lv2): np.zeros(T) for lv1, lv2, _ in pairs}

for f in files:
    df = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
    lv_arrs = {}
    for lv in LEVELS:
        cols = [f"{lv}_emb{i2}" for i2 in range(EMB_DIM)]
        available = [c for c in cols if c in df.columns]
        if not available:
            lv_arrs[lv] = np.zeros((len(df), EMB_DIM), dtype=np.float32)
            continue
        arr_df = df_ref[["date"]].merge(df[["date"] + available], on="date", how="left").drop("date", axis=1).fillna(0).values.astype(np.float32)
        lv_arrs[lv] = arr_df

    for lv1, lv2, _ in pairs:
        a = lv_arrs[lv1]
        b = lv_arrs[lv2]
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        valid = (na > 1e-6) & (nb > 1e-6)
        dot = (a * b).sum(axis=1)
        sim = np.where(valid, dot / (na * nb + 1e-10), 0.0)
        sim_sum[(lv1, lv2)]   += sim
        sim_count[(lv1, lv2)] += valid.astype(float)

for lv1, lv2, c in pairs:
    avg_sim = sim_sum[(lv1, lv2)] / np.maximum(sim_count[(lv1, lv2)], 1)
    smooth  = pd.Series(avg_sim).rolling(14, min_periods=1).mean().values
    ax4.plot(dates_ref, smooth, color=c, label=f"{lv1[:3]} ↔ {lv2[:3]}", alpha=0.8, linewidth=1.2)

ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax4.set_xlabel("Date", fontsize=9)
ax4.set_ylabel("Cosine Similarity (14d avg)", fontsize=9)
ax4.set_title("Inter-level Cosine Similarity over Time (100 tickers avg)\n14-day rolling", fontsize=11, fontweight="bold")
ax4.legend(fontsize=8)
ax4.tick_params(axis='x', rotation=30)

plt.suptitle("Text Embedding Level Analysis", fontsize=14, fontweight="bold", y=1.01)
out_path = "/home/eb/LG_AI/FinTexTS/level_analysis.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.close()
