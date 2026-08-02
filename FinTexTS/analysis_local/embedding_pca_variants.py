"""
PCA 분리도 탐색: 레벨 조합 x 윈도우 집계 방식을 바꿔가며
"20일 후 수익률 상승/하락이 임베딩 공간에서 분리되는가"를 비교.

panelC(targetCompany 단독, 당일 스냅샷)는 분리가 거의 없었음(중심점거리/분산=0.034).
다른 조합에서 더 나은 분리가 나오는지 체계적으로 스캔.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os
from sklearn.decomposition import PCA

EMB_DIM = 64
ALL_LEVELS = ["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"]
DATA_DIR = "/home/eb/LG_AI/data/linq_fintexts_perlevel"
GAP = 20
OUT_DIR = "/home/eb/LG_AI/FinTexTS/analysis_local"
WINDOW = 64  # 누적평균 윈도우 (seq_len과 동일)

LEVEL_COMBOS = {
    "targetCompany_only": ["targetCompany"],
    "company_related_sector": ["targetCompany", "relatedCompany", "sector"],
    "all6_concat": ALL_LEVELS,
}
AGG_MODES = ["snapshot", "cum_mean"]  # snapshot=당일 1일치, cum_mean=직전 64일 누적평균

files = sorted(glob.glob(os.path.join(DATA_DIR, "*_train.parquet")))
print(f"{len(files)} tickers")


def load_levels(f, levels):
    df = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
    close = df["close"].values.astype(np.float32)
    arrs = []
    for lv in levels:
        cols = [f"{lv}_emb{i}" for i in range(EMB_DIM)]
        arrs.append(df[cols].fillna(0).values.astype(np.float32))
    emb = np.concatenate(arrs, axis=1)  # [T, levels*64]
    return close, emb


results = []
for combo_name, levels in LEVEL_COMBOS.items():
    for agg in AGG_MODES:
        X_all, y_all = [], []
        for f in files:
            close, emb = load_levels(f, levels)
            T = len(close)
            for t in range(WINDOW, T - GAP):
                if close[t] < 1e-6 or close[t + GAP] < 1e-6:
                    continue
                ret = close[t + GAP] / close[t] - 1.0
                if abs(ret) > 2.0:
                    continue
                if agg == "snapshot":
                    if np.linalg.norm(emb[t]) < 1e-6:
                        continue
                    feat = emb[t]
                else:  # cum_mean
                    window = emb[t - WINDOW + 1 : t + 1]
                    mask = np.linalg.norm(window, axis=1) > 1e-6
                    if mask.sum() == 0:
                        continue
                    feat = window[mask].mean(axis=0)
                X_all.append(feat)
                y_all.append(ret)

        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=np.float32)

        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X_all)
        up_mask = y_all > 0
        down_mask = y_all <= 0
        centroid_up = X2[up_mask].mean(axis=0)
        centroid_down = X2[down_mask].mean(axis=0)
        centroid_dist = np.linalg.norm(centroid_up - centroid_down)
        spread = X2.std(axis=0).mean()
        sep_ratio = centroid_dist / spread

        tag = f"{combo_name}__{agg}"
        results.append((tag, len(X_all), sep_ratio, pca.explained_variance_ratio_))
        print(f"[{tag}] n={len(X_all)} sep_ratio={sep_ratio:.4f} "
              f"var_ratio={pca.explained_variance_ratio_.round(3)}")

        plt.figure(figsize=(6, 5.5))
        plt.scatter(X2[down_mask, 0], X2[down_mask, 1], s=3, alpha=0.2, color="tab:blue", label=f"하락 n={down_mask.sum()}")
        plt.scatter(X2[up_mask, 0], X2[up_mask, 1], s=3, alpha=0.2, color="tab:red", label=f"상승 n={up_mask.sum()}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.title(f"{tag}\nsep_ratio={sep_ratio:.4f}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"pca_{tag}.png"), dpi=110)
        plt.close()

print("\n=== 요약 (분리도 내림차순) ===")
for tag, n, sep, var in sorted(results, key=lambda r: -r[2]):
    print(f"{tag:35s} n={n:7d} sep_ratio={sep:.4f}")
