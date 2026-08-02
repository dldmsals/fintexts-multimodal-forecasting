"""
Linq 임베딩(64d, targetCompany 레벨)과 주가의 관계를 직관적으로 시각화.

기존 price_text_relation.py / visualize_cosine_sim_vs_price.py 등은
Ridge/MLP R^2, Pearson r 히트맵 등 통계 수치 위주였음. 여기서는 "임베딩
공간에서의 변화가 실제로 주가와 어떻게 보이는지"를 눈으로 바로 볼 수 있는
3종 플롯을 만든다.

Panel A: 대표 종목(AAPL/TSLA/JPM/XOM) 시계열 — 정규화 종가 vs 텍스트 임베딩
         day-to-day cosine distance(전일 대비 텍스트가 얼마나 바뀌었는지)
Panel B: 임베딩 변화량(전일 대비 L2 거리) vs 다음날 |수익률| 산점도 + 상관계수
Panel C: 전 종목 targetCompany 임베딩을 PCA 2D로 투영, 20일 forward return
         부호(상승/하락)로 색칠 — "임베딩 공간에서 상승일/하락일이 실제로
         분리되어 보이는가"를 직접 확인
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
from scipy.stats import pearsonr

EMB_DIM = 64
LEVEL = "targetCompany"
DATA_DIR = "/home/eb/LG_AI/data/linq_fintexts_perlevel"
GAP = 20
OUT_DIR = "/home/eb/LG_AI/FinTexTS/analysis_local"
HIGHLIGHT_TICKERS = ["AAPL", "TSLA", "JPM", "XOM"]

os.makedirs(OUT_DIR, exist_ok=True)
cols = [f"{LEVEL}_emb{i}" for i in range(EMB_DIM)]


def load_ticker(ticker):
    path = os.path.join(DATA_DIR, f"{ticker}_train.parquet")
    df = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    emb = df[cols].fillna(0).values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    return df["date"].values, close, emb


def cos_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return np.nan
    return 1.0 - float(np.dot(a, b) / (na * nb))


# ── Panel A: 대표 종목 시계열 (정규화 종가 vs 임베딩 day-to-day cosine distance) ──
fig, axes = plt.subplots(len(HIGHLIGHT_TICKERS), 1, figsize=(12, 3 * len(HIGHLIGHT_TICKERS)), sharex=False)
for ax, ticker in zip(axes, HIGHLIGHT_TICKERS):
    dates, close, emb = load_ticker(ticker)
    norm_close = close / close[0]
    drift = [np.nan] + [cos_dist(emb[t - 1], emb[t]) for t in range(1, len(emb))]
    drift = np.array(drift)

    ax.plot(dates, norm_close, color="tab:blue", label="정규화 종가 (t=0 기준)")
    ax.set_ylabel("정규화 종가", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(dates, drift, color="tab:red", alpha=0.4, linewidth=0.8, label="텍스트 변화량(전일 대비 cosine distance)")
    ax2.set_ylabel("텍스트 cosine distance", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title(f"{ticker}: 정규화 종가 vs targetCompany 임베딩 일일 변화량")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "panelA_price_vs_text_drift_timeseries.png"), dpi=120)
plt.close()
print("[Saved] panelA_price_vs_text_drift_timeseries.png")

# ── Panel B: 임베딩 변화량 vs 다음날 |수익률| 산점도 ──
all_drift, all_absret = [], []
for ticker in HIGHLIGHT_TICKERS:
    dates, close, emb = load_ticker(ticker)
    for t in range(1, len(emb) - 1):
        d = cos_dist(emb[t - 1], emb[t])
        if np.isnan(d) or close[t] < 1e-6:
            continue
        ret = close[t + 1] / close[t] - 1.0
        all_drift.append(d)
        all_absret.append(abs(ret))

all_drift = np.array(all_drift)
all_absret = np.array(all_absret)
r, p = pearsonr(all_drift, all_absret)

plt.figure(figsize=(7, 6))
plt.scatter(all_drift, all_absret, s=6, alpha=0.3)
plt.xlabel("텍스트 임베딩 변화량 (전일 대비 cosine distance)")
plt.ylabel("다음날 |수익률|")
plt.title(f"임베딩 변화량 vs 다음날 가격 변동폭 (대표 4종목 합산)\nPearson r={r:.4f} (p={p:.3g}), n={len(all_drift)}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "panelB_drift_vs_absreturn_scatter.png"), dpi=120)
plt.close()
print(f"[Saved] panelB_drift_vs_absreturn_scatter.png  r={r:.4f} p={p:.3g} n={len(all_drift)}")

# ── Panel C: 전 종목 targetCompany 임베딩 PCA 2D, forward return 부호로 색칠 ──
files = sorted(glob.glob(os.path.join(DATA_DIR, "*_train.parquet")))
X_all, y_all = [], []
for fi, f in enumerate(files):
    ticker = os.path.basename(f).replace("_train.parquet", "")
    dates, close, emb = load_ticker(ticker)
    T = len(close)
    for t in range(T - GAP):
        if np.linalg.norm(emb[t]) < 1e-6 or close[t] < 1e-6 or close[t + GAP] < 1e-6:
            continue
        ret = close[t + GAP] / close[t] - 1.0
        if abs(ret) > 2.0:
            continue
        X_all.append(emb[t])
        y_all.append(ret)

X_all = np.array(X_all, dtype=np.float32)
y_all = np.array(y_all, dtype=np.float32)
print(f"PCA fit on {len(X_all)} (ticker, day) samples across {len(files)} tickers")

pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X_all)
var_ratio = pca.explained_variance_ratio_

up_mask = y_all > 0
down_mask = y_all <= 0

plt.figure(figsize=(8, 7))
plt.scatter(X2[down_mask, 0], X2[down_mask, 1], s=4, alpha=0.25, color="tab:blue", label=f"하락(20일 후) n={down_mask.sum()}")
plt.scatter(X2[up_mask, 0], X2[up_mask, 1], s=4, alpha=0.25, color="tab:red", label=f"상승(20일 후) n={up_mask.sum()}")
plt.xlabel(f"PC1 ({var_ratio[0]*100:.1f}% 분산)")
plt.ylabel(f"PC2 ({var_ratio[1]*100:.1f}% 분산)")
plt.title("targetCompany 임베딩(64d) PCA 2D 투영, 20일 후 수익률 부호로 색칠\n(전 종목, 전 구간)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "panelC_pca_scatter_by_forward_return.png"), dpi=120)
plt.close()
print("[Saved] panelC_pca_scatter_by_forward_return.png")

# 상승/하락 두 그룹의 PCA 공간 중심점 거리 (분리도 정량 지표)
centroid_up = X2[up_mask].mean(axis=0)
centroid_down = X2[down_mask].mean(axis=0)
centroid_dist = np.linalg.norm(centroid_up - centroid_down)
spread = X2.std(axis=0).mean()
print(f"상승/하락 그룹 중심점 거리={centroid_dist:.4f}, 전체 분산(std) 평균={spread:.4f}, 비율={centroid_dist/spread:.4f}")
