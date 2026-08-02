"""
Time-varying inter-level cosine similarity (rolling window)
- 100 tickers 평균
- 주요 이벤트 마킹: COVID crash, Fed hikes, earnings seasons
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
import glob, os

EMB_DIM  = 384
LEVELS   = ["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"]
DATA_DIR = "/home/eb/LG_AI/data/fintexts"
WINDOW   = 21  # placeholder, will iterate

# ── 주요 이벤트 ───────────────────────────────────────────────
EVENTS = [
    ("2020-02-20", "COVID\nCrash",   "red"),
    ("2022-03-16", "Fed\nHike",      "darkorange"),
    ("2022-11-02", "Fed\n75bp",      "darkorange"),
]
# 분기 실적 시즌 (earnings): 1월, 4월, 7월, 10월 중순~말
EARNINGS_PERIODS = [
    ("2019-01-14", "2019-02-15"),
    ("2019-04-15", "2019-05-15"),
    ("2019-07-15", "2019-08-15"),
    ("2019-10-14", "2019-11-15"),
    ("2020-01-13", "2020-02-14"),
    ("2020-04-13", "2020-05-15"),
    ("2020-07-13", "2020-08-14"),
    ("2020-10-12", "2020-11-13"),
    ("2021-01-11", "2021-02-12"),
    ("2021-04-12", "2021-05-14"),
    ("2021-07-12", "2021-08-13"),
    ("2021-10-11", "2021-11-12"),
    ("2022-01-10", "2022-02-11"),
    ("2022-04-11", "2022-05-13"),
    ("2022-07-11", "2022-08-12"),
    ("2022-10-10", "2022-11-11"),
]

# ── 날짜 기준 로드 ────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*_train.parquet")))
df_ref = pd.read_parquet(files[0]).sort_values("date").reset_index(drop=True)
dates  = pd.to_datetime(df_ref["date"])
T      = len(dates)

# ── 모든 레벨 쌍 rolling similarity 계산 ─────────────────────
pairs = list(combinations(LEVELS, 2))  # 15쌍
sim_sum   = {p: np.zeros(T) for p in pairs}
sim_count = {p: np.zeros(T) for p in pairs}

print(f"Processing {len(files)} tickers...")
for fi, f in enumerate(files):
    df = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df_aligned = df_ref[["date"]].merge(df, on="date", how="left")
    lv_arrs = {}
    for lv in LEVELS:
        cols  = [f"{lv}_emb{i}" for i in range(EMB_DIM)]
        avail = [c for c in cols if c in df_aligned.columns]
        lv_arrs[lv] = df_aligned[avail].fillna(0).values.astype(np.float32) if avail else np.zeros((T, EMB_DIM), dtype=np.float32)
    for lv1, lv2 in pairs:
        a, b   = lv_arrs[lv1], lv_arrs[lv2]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        valid  = (na > 1e-6) & (nb > 1e-6)
        dot    = (a * b).sum(axis=1)
        sim    = np.where(valid, dot / (na * nb + 1e-10), 0.0)
        sim_sum[(lv1,lv2)]   += sim
        sim_count[(lv1,lv2)] += valid.astype(float)

# ── raw average (window 전 기본값) ───────────────────────────
raw_avg = {}
for p in pairs:
    raw_avg[p] = sim_sum[p] / np.maximum(sim_count[p], 1)

# ── 여러 window로 rolling & variance 분석 ────────────────────
WINDOWS = [5, 10, 21, 42, 63]

print("Computing done. Analyzing windows...")

# 각 (pair, window) 조합에서 rolling series의 std (시간적 변동성) 계산
results = []
for w in WINDOWS:
    for p in pairs:
        rolled = pd.Series(raw_avg[p]).rolling(w, min_periods=max(3,w//2)).mean()
        std    = rolled.std()
        # event 근처 변화량: COVID crash 전후 10일 평균 차이
        covid_idx = np.searchsorted(dates.values, np.datetime64("2020-02-20"))
        pre_covid  = rolled.iloc[max(0, covid_idx-21):covid_idx].mean()
        post_covid = rolled.iloc[covid_idx:covid_idx+21].mean()
        covid_delta = abs(post_covid - pre_covid) if not np.isnan(pre_covid) else 0

        fed_idx = np.searchsorted(dates.values, np.datetime64("2022-03-16"))
        pre_fed  = rolled.iloc[max(0, fed_idx-21):fed_idx].mean()
        post_fed = rolled.iloc[fed_idx:fed_idx+21].mean()
        fed_delta = abs(post_fed - pre_fed) if not np.isnan(pre_fed) else 0

        results.append({
            "pair": f"{p[0][:4]}↔{p[1][:4]}",
            "window": w,
            "std": std,
            "covid_delta": covid_delta,
            "fed_delta": fed_delta,
            "score": std + covid_delta * 2 + fed_delta * 2,
        })

df_res = pd.DataFrame(results)

# window별 top pair
print("\n=== Top 5 most time-varying pairs per window ===")
for w in WINDOWS:
    sub = df_res[df_res.window == w].nlargest(5, "std")
    print(f"\nWindow={w:2d}d:")
    for _, r in sub.iterrows():
        print(f"  {r['pair']:25s}  std={r['std']:.4f}  covid_Δ={r['covid_delta']:.4f}  fed_Δ={r['fed_delta']:.4f}")

# 전체 top 10 (score 기준)
top10 = df_res.nlargest(10, "score")
print("\n=== Overall top 10 (std + event response) ===")
print(top10[["pair","window","std","covid_delta","fed_delta","score"]].to_string(index=False))

# ── 가장 유의미한 window + pair 시각화 ───────────────────────
# score 상위 pair들 + 대표 window(21d)로 그림
best_pairs_per_w = {}
for w in WINDOWS:
    sub = df_res[df_res.window == w].nlargest(3, "score")
    best_pairs_per_w[w] = list(sub["pair"])

# 최종 플롯: window 5개 × 공통 관심 pair
# 관심 pair: score 기준 전체 top5 unique pair
top_pairs_unique = df_res.groupby("pair")["score"].max().nlargest(6).index.tolist()
print(f"\nTop unique pairs: {top_pairs_unique}")

def add_events(ax):
    for s, e in EARNINGS_PERIODS:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.06, color="gold", zorder=0)
    for dt, label, c in EVENTS:
        ax.axvline(pd.Timestamp(dt), color=c, linestyle="--", linewidth=1.0, alpha=0.8)

# pair 이름 → tuple 변환
pair_name_map = {f"{p[0][:4]}↔{p[1][:4]}": p for p in pairs}

CMAP = plt.cm.tab10
fig, axes = plt.subplots(len(WINDOWS), 1, figsize=(18, 3.5*len(WINDOWS)), sharex=True)
fig.subplots_adjust(hspace=0.1)

for ax, w in zip(axes, WINDOWS):
    for ci, pname in enumerate(top_pairs_unique):
        p = pair_name_map.get(pname)
        if p is None:
            continue
        rolled = pd.Series(raw_avg[p]).rolling(w, min_periods=max(3,w//2)).mean().values
        ax.plot(dates, rolled, label=pname, linewidth=1.3, alpha=0.85, color=CMAP(ci))
    ax.set_ylim(0.35, 1.05)
    ax.set_ylabel(f"w={w}d", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left", ncol=6)
    ax.grid(axis="y", alpha=0.3)
    add_events(ax)

axes[-1].set_xlabel("Date", fontsize=10)
axes[-1].tick_params(axis="x", rotation=30)
fig.suptitle("Time-varying Cosine Similarity — Top pairs across windows\n(100 tickers avg)",
             fontsize=13, fontweight="bold")

out = "/home/eb/LG_AI/FinTexTS/time_similarity.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
