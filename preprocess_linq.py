"""
LINQ 임베딩(4096d)을 레벨별 PCA로 64d 압축 후 per-ticker parquet 저장.

변경점: 레벨마다 독립적인 PCA를 학습 (BERT pca_64d.pkl과 동일한 방식).

출력: data/linq_fintexts/{TICKER}_{split}.parquet
  - date, open, high, low, close, volume
  - macro_emb0 .. macro_emb63
  - sector_emb0 .. sector_emb63
  - ...

Usage:
  python preprocess_linq.py --n_components 64
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

DATA_DIR   = "/home/eb/LG_AI"
LINQ_PATH  = os.path.join(DATA_DIR, "data/linq_embeddings.tmp")
TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "test.parquet")
LINQ_DIM   = 4096

LEVELS = {
    "macro": [
        "macro_category1", "macro_category2", "macro_category3",
        "macro_category4", "macro_category5",
    ],
    "sector": [
        "sector_category1", "sector_category2", "sector_category3",
        "sector_category4", "sector_category5",
    ],
    "targetCompany": [
        "targetCompany_category1", "targetCompany_category2",
        "targetCompany_category3",
    ],
    "relatedCompany": [
        "relatedCompany_category1", "relatedCompany_category2",
        "relatedCompany_category3",
    ],
    "filing": [
        "filing_financialStatement", "filing_governanceRisks",
        "filing_overviewProduct",   "filing_recentEventCatalyst",
        "filing_strategyMarketOps",
    ],
    "lseg": [f"lseg_news{i}" for i in range(1, 11)],
}

PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def load_linq(path: str) -> pd.DataFrame:
    print(f"Loading LINQ embeddings from {path} ...")
    t0 = time.time()
    f = pq.ParquetFile(path)
    df = f.read().to_pandas()
    df = df.set_index("text_id")
    df.columns = [f"emb_{i}" for i in range(LINQ_DIM)]
    print(f"  done ({time.time()-t0:.1f}s)  shape={df.shape}")
    return df


def get_level_train_ids(train_df: pd.DataFrame, level_cols: list, linq_index) -> list:
    """학습 기간(2022년 이전) 데이터의 레벨별 고유 text_id 수집."""
    dates = pd.to_datetime(train_df["date"])
    train_mask = (dates < "2022-01-01").values
    sub = train_df[train_mask]
    ids = pd.unique(sub[level_cols].values.ravel())
    ids = [x for x in ids if pd.notna(x) and x in linq_index]
    return ids


def fit_level_pcas(linq: pd.DataFrame, train_df: pd.DataFrame,
                   n_components: int, pca_save_path: str) -> dict:
    """레벨별로 독립 PCA 학습. {level: sklearn.PCA} 반환."""
    pcas = {}
    for level_name, level_cols in LEVELS.items():
        cols = [c for c in level_cols if c in train_df.columns]
        if not cols:
            print(f"  [{level_name}] no columns — skip")
            continue
        ids = get_level_train_ids(train_df, cols, linq.index)
        if not ids:
            print(f"  [{level_name}] no valid text_ids — skip")
            continue
        X = linq.loc[ids].values.astype(np.float32)
        print(f"  [{level_name}] {len(ids):,} text_ids → X={X.shape}", end=" ")
        t0 = time.time()
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        pca.fit(X)
        var = pca.explained_variance_ratio_.sum()
        print(f"var={var:.3f}  {time.time()-t0:.1f}s")
        pcas[level_name] = pca

    os.makedirs(os.path.dirname(pca_save_path), exist_ok=True)
    with open(pca_save_path, "wb") as fh:
        pickle.dump(pcas, fh)
    print(f"Saved per-level PCAs → {pca_save_path}")
    return pcas


def compute_level_emb(df, level_cols, linq, pca, n_out):
    n_rows = len(df)
    emb_sum   = np.zeros((n_rows, LINQ_DIM), dtype=np.float32)
    emb_count = np.zeros(n_rows, dtype=np.float32)

    for col in level_cols:
        if col not in df.columns:
            continue
        col_s = df[col].reset_index(drop=True)
        valid = col_s.notna()
        row_idx = valid.values.nonzero()[0]
        tids = col_s[valid].tolist()
        if not tids:
            continue
        embs = linq.reindex(tids).values.astype(np.float32)
        ok = ~np.isnan(embs[:, 0])
        np.add.at(emb_sum,   row_idx[ok], embs[ok])
        np.add.at(emb_count, row_idx[ok], 1.0)

    nonzero = emb_count > 0
    raw = np.zeros_like(emb_sum)
    raw[nonzero] = emb_sum[nonzero] / emb_count[nonzero, None]

    result = np.zeros((n_rows, n_out), dtype=np.float32)
    if nonzero.any():
        result[nonzero] = pca.transform(raw[nonzero]).astype(np.float32)
    return result


def process_df(df, linq, pcas, n_out):
    df = df.reset_index(drop=True)
    base_cols = [c for c in PRICE_COLS if c in df.columns]
    base = df[base_cols].copy()
    for level_name, level_cols in LEVELS.items():
        pca = pcas.get(level_name)
        if pca is None:
            # 레벨 PCA 없으면 zero 벡터
            for i in range(n_out):
                base[f"{level_name}_emb{i}"] = 0.0
            continue
        print(f"  {level_name} ...", end=" ", flush=True)
        t0 = time.time()
        emb = compute_level_emb(df, level_cols, linq, pca, n_out)
        arr = np.column_stack([emb[:, i] for i in range(n_out)])
        for i in range(n_out):
            base[f"{level_name}_emb{i}"] = arr[:, i]
        print(f"{time.time()-t0:.1f}s")
    return base


def save_per_ticker(df, split, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ticker, grp in df.groupby("ticker"):
        out_path = os.path.join(out_dir, f"{ticker}_{split}.parquet")
        grp = grp.drop(columns=["ticker"], errors="ignore").reset_index(drop=True)
        grp.to_parquet(out_path, index=False)
    print(f"  saved {df['ticker'].nunique()} ticker files → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=64)
    parser.add_argument("--out_dir", type=str,
                        default="/home/eb/LG_AI/data/linq_fintexts_perlevel")
    parser.add_argument("--pca_save_path", type=str,
                        default="/home/eb/LG_AI/data/pca/linq_pca_perlevel_64d.pkl")
    args = parser.parse_args()

    linq = load_linq(LINQ_PATH)

    print(f"\nFitting per-level PCA ({LINQ_DIM} → {args.n_components}) ...")
    train_df = pd.read_parquet(TRAIN_PATH)
    pcas = fit_level_pcas(linq, train_df, args.n_components, args.pca_save_path)

    for split, path in [("train", TRAIN_PATH), ("test", TEST_PATH)]:
        print(f"\n{'='*40}\nProcessing {split}.parquet ...")
        df = pd.read_parquet(path)
        processed = process_df(df, linq, pcas, args.n_components)
        save_per_ticker(processed, split, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
