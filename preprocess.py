"""
train.parquet / test.parquet의 text_id 컬럼을 BERT 임베딩으로 변환하여
레벨별 mean-pooled 임베딩 컬럼을 가진 per-ticker parquet 파일을 생성.

출력: data/fintexts/{TICKER}.parquet
  - date, open, high, low, close, volume
  - macro_emb0 .. macro_emb383
  - sector_emb0 .. sector_emb383
  - targetCompany_emb0 .. targetCompany_emb383
  - relatedCompany_emb0 .. relatedCompany_emb383
  - filing_emb0 .. filing_emb383
  - lseg_emb0 .. lseg_emb383
"""

import os
import time
import numpy as np
import pandas as pd

# ── 경로 설정 ────────────────────────────────────────────
DATA_DIR = "/home/eb/LG_AI"
OUT_DIR  = os.path.join(DATA_DIR, "data/fintexts")
EMB_DIM  = 384

TRAIN_PATH = os.path.join(DATA_DIR, "train.parquet")
TEST_PATH  = os.path.join(DATA_DIR, "test.parquet")
BERT_PATH  = os.path.join(DATA_DIR, "BERT Text Embeddings.parquet")

os.makedirs(OUT_DIR, exist_ok=True)

# ── 레벨 정의 ────────────────────────────────────────────
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


def load_bert(path: str) -> pd.DataFrame:
    """BERT 임베딩 로드 후 text_id 인덱스로 설정."""
    print("Loading BERT embeddings ...")
    t0 = time.time()
    bert = pd.read_parquet(path)
    bert = bert.set_index("text_id")
    bert.columns = [f"emb_{i}" for i in range(EMB_DIM)]   # 통일: emb_0 ~ emb_383
    print(f"  done ({time.time()-t0:.1f}s)  shape={bert.shape}")
    return bert


def compute_level_emb(df: pd.DataFrame, level_cols: list, bert: pd.DataFrame) -> np.ndarray:
    """
    df의 level_cols에 있는 text_id들을 lookup해 행별 mean pool.
    반환: [len(df), EMB_DIM] float32, 텍스트 없는 행은 0 벡터
    """
    n_rows = len(df)
    emb_sum   = np.zeros((n_rows, EMB_DIM), dtype=np.float32)
    emb_count = np.zeros(n_rows, dtype=np.float32)

    for col in level_cols:
        if col not in df.columns:
            continue
        col_series = df[col].reset_index(drop=True)
        valid_mask = col_series.notna()
        row_idx = valid_mask.values.nonzero()[0]          # 유효한 행 인덱스
        tid_arr = col_series[valid_mask].tolist()

        if not tid_arr:
            continue

        # bert lookup: 없는 text_id는 NaN 행으로 채워짐
        embs = bert.reindex(tid_arr).values.astype(np.float32)  # [k, 384]
        is_valid = ~np.isnan(embs[:, 0])                        # 첫 차원으로 유효성 판별

        valid_row_idx = row_idx[is_valid]
        valid_embs    = embs[is_valid]

        # 벡터화 누산 (np.add.at: 중복 인덱스 허용)
        np.add.at(emb_sum,   valid_row_idx, valid_embs)
        np.add.at(emb_count, valid_row_idx, 1.0)

    nonzero = emb_count > 0
    result = np.zeros_like(emb_sum)
    result[nonzero] = emb_sum[nonzero] / emb_count[nonzero, None]
    return result


def process_df(df: pd.DataFrame, bert: pd.DataFrame) -> pd.DataFrame:
    """
    하나의 parquet(train or test)을 처리해
    레벨별 임베딩 컬럼이 추가된 DataFrame 반환.
    """
    df = df.reset_index(drop=True)
    base = df[PRICE_COLS].copy()

    for level_name, level_cols in LEVELS.items():
        print(f"  computing {level_name} embeddings ...", end=" ", flush=True)
        t0 = time.time()
        emb = compute_level_emb(df, level_cols, bert)
        for i in range(EMB_DIM):
            base[f"{level_name}_emb{i}"] = emb[:, i]
        print(f"{time.time()-t0:.1f}s")

    return base


def save_per_ticker(df: pd.DataFrame, split: str):
    """ticker별로 분리해서 저장."""
    for ticker, grp in df.groupby("ticker"):
        out_path = os.path.join(OUT_DIR, f"{ticker}_{split}.parquet")
        grp = grp.drop(columns=["ticker"]).reset_index(drop=True)
        grp.to_parquet(out_path, index=False)
    print(f"  saved {df['ticker'].nunique()} ticker files → {OUT_DIR}")


def main():
    bert = load_bert(BERT_PATH)

    for split, path in [("train", TRAIN_PATH), ("test", TEST_PATH)]:
        print(f"\n{'='*40}")
        print(f"Processing {split}.parquet ...")
        df = pd.read_parquet(path)
        processed = process_df(df, bert)
        save_per_ticker(processed, split)

    print("\nDone.")


if __name__ == "__main__":
    main()
