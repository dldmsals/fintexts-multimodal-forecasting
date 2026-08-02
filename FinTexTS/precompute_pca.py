"""
Train split(2022 이전) 텍스트 임베딩에서 레벨별 PCA 피팅 후 저장.

Usage:
    python precompute_pca.py [--n_components 32] [--levels macro sector ...]
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=32)
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"],
    )
    parser.add_argument("--root_path", default="/home/eb/LG_AI/FinTexTS/data")
    args = parser.parse_args()

    emb_dim = 384
    files = sorted(glob.glob(os.path.join(args.root_path, "fintexts", "*_train.parquet")))
    print(f"Found {len(files)} train parquet files")

    all_embeddings = {lv: [] for lv in args.levels}

    for f in files:
        df = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(df["date"])
        train_mask = dates < "2022-01-01"

        for lv in args.levels:
            cols = [f"{lv}_emb{i}" for i in range(emb_dim)]
            if cols[0] not in df.columns:
                continue
            arr = df.loc[train_mask, cols].fillna(0).values.astype(np.float32)
            nonzero = np.abs(arr).sum(axis=1) > 1e-6
            if nonzero.sum() > 0:
                all_embeddings[lv].append(arr[nonzero])

    os.makedirs(os.path.join(args.root_path, "pca"), exist_ok=True)
    pca_models = {}

    for lv in args.levels:
        if not all_embeddings[lv]:
            print(f"[{lv}] no data — skip")
            continue
        X = np.concatenate(all_embeddings[lv], axis=0)
        n_comp = min(args.n_components, X.shape[0], X.shape[1])
        print(f"[{lv}] {X.shape[0]:,} non-zero samples → PCA(n={n_comp})")

        pca = PCA(n_components=n_comp, whiten=False)
        pca.fit(X)

        cumvar = pca.explained_variance_ratio_.cumsum()
        print(f"  explained variance: {cumvar[-1]:.3f} ({n_comp} components)")
        pca_models[lv] = pca

    out_path = os.path.join(args.root_path, "pca", f"pca_{args.n_components}d.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(pca_models, fh)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
