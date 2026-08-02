"""
PCA-compatible dimensionality reduction precomputation.
Saves each method as {level: obj} where obj has .mean_ and .components_,
identical interface to sklearn PCA — so the existing model code works unchanged.

Methods:
  ica    - FastICA (unsupervised, independent components)
  random - Gaussian random projection (no training, J-L guarantee)
  pls    - Partial Least Squares (supervised: maximises text-price covariance)

Usage:
  python precompute_dimred.py --method ica   --n_components 64
  python precompute_dimred.py --method random --n_components 64
  python precompute_dimred.py --method pls   --n_components 64
"""
import argparse, glob, os, pickle, time
import numpy as np
import pandas as pd


LEVELS = ["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"]
EMB_DIM = 384


# ── data loading ──────────────────────────────────────────────────────────────

def load_train_embeddings(root_path, levels):
    """Returns {level: X[N, 384]} using rows before 2022-01-01 (non-zero only)."""
    files = sorted(glob.glob(os.path.join(root_path, "fintexts", "*_train.parquet")))
    print(f"Found {len(files)} train parquets")
    Xs = {lv: [] for lv in levels}
    ys_dict = {lv: [] for lv in levels}   # next-day log-return aligned to each embedding row

    for f in files:
        df = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
        dates = pd.to_datetime(df["date"])
        train_mask = (dates < "2022-01-01").values

        # next-day log-return (shifted close)
        close = df["close"].values.astype(np.float32)
        ret = np.zeros(len(df), dtype=np.float32)
        ret[:-1] = np.log(close[1:] / close[:-1].clip(1e-8))
        # last row has no next-day → keep 0, will be dropped via mask below

        for lv in levels:
            cols = [f"{lv}_emb{i}" for i in range(EMB_DIM)]
            if cols[0] not in df.columns:
                continue
            arr = df[cols].fillna(0).values.astype(np.float32)
            nonzero = (np.abs(arr).sum(axis=1) > 1e-6) & train_mask
            # exclude last row (no return)
            nonzero[-1] = False
            if nonzero.sum() == 0:
                continue
            Xs[lv].append(arr[nonzero])
            ys_dict[lv].append(ret[nonzero])

    result_X, result_y = {}, {}
    for lv in levels:
        if Xs[lv]:
            result_X[lv] = np.concatenate(Xs[lv], axis=0)
            result_y[lv] = np.concatenate(ys_dict[lv], axis=0)
    return result_X, result_y


# ── projection wrappers with .mean_ / .components_ ───────────────────────────

class SimpleProjection:
    """Duck-types sklearn PCA: has .mean_ and .components_."""
    def __init__(self, mean, components):
        self.mean_ = mean.astype(np.float32)              # [D]
        self.components_ = components.astype(np.float32)  # [n, D]

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


# ── ICA ───────────────────────────────────────────────────────────────────────

def fit_ica(X, n_components, seed=42):
    from sklearn.decomposition import FastICA
    print(f"  Fitting FastICA(n={n_components}) on {X.shape[0]:,} samples ...")
    t0 = time.time()
    ica = FastICA(n_components=n_components, random_state=seed, max_iter=1000, tol=1e-4)
    ica.fit(X)
    print(f"  done in {time.time()-t0:.1f}s")
    # ICA already exposes .mean_ and .components_ [n, D]
    return SimpleProjection(ica.mean_, ica.components_)


# ── Random Projection ─────────────────────────────────────────────────────────

def fit_random(X, n_components, seed=42):
    rng = np.random.RandomState(seed)
    D = X.shape[1]
    R = rng.randn(n_components, D).astype(np.float32)
    # Normalise rows so each column of the projection has unit expected norm
    R /= np.sqrt(D)
    mean = X.mean(axis=0)   # centre the data (same as PCA)
    print(f"  Random projection: R shape {R.shape}, mean centred")
    return SimpleProjection(mean, R)


# ── PLS ───────────────────────────────────────────────────────────────────────

def fit_pls(X, y, n_components):
    from sklearn.cross_decomposition import PLSRegression
    print(f"  Fitting PLS(n={n_components}) on {X.shape[0]:,} samples ...")
    t0 = time.time()
    pls = PLSRegression(n_components=n_components, scale=True, max_iter=500)
    pls.fit(X, y)
    print(f"  done in {time.time()-t0:.1f}s")
    # PLSRegression.transform(X) = (X - x_mean) @ x_rotations_
    # sklearn >= 1.1 renamed x_mean_ → _x_mean
    mean = pls._x_mean if hasattr(pls, '_x_mean') else pls.x_mean_
    components = pls.x_rotations_.T   # [n_components, D]
    return SimpleProjection(mean, components)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ica", "random", "pls"], required=True)
    parser.add_argument("--n_components", type=int, default=64)
    parser.add_argument("--root_path", default="/home/eb/LG_AI/FinTexTS/data")
    parser.add_argument("--levels", nargs="+", default=["macro", "sector", "relatedCompany", "filing"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.join(args.root_path, "pca")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Method: {args.method.upper()}, n_components={args.n_components} ===")
    X_dict, y_dict = load_train_embeddings(args.root_path, args.levels)

    models = {}
    for lv in args.levels:
        if lv not in X_dict:
            print(f"[{lv}] no data — skip")
            continue
        X = X_dict[lv]
        print(f"\n[{lv}] {X.shape[0]:,} samples")

        if args.method == "ica":
            models[lv] = fit_ica(X, args.n_components, seed=args.seed)
        elif args.method == "random":
            models[lv] = fit_random(X, args.n_components, seed=args.seed)
        elif args.method == "pls":
            y = y_dict[lv]
            models[lv] = fit_pls(X, y, args.n_components)

    out_path = os.path.join(out_dir, f"{args.method}_{args.n_components}d.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(models, fh)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
