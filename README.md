# FinTexTS — Text-Conditioned Multimodal Stock Forecasting (FiLM / TextMoE)

Multimodal forecasting on the **FinTexTS** task (LG AI), which pairs **historical stock prices with financial-news text**. A model must predict weekday closing prices using only information available **up to 4 weeks earlier** — no look-ahead bias (see [`multimodal_dataset_dsl.md`](multimodal_dataset_dsl.md) for the full data spec).

This repo covers the whole pipeline: **text-embedding preprocessing** → **text-conditioned time-series models** (FiLM conditioning, a **Text Mixture-of-Experts**, cross-attention fusion) → **ensembling & analysis**.

## Contributions (this work)

Built on top of the base FinTexTS framework (see *Attribution* below), the modeling work here adds:

- **`FinTexTS/forecasting_task/models/TextMoE.py`** — a **text-conditioned Mixture-of-Experts** over patch-embedded price series, gated by level-wise news embeddings (macro / sector / target- & related-company / filing / lseg). Supports FiLM-style conditioning, cross-attention text fusion, and on-the-fly PCA projection of raw embeddings.
- **`FinTexTS/forecasting_task/models/FiLM.py` + modified `PatchTST.py`, `run.py`, `layers/SelfAttention_Family.py`, `data_provider/dataset.py`** — FiLM conditioning and cross-attention wired into the training/eval entry point.
- **Embedding preprocessing** (repo root):
  - `preprocess.py` — BERT (384-d) → per-ticker, level-wise mean-pooled features → `data/fintexts/{TICKER}.parquet`
  - `preprocess_linq.py` — Linq (4096-d) compressed to 64-d via **per-level PCA** → `data/linq_fintexts/{TICKER}_{split}.parquet` (`--n_components 64`)
  - `FinTexTS/precompute_{pca,dimred,ae,vae}.py` — dimensionality-reduction variants
- **Ensembling & analysis** — `FinTexTS/ensemble_eval.py`, `FinTexTS/analysis_local/`, `FinTexTS/forecasting_task/analysis/`, `FinTexTS/forecasting_task/cluster/`.

## Text-embedding features

News for each `(ticker, date)` carries several **semantic levels** — `macro · sector · targetCompany · relatedCompany · filing · lseg` — which are mapped from `text_id`, mean-pooled per level, and concatenated with price columns (`open, high, low, close, volume`).

## Running

```bash
# 1. build per-ticker text-embedding features
python preprocess.py                       # BERT features
python preprocess_linq.py --n_components 64 # Linq + per-level PCA

# 2. train / evaluate a text-conditioned model
cd FinTexTS/forecasting_task
python run.py --model TextMoE ...          # or FiLM, PatchTST, ...

# 3. ensemble + analysis
python ../ensemble_eval.py
```

## Attribution

- The forecasting framework (`FinTexTS/forecasting_task/`, `make_dataset/`) is based on **[leejaehoon2016/FinTexTS](https://github.com/leejaehoon2016/FinTexTS)**. The baseline model zoo (Autoformer, Crossformer, DLinear, Informer, iTransformer, TimeMixer, TimesNet, …) derives from the **[Time-Series-Library](https://github.com/thuml/Time-Series-Library)** (thuml).
- Raw news originates from the **[FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)** dataset; embeddings reference **[EXAONE-BI](https://huggingface.co/EXAONE-BI)**.
- Original contributions in this repo are the text-conditioning models and pipeline listed under *Contributions*.

> **Data is not tracked.** Price/embedding parquets (1.5–16 GB each), training logs, and checkpoints under `data/`, `logs/`, and `FinTexTS/logs/` are git-ignored — regenerate them from the challenge dataset.
