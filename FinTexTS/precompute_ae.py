"""
Trains a simple Autoencoder on all BERT text embeddings (857K x 384) and saves
the encoder weights so PatchTSTWithFiLM can use them instead of PCA.

Architecture: 384 -> 256 -> 128 -> 64 -> 128 -> 256 -> 384 (ReLU, no BN)
Saves: ae_encoder.pt  (state_dict of the encoder Sequential)
       ae_encoder_full.pt (full AE state_dict for inspection)

Usage:
  python precompute_ae.py --out_dir /home/eb/LG_AI/data/ae --n_latent 64 --epochs 30
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pyarrow.parquet as pq

def load_bert_embeddings(bert_path: str, device: str) -> torch.Tensor:
    print(f"Loading BERT embeddings from {bert_path} ...")
    t0 = time.time()
    f = pq.ParquetFile(bert_path)
    df = f.read().to_pandas()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = torch.FloatTensor(df[emb_cols].values)  # [N, 384]
    print(f"  Loaded {X.shape[0]:,} x {X.shape[1]}  in {time.time()-t0:.1f}s")
    return X


class TextAutoEncoder(nn.Module):
    def __init__(self, input_dim: int = 384, latent_dim: int = 64):
        super().__init__()
        h1, h2 = 256, 128
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(),
            nn.Linear(h2, h1),         nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X = load_bert_embeddings(args.bert_path, device)

    # Standardise (zero-mean, unit-var) — same split of text_ids used in training
    mu  = X.mean(0)
    sig = X.std(0).clamp(min=1e-8)
    X_norm = (X - mu) / sig

    dataset = TensorDataset(X_norm)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, pin_memory=True)

    model = TextAutoEncoder(input_dim=X.shape[1], latent_dim=args.n_latent).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
            n     += len(xb)
        sched.step()
        avg = total / n
        tag = " ← best" if avg < best_loss else ""
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(args.out_dir, "ae_full.pt"))
        print(f"[Epoch {ep:3d}/{args.epochs}] recon_mse={avg:.6f}{tag}")

    # Save encoder-only state dict + normalisation stats
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "ae_full.pt")))
    enc_sd = {k.replace("encoder.", "", 1): v
              for k, v in model.state_dict().items() if k.startswith("encoder.")}
    torch.save({
        "encoder_state_dict": enc_sd,
        "mu":  mu,
        "sig": sig,
        "input_dim":  X.shape[1],
        "latent_dim": args.n_latent,
    }, os.path.join(args.out_dir, "ae_encoder.pt"))
    print(f"\nSaved encoder → {args.out_dir}/ae_encoder.pt  (best recon_mse={best_loss:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", default="/home/eb/LG_AI/BERT Text Embeddings.parquet")
    parser.add_argument("--out_dir",   default="/home/eb/LG_AI/data/ae")
    parser.add_argument("--n_latent",  type=int,   default=64)
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch_size",type=int,   default=4096)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--gpu",       type=int,   default=3)
    args = parser.parse_args()
    train(args)
