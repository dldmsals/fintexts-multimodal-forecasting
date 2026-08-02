"""
Variational Autoencoder for BERT text embeddings.
Latent space is regularised to N(0,1) via KL divergence.
Saves encoder in same format as ae_encoder.pt so the model can use --ae_path.

Architecture: 384 -> 256 -> 128 -> mu/logvar (64) [reparameterise] -> 128 -> 256 -> 384

Usage:
  python precompute_vae.py --out_dir /home/eb/LG_AI/data/ae --n_latent 64 --epochs 30
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pyarrow.parquet as pq


def load_bert_embeddings(bert_path):
    print(f"Loading BERT embeddings from {bert_path} ...")
    t0 = time.time()
    f = pq.ParquetFile(bert_path)
    df = f.read().to_pandas()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = torch.FloatTensor(df[emb_cols].values)
    print(f"  Loaded {X.shape[0]:,} x {X.shape[1]}  in {time.time()-t0:.1f}s")
    return X


class TextVAE(nn.Module):
    def __init__(self, input_dim=384, latent_dim=64):
        super().__init__()
        h1, h2 = 256, 128
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(h2, latent_dim)
        self.fc_logvar = nn.Linear(h2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(),
            nn.Linear(h2, h1),         nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x):
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encoder_state_dict(self):
        """Returns state dict compatible with _AEEncoder.net (3 Linear layers)."""
        # _AEEncoder.net: Linear(384,256) ReLU Linear(256,128) ReLU Linear(128,64)
        # VAE encoder: encoder_net[0,2] + fc_mu  → same structure
        sd = {}
        for k, v in self.encoder_net.state_dict().items():
            sd[k] = v            # 0.weight, 0.bias, 2.weight, 2.bias
        sd["4.weight"] = self.fc_mu.weight.clone()
        sd["4.bias"]   = self.fc_mu.bias.clone()
        return sd


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X = load_bert_embeddings(args.bert_path)
    mu_data  = X.mean(0)
    sig_data = X.std(0).clamp(min=1e-8)
    X_norm   = (X - mu_data) / sig_data

    dataset = TensorDataset(X_norm)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=4, pin_memory=True)

    model = TextVAE(input_dim=X.shape[1], latent_dim=args.n_latent).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        total_recon, total_kl, n = 0.0, 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon, mu_z, logvar_z = model(xb)
            recon_loss = F.mse_loss(recon, xb)
            # KL: -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp()).mean()
            loss = recon_loss + args.beta * kl_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_recon += recon_loss.item() * len(xb)
            total_kl    += kl_loss.item()    * len(xb)
            n += len(xb)
        sched.step()
        avg_recon = total_recon / n
        avg_kl    = total_kl    / n
        avg_total = avg_recon + args.beta * avg_kl
        tag = " ← best" if avg_total < best_loss else ""
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), os.path.join(args.out_dir, "vae_full.pt"))
        print(f"[Epoch {ep:3d}/{args.epochs}] recon={avg_recon:.6f} kl={avg_kl:.6f}{tag}")

    model.load_state_dict(torch.load(os.path.join(args.out_dir, "vae_full.pt"),
                                     weights_only=False))
    torch.save({
        "encoder_state_dict": model.encoder_state_dict(),
        "mu":  mu_data,
        "sig": sig_data,
        "input_dim":  X.shape[1],
        "latent_dim": args.n_latent,
    }, os.path.join(args.out_dir, "vae_encoder.pt"))
    print(f"\nSaved → {args.out_dir}/vae_encoder.pt  (best loss={best_loss:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", default="/home/eb/LG_AI/BERT Text Embeddings.parquet")
    parser.add_argument("--out_dir",   default="/home/eb/LG_AI/data/ae")
    parser.add_argument("--n_latent",  type=int,   default=64)
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--batch_size",type=int,   default=4096)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--beta",      type=float, default=0.1,
                        help="KL weight (beta-VAE). smaller = more reconstruction focus")
    args = parser.parse_args()
    train(args)
