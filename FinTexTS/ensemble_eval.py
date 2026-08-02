"""
여러 seed로 학습한 모델의 test_io.pt를 불러와 앙상블 평가.
예측값 평균 → MSE/MAE 계산 및 개별 모델과 비교.
"""
import torch
import os
import sys

LOGDIRS = [
    "logs/ablation_ms_64d",      # seed=7  (기존)
    "logs/ensemble_seed42",
    "logs/ensemble_seed123",
    "logs/ensemble_seed2023",
    "logs/ensemble_seed777",
]

all_outputs = []
ground_truths = None

print("Loading model outputs...")
for d in LOGDIRS:
    pt_path = os.path.join(d, "test_io.pt")
    if not os.path.exists(pt_path):
        print(f"  MISSING: {d}")
        continue
    pt = torch.load(pt_path, weights_only=False)
    out = pt["outputs"]   # [N, pred_len, n_vars]
    gt  = pt["ground_truths"]

    if ground_truths is None:
        ground_truths = gt
    else:
        # ground_truths 동일한지 확인
        if not torch.allclose(ground_truths, gt, atol=1e-5):
            print(f"  WARNING: ground_truths mismatch in {d}")

    mse = ((out - gt) ** 2).mean().item()
    mae = out.sub(gt).abs().mean().item()
    seed = d.split("seed")[-1] if "seed" in d else "7"
    print(f"  seed={seed:>4s}  MSE={mse:.6f}  MAE={mae:.6f}")
    all_outputs.append(out)

if len(all_outputs) == 0:
    print("No outputs found.")
    sys.exit(1)

print(f"\nLoaded {len(all_outputs)} models.")

# ── 앙상블: 점진적으로 모델 추가하며 MSE 변화 확인 ──────────
print("\n=== Ensemble MSE as models are added ===")
for k in range(1, len(all_outputs) + 1):
    ensemble = torch.stack(all_outputs[:k]).mean(dim=0)
    mse = ((ensemble - ground_truths) ** 2).mean().item()
    mae = ensemble.sub(ground_truths).abs().mean().item()
    print(f"  {k} model(s):  MSE={mse:.6f}  MAE={mae:.6f}")

# ── 최종 앙상블 ────────────────────────────────────────────
ensemble_final = torch.stack(all_outputs).mean(dim=0)
mse_final = ((ensemble_final - ground_truths) ** 2).mean().item()
mae_final  = ensemble_final.sub(ground_truths).abs().mean().item()
print(f"\n[Final ensemble ({len(all_outputs)} models)]")
print(f"  MSE={mse_final:.6f}  MAE={mae_final:.6f}")
