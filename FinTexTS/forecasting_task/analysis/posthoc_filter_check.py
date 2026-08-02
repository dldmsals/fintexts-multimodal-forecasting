"""사후 필터링 sanity check.

test_io.pt (run.py가 저장하는 best-epoch 예측/정답)를 불러와서,
실제 가격이 오른 날 / 내린 날로 표본을 나눈 뒤 각 그룹의 MSE/방향 정확도를
따로 계산한다. 모델이 특정 시장 상황(예: 상승장)에서만 잘 맞고 다른 상황에서는
무너지는지 확인하기 위한 진단용 스크립트.

Usage:
    python -m forecasting_task.analysis.posthoc_filter_check logs/<run>/test_io.pt
"""
import sys

import numpy as np
import torch


def _direction_groups(inputs, outputs, gts, close_idx=3):
    last_close = inputs[:, -1, close_idx].numpy()
    pred_close = outputs[:, 0, close_idx].numpy()
    gt_close = gts[:, 0, close_idx].numpy()

    actual_diff = gt_close - last_close
    up_mask = actual_diff > 0
    down_mask = actual_diff < 0
    flat_mask = actual_diff == 0

    pred_dir = np.sign(pred_close - last_close)
    actual_dir = np.sign(actual_diff)

    def _report(name, mask):
        n = int(mask.sum())
        if n == 0:
            print(f"  {name}: n=0")
            return
        mse = float(np.mean((pred_close[mask] - gt_close[mask]) ** 2))
        dir_acc = float(np.mean(pred_dir[mask] == actual_dir[mask]))
        print(f"  {name}: n={n} close_mse={mse:.6f} dir_acc={dir_acc:.4f}")

    print(f"Total samples: {len(actual_diff)}")
    _report("UP days  (actual price rose)", up_mask)
    _report("DOWN days(actual price fell)", down_mask)
    _report("FLAT days(no change)", flat_mask)


def main(path):
    data = torch.load(path, map_location="cpu")
    inputs = data.get("inputs_inv")
    outputs = data.get("outputs_inv")
    gts = data.get("ground_truths_inv")
    if inputs is None:
        inputs, outputs, gts = data["inputs"], data["outputs"], data["ground_truths"]
        print("[warn] inverse-transformed tensors not found, using normalized-space values")

    inputs = torch.as_tensor(inputs)
    outputs = torch.as_tensor(outputs)
    gts = torch.as_tensor(gts)
    _direction_groups(inputs, outputs, gts)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
