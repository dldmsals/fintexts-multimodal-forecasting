import argparse
import os
import random
import sys

# layers/, models/ 등을 top-level로 찾을 수 있도록 경로 추가
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from forecasting_task.models.DLinear import Model as DLinear
from forecasting_task.models.PatchTST import Model as PatchTST
from forecasting_task.models.Informer import Model as Informer
from forecasting_task.models.Autoformer import Model as Autoformer
from forecasting_task.models.iTransformer import Model as iTransformer

from forecasting_task.models.Reformer import Model as Reformer
from forecasting_task.models.Crossformer import Model as Crossformer
from forecasting_task.models.Transformer import Model as Transformer
from forecasting_task.models.FiLM import Model as FiLM
from forecasting_task.models.Nonstationary_Transformer import Model as Nonstationary_Transformer
from forecasting_task.models.TSMixer import Model as TSMixer
from forecasting_task.models.TiDE import Model as TiDE
from forecasting_task.models.TextMoE import LevelConditionedMoE, TextRoutedMoE, FinTextBaseline, FinTextBaselineV2, PatchTSTWithPrefix, DLinearWithText, DLinearWithNormProxy, DLinearWithPCAText, PatchTSTWithPCAPrefix, PatchTSTWithCrossAttn, PatchTSTWithSoftGate, PatchTSTWithTextRevIN, DLinearTextOnly, PatchTSTWithFiLM, PatchTSTWithFiLMDeep, PatchTSTWithFiLMLayered

from forecasting_task.data_provider.dataset import get_dataset_dataloader, get_multi_ticker_dataloader

def _parse_used_col_prefixes(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="forecasting_task runner")
    parser.add_argument("--task_name", default="short_term_forecast")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--pred_len", type=int, default=1)   # 하루 종가 예측
    parser.add_argument("--label_len", type=int, default=16)
    
    
    parser.add_argument(
        "--used_col_prefixes",
        type=_parse_used_col_prefixes,
        default=["macro", "sector", "targetCompany", "relatedCompany", "filing", "lseg"],
        help="Comma-separated list, e.g. macro,sector,targetCompany,relatedCompany,filing,lseg",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--root_path", type=str, default="/home/eb/LG_AI/data/")
    parser.add_argument("--data_path", default="fintexts/AAPL_train.parquet")
    parser.add_argument("--multi_ticker", action="store_true",
                        help="Train on all tickers under root_path/fintexts/")
    parser.add_argument("--gap", type=int, default=20,
                        help="Trading days between context end and prediction target (20=4weeks)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="PatchTST")
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--patch_len", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=64)
    
    parser.add_argument("--text_hidden_size", type=int, default=64)
    parser.add_argument("--text_num_layers", type=int, default=1)
    parser.add_argument("--pca_path", type=str, default="/home/eb/LG_AI/data/pca/pca_32d.pkl")
    parser.add_argument("--n_pca", type=int, default=32)
    parser.add_argument("--pool_mode", type=str, default="decay", choices=["decay", "mean", "attn", "last", "level_attn"])
    parser.add_argument("--text_weight", type=float, default=0.1)
    # LevelConditionedMoE 전용 인자
    parser.add_argument("--d_router", type=int, default=64)
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--fusion_mode", type=str, default="dual_branch",
                        choices=["additive", "gating", "cross_attn", "dual_branch"],
                        help="Text-price fusion method for LevelConditionedMoE")
    parser.add_argument("--text_window", type=int, default=None,
                        help="마지막 N일 텍스트만 pooling에 사용 (None=전체 seq_len)")
    parser.add_argument("--ae_path", type=str, default=None,
                        help="AE encoder checkpoint (ae_encoder.pt); 지정시 PCA 대신 사용")
    parser.add_argument("--unfreeze_ae", action="store_true",
                        help="AE encoder weights 전체 fine-tune 허용")
    parser.add_argument("--unfreeze_ae_last_n", type=int, default=0,
                        help="AE encoder 마지막 N개 linear layer만 unfreeze (0=frozen, 1=8K, 2=41K params)")
    parser.add_argument("--ae_lr_scale", type=float, default=1.0,
                        help="AE encoder LR = lr * text_lr_scale * ae_lr_scale (default=1.0)")

    parser.add_argument("--two_stage_warmup", type=int, default=0,
                        help="Phase-1 epochs training price-only; phase-2 freezes price, trains text")
    parser.add_argument("--text_lr_scale", type=float, default=0.01,
                        help="LR multiplier for text params relative to price params (default 0.01)")
    parser.add_argument("--save_ckpt", action="store_true",
                        help="Save best-val model state_dict to logdir/best_model.pt")
    parser.add_argument("--save_every_n", type=int, default=0,
                        help="Save test_io_e{epoch}.pt every N epochs (0=disabled)")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--output_attention", type=bool, default=False)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument("--distil", type=bool, default=True)
    
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/tmp_test",
    )
    
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument("--emb_dim", type=int, default=384,
                        help="Embedding dim stored in parquet (384=BERT, 64=pre-compressed LINQ)")
    parser.add_argument('--no_text', action='store_true',
                        help='Force text_x=None (ablation: same arch, no text contribution)')
    parser.add_argument('--random_text', action='store_true',
                        help='Replace text embeddings with random noise (ablation: semantic content vs. regularization)')
    parser.add_argument('--unfreeze_pca', action='store_true',
                        help='Allow PCA projection weights to be fine-tuned (dim reduction ablation)')
    parser.add_argument('--oracle_text', action='store_true',
                        help='DIAGNOSTIC ONLY (intentional leakage): replace text_x with the true future '
                             'price-direction sign, to upper-bound whether the architecture can use a '
                             'text-channel signal at all. Never use for a real submission/result.')
    parser.add_argument('--oracle_ratio', type=float, default=1.0,
                        help='Fraction of samples per batch that receive the oracle signal; remainder keep '
                             'real text_x (e.g. 0.5/0.8 for oracle mixing experiments). Only used with --oracle_text.')
    parser.add_argument('--oracle_scale', type=float, default=1.0,
                        help='Magnitude of the injected oracle signal (+/- this value broadcast across all '
                             'levels/dims). Increase if the model architecture cannot pick up a +/-1 signal.')
    parser.add_argument('--predict_return', action='store_true',
                        help='Predict day-over-day price change rate instead of the raw (scaled) price level. '
                             'Price level is non-stationary, so test-period extrapolation outside the train '
                             'price range destabilizes price-level prediction; returns are far more stationary. '
                             'Encoder input (seq_price_x) stays in price-level scale; only the target changes.')
    parser.add_argument('--select_metric', type=str, default='val_mse', choices=['val_mse', 'val_dir_acc'],
                        help='Which validation metric to use for best-checkpoint selection / early stopping. '
                             'The xforecast challenge scores submissions with a Hit-Rate-style (direction) '
                             'metric, not MSE, so val_dir_acc may pick a more competition-aligned checkpoint.')

    args = parser.parse_args()
    if args.pca_path and args.pca_path.lower() in ("none", ""):
        args.pca_path = None
    args.enc_in = 4
    args.dec_in = 4
    args.c_out = 4
    args.d_layers = args.e_layers
    args.p_hidden_dims = [128,128]
    args.p_hidden_layers = 2
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TextModel(nn.Module):
    def __init__(self, hidden_size, pred_len, num_layers):
        super(TextModel, self).__init__()
        self.relu = nn.ReLU()
        self.num_layers = num_layers

        if num_layers <= 1:
            self.layers = nn.ModuleList([nn.Linear(384, pred_len)])
        else:
            layers = [nn.Linear(384, hidden_size)]
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Linear(hidden_size, pred_len))
            self.layers = nn.ModuleList(layers)

    def forward(self, price_x, x):
        ori_x = x.clone()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)
        out = x.unsqueeze(-1).repeat(1, 1, price_x.shape[1]) + price_x.unsqueeze(1).repeat(1, x.shape[1], 1)
        valid_mask = (ori_x.abs().sum(1) >= 1e-6).view(-1, 1, 1).float()
        return out * valid_mask

def _prepare_batch(batch, device, pred_len, label_len):
    seq_price_x, seq_price_y, seq_text_x, seq_x_mark, seq_y_mark, _ = batch
    seq_price_x = seq_price_x.float().to(device)
    seq_price_y = seq_price_y.float().to(device)
    seq_text_x = seq_text_x.float().to(device)
    seq_x_mark = seq_x_mark.float().to(device)
    seq_y_mark = seq_y_mark.float().to(device)

    dec_inp = torch.zeros_like(seq_price_y[:, -pred_len:, :]).float().to(device)
    dec_inp = torch.cat([seq_price_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
    ground_truth = seq_price_y[:, -pred_len:, :]
    return seq_price_x, seq_text_x, seq_x_mark, seq_y_mark, dec_inp, ground_truth


def _inverse_transform_batch(dataset, tensor):
    """seq_price_x(입력, 항상 가격 레벨) 역변환용."""
    array = tensor.detach().cpu().numpy()
    batch, length, channels = array.shape
    flat = array.reshape(-1, channels)
    inv = dataset.inverse_transform(flat)
    return inv.reshape(batch, length, channels)


def _inverse_transform_y_batch(dataset, tensor):
    """seq_price_y(타깃) 역변환용. predict_return이면 변화율 스케일러 사용."""
    array = tensor.detach().cpu().numpy()
    batch, length, channels = array.shape
    flat = array.reshape(-1, channels)
    inv = dataset.inverse_transform_y(flat)
    return inv.reshape(batch, length, channels)


def _direction_pnl_metrics(inputs, outputs, gts, close_idx=3, predict_return=False):
    """방향 정확도 + 정규화 공간 롱/숏 P&L.

    [가격 레벨 모드] StandardScaler는 ticker별로 (x-mean)/std 형태의 affine
    변환이라 diff의 부호(sign)는 raw price와 정규화 공간에서 동일하게
    보존된다. 따라서 multi-ticker 합산 ConcatDataset에서도 inverse_transform
    없이 방향 정확도는 그대로 유효하다.

    [변화율(predict_return) 모드] outputs/gts 자체가 이미 (스케일된) 변화율이므로
    last_close와의 차분이 필요 없이 부호만 보면 된다. 일별 수익률의 평균은
    0에 매우 가까워(표준편차 대비) StandardScaler가 적용된 값의 부호도 raw
    변화율의 부호를 사실상 그대로 보존한다 (가격 레벨 모드와 동일한 근사).

    P&L은 정규화 공간 기준 평균 수익으로, ticker간 스케일이 달라 달러 단위
    P&L은 아니지만 전략 비교용 지표로 사용.

    xforecast 챌린지 공식 Hit Rate 정의: "두 종가가 동일한(가격 변화 없는)
    샘플은 평가에서 제외". 여기서도 actual_dir==0(보합)인 샘플은 분모에서
    제외해 공식 채점 방식과 맞춘다.
    """
    if predict_return:
        pred_r = outputs[:, :, close_idx]
        actual_r = gts[:, :, close_idx]
        pred_dir = torch.sign(pred_r)
        actual_dir = torch.sign(actual_r)
        valid = actual_dir != 0
        if valid.sum() == 0:
            return 0.0, 0.0
        direction_acc = (pred_dir[valid] == actual_dir[valid]).float().mean().item()
        pnl = (pred_dir[valid] * actual_r[valid]).mean().item()
        return direction_acc, pnl

    last_close = inputs[:, -1, close_idx : close_idx + 1]  # [B, 1]
    pred_close = outputs[:, :, close_idx]                   # [B, pred_len]
    gt_close = gts[:, :, close_idx]                          # [B, pred_len]

    pred_dir = torch.sign(pred_close - last_close)
    actual_diff = gt_close - last_close
    actual_dir = torch.sign(actual_diff)
    valid = actual_dir != 0
    if valid.sum() == 0:
        return 0.0, 0.0
    pred_dir, actual_dir, actual_diff = pred_dir[valid], actual_dir[valid], actual_diff[valid]

    direction_acc = (pred_dir == actual_dir).float().mean().item()
    pnl = (pred_dir * actual_diff).mean().item()
    return direction_acc, pnl


def _save_pred_vs_actual_plot(outputs_inv, gts_inv, save_path, close_idx=3, max_points=300, predict_return=False):
    pred_close = outputs_inv[:, 0, close_idx]
    gt_close = gts_inv[:, 0, close_idx]
    n = min(len(pred_close), max_points)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_close[:n], label="actual", linewidth=1.2)
    ax.plot(pred_close[:n], label="predicted", linewidth=1.2, linestyle="--")
    ax.set_xlabel("test sample index")
    if predict_return:
        ax.set_ylabel("day-over-day return rate")
        ax.set_title("Predicted vs Actual return rate (best val epoch)")
    else:
        ax.set_ylabel("close price")
        ax.set_title("Prediction vs Actual (best val epoch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _inject_oracle_text(seq_price_x, seq_text_x, ground_truth, oracle_ratio, oracle_scale, close_idx=3, predict_return=False):
    """DIAGNOSTIC ONLY: broadcast the true future price-direction sign into text_x.

    Sanity-check for whether the model's text fusion path can use a signal at all
    (upper bound), not a real result — ground_truth is leaked into the input by design.
    oracle_ratio<1.0 mixes oracle samples with real text in the same batch.
    """
    if predict_return:
        # ground_truth is already a (scaled) return value; no need to diff against last_close.
        oracle_dir = torch.sign(ground_truth[:, -1, close_idx])           # [B]
    else:
        last_close = seq_price_x[:, -1, close_idx]                       # [B]
        gt_close = ground_truth[:, -1, close_idx]                        # [B]
        oracle_dir = torch.sign(gt_close - last_close)                   # [B]
    oracle_val = (
        oracle_dir.view(-1, 1, 1, 1).expand_as(seq_text_x) * oracle_scale
    )
    if oracle_ratio >= 1.0:
        return oracle_val
    mask = (torch.rand(seq_text_x.shape[0], device=seq_text_x.device) < oracle_ratio)
    mask = mask.view(-1, 1, 1, 1)
    return torch.where(mask, oracle_val, seq_text_x)


def _run_epoch(
    dataloader, dataset, model, args, device, optimizer=None, collect_io=False
):
    is_moe = isinstance(model, (LevelConditionedMoE, TextRoutedMoE, FinTextBaseline, FinTextBaselineV2, PatchTSTWithPrefix, DLinearWithText, DLinearWithNormProxy, DLinearWithPCAText, PatchTSTWithPCAPrefix, PatchTSTWithCrossAttn, PatchTSTWithSoftGate, PatchTSTWithTextRevIN, DLinearTextOnly, PatchTSTWithFiLM, PatchTSTWithFiLMDeep, PatchTSTWithFiLMLayered))
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    close_mse_sum = 0.0
    count = 0
    input_chunks = []
    output_chunks = []
    gt_chunks = []

    for batch in dataloader:
        seq_price_x, seq_text_x, seq_x_mark, seq_y_mark, dec_inp, ground_truth = _prepare_batch(
            batch, device, args.pred_len, args.label_len
        )

        if is_train:
            optimizer.zero_grad()

        if is_moe:
            if args.no_text:
                text_input = None
            elif args.random_text:
                text_input = torch.randn_like(seq_text_x)
            elif args.oracle_text:
                text_input = _inject_oracle_text(
                    seq_price_x, seq_text_x, ground_truth, args.oracle_ratio, args.oracle_scale,
                    predict_return=args.predict_return,
                )
            else:
                text_input = seq_text_x
            output, balance_loss = model(
                seq_price_x, seq_x_mark, dec_inp, seq_y_mark, text_x=text_input
            )
        else:
            output = model(seq_price_x, seq_x_mark, dec_inp, seq_y_mark)
            balance_loss = torch.tensor(0.0, device=device)

        mse_loss = torch.mean((output - ground_truth) ** 2)
        mae_loss = torch.mean(torch.abs(output - ground_truth))
        total_loss = mse_loss + balance_loss

        if is_train:
            total_loss.backward()
            optimizer.step()

        if collect_io:
            input_chunks.append(seq_price_x.detach().cpu())
            output_chunks.append(output.detach().cpu())
            gt_chunks.append(ground_truth.detach().cpu())

        # close는 마지막 채널(index 3: open/high/low/close 순)
        close_mse = torch.mean((output[:, :, 3] - ground_truth[:, :, 3]) ** 2)

        batch_size = output.shape[0]
        mse_sum += mse_loss.item() * batch_size
        mae_sum += mae_loss.item() * batch_size
        close_mse_sum += close_mse.item() * batch_size
        count += batch_size

    mse = mse_sum / max(count, 1)
    mae = mae_sum / max(count, 1)
    close_mse = close_mse_sum / max(count, 1)
    if collect_io:
        inputs = torch.cat(input_chunks, dim=0) if input_chunks else None
        outputs = torch.cat(output_chunks, dim=0) if output_chunks else None
        gts = torch.cat(gt_chunks, dim=0) if gt_chunks else None
        return mse, mae, close_mse, inputs, outputs, gts
    return mse, mae, close_mse


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _loader_fn = get_multi_ticker_dataloader if args.multi_ticker else get_dataset_dataloader
    _common = dict(
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        used_col_prefixes=args.used_col_prefixes,
        root_path=args.root_path,
        batch_size=args.batch_size,
        gap=args.gap,
        emb_dim=args.emb_dim,
        predict_return=args.predict_return,
    )
    if not args.multi_ticker:
        _common["data_path"] = args.data_path

    train_dataset, train_dataloader = _loader_fn(flag="train", **_common)
    val_dataset,   val_dataloader   = _loader_fn(flag="val",   **_common)
    test_dataset,  test_dataloader  = _loader_fn(flag="test",  **_common)
    if args.model_type.lower() == "textmoe":
        model = LevelConditionedMoE(
            args,
            text_dim=args.emb_dim,
            fusion_mode=args.fusion_mode,
            text_window=args.text_window,
        )
    elif args.model_type.lower() == "fintextbaseline":
        model = FinTextBaseline(args, text_dim=args.emb_dim)
    elif args.model_type.lower() == "fintextbaselinev2":
        model = FinTextBaselineV2(args, pca_path=args.pca_path, n_pca=args.n_pca)
    elif args.model_type.lower() == "patchtst_crossattn":
        model = PatchTSTWithCrossAttn(args, pca_path=args.pca_path, n_pca=args.n_pca)
    elif args.model_type.lower() == "patchtst_film":
        model = PatchTSTWithFiLM(args, pca_path=args.pca_path, n_pca=args.n_pca,
                                 text_dim=args.emb_dim,
                                 pool_mode=args.pool_mode, text_window=args.text_window,
                                 unfreeze_pca=args.unfreeze_pca,
                                 ae_path=args.ae_path, unfreeze_ae=args.unfreeze_ae,
                                 unfreeze_ae_last_n=args.unfreeze_ae_last_n)
    elif args.model_type.lower() == "patchtst_film_deep":
        model = PatchTSTWithFiLMDeep(args, pca_path=args.pca_path, n_pca=args.n_pca)
    elif args.model_type.lower() == "patchtst_film_layered":
        model = PatchTSTWithFiLMLayered(args, pca_path=args.pca_path, n_pca=args.n_pca,
                                        text_dim=args.emb_dim)
    elif args.model_type.lower() == "patchtst_softgate":
        model = PatchTSTWithSoftGate(args, pca_path=args.pca_path, n_pca=args.n_pca)
    elif args.model_type.lower() == "patchtst_textrevin":
        model = PatchTSTWithTextRevIN(args, pca_path=args.pca_path, n_pca=args.n_pca)
    elif args.model_type.lower() == "prefix":
        model = PatchTSTWithPrefix(args)
    elif args.model_type.lower() == "dlinear_text":
        model = DLinearWithText(args)
    elif args.model_type.lower() == "dlinear_norm":
        model = DLinearWithNormProxy(args)
    elif args.model_type.lower() == "dlinear_pca":
        model = DLinearWithPCAText(args, pca_path=args.pca_path, n_pca=args.n_pca, pool_mode=args.pool_mode)
    elif args.model_type.lower() == "patchtst_pca":
        model = PatchTSTWithPCAPrefix(args, pca_path=args.pca_path, n_pca=args.n_pca, pool_mode=args.pool_mode)
    elif args.model_type.lower() == "text_only":
        model = DLinearTextOnly(args, pca_path=args.pca_path, n_pca=args.n_pca, pool_mode=args.pool_mode)
    elif args.model_type.lower() == "dlinear":
        model = DLinear(args)
    elif args.model_type.lower() == "patchtst":
        model = PatchTST(args)
    elif args.model_type.lower() == "informer":
        model = Informer(args)
    elif args.model_type.lower() == "itransformer":
        model = iTransformer(args)
    elif args.model_type.lower() == "autoformer":
        model = Autoformer(args)
    elif args.model_type.lower() == "reformer":
        model = Reformer(args)
    elif args.model_type.lower() == "crossformer":
        model = Crossformer(args)
    elif args.model_type.lower() == "transformer":
        model = Transformer(args)
    elif args.model_type.lower() == "film":
        model = FiLM(args)
    elif args.model_type.lower() == "nonstationary_transformer":
        model = Nonstationary_Transformer(args)
    elif args.model_type.lower() == "tsmixer":
        model = TSMixer(args)
    elif args.model_type.lower() == "tide":
        model = TiDE(args)
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
    model = model.to(device)

    # Identify text vs price params for two-stage or differential LR
    has_text_split = hasattr(model, 'get_text_params')
    if has_text_split:
        ret = model.get_text_params()
        price_params, text_params = ret[0], ret[1]
        ae_params = ret[2] if len(ret) > 2 else []
    elif hasattr(model, 'text_heads'):
        text_params  = list(model.text_heads.parameters())
        text_id_set  = {id(p) for p in text_params}
        price_params = [p for p in model.parameters() if id(p) not in text_id_set]
        ae_params    = []
    else:
        price_params, text_params, ae_params = list(model.parameters()), [], []

    two_stage = args.two_stage_warmup > 0 and len(text_params) > 0

    if two_stage:
        # Phase-1: price only
        for p in text_params + ae_params:
            p.requires_grad_(False)
        optimizer = torch.optim.Adam(
            [p for p in price_params if p.requires_grad], lr=args.lr)
        print(f"[Two-stage] Phase-1: training price branch for {args.two_stage_warmup} epochs")
    elif text_params:
        param_groups = [
            {"params": price_params, "lr": args.lr},
            {"params": text_params,  "lr": args.lr * args.text_lr_scale},
        ]
        if ae_params:
            ae_lr = args.lr * args.text_lr_scale * getattr(args, "ae_lr_scale", 1.0)
            param_groups.append({"params": ae_params, "lr": ae_lr})
            print(f"[AE unfrozen] ae_lr={ae_lr:.2e}  (text_lr_scale={args.text_lr_scale}, ae_lr_scale={getattr(args, 'ae_lr_scale', 1.0)})")
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf") if args.select_metric == "val_mse" else float("-inf")
    best_test_mse = float("inf")
    best_test_close_mse = float("inf")
    best_test_mae = float("inf")
    best_test_dir_acc = 0.0
    best_test_pnl = 0.0
    best_test_inputs = None
    best_test_outputs = None
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, args.num_epoch + 1):
        # Two-stage: switch to phase-2 at warmup boundary
        if two_stage and epoch == args.two_stage_warmup + 1:
            print(f"[Two-stage] Phase-2: freezing price, training text branch")
            for p in price_params:
                p.requires_grad_(False)
            for p in text_params:
                p.requires_grad_(True)
            optimizer = torch.optim.Adam(
                [p for p in text_params if p.requires_grad], lr=args.lr * 0.1)
            # Disable modality dropout in phase-2: must use text to get gradient
            if hasattr(model, 'modal_drop'):
                model.modal_drop = 0.0
            # Reset patience for phase-2
            best_val = float("inf") if args.select_metric == "val_mse" else float("-inf")
            patience_counter = 0

        train_mse, _, _ = _run_epoch(
            train_dataloader, train_dataset, model, args, device, optimizer=optimizer
        )
        with torch.no_grad():
            val_mse, _, _, val_inputs, val_outputs, val_gts = _run_epoch(
                val_dataloader, val_dataset, model, args, device, collect_io=True
            )
            test_mse, test_mae, test_close_mse, test_inputs, test_outputs, test_gts = _run_epoch(
                test_dataloader, test_dataset, model, args, device, collect_io=True
            )
        test_dir_acc, test_pnl = _direction_pnl_metrics(
            test_inputs, test_outputs, test_gts, predict_return=args.predict_return
        )
        val_dir_acc, val_pnl = _direction_pnl_metrics(
            val_inputs, val_outputs, val_gts, predict_return=args.predict_return
        )
        cur_metric = val_mse if args.select_metric == "val_mse" else val_dir_acc

        epoch_line = (
            f"[Epoch {epoch}] "
            f"train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} "
            f"val_dir_acc={val_dir_acc:.4f} "
            f"test_mse={test_mse:.6f} "
            f"test_close_mse={test_close_mse:.6f} "
            f"test_mae={test_mae:.6f} "
            f"test_dir_acc={test_dir_acc:.4f} "
            f"test_pnl={test_pnl:.6f}"
        )
        if hasattr(model, "film_stats"):
            fs = model.film_stats()
            decay_str = ",".join(f"{d:.3f}" for d in fs["decay"])
            epoch_line += (
                f" | γ_w={fs['gamma_w']:.4f} γ_b={fs['gamma_b_mean']:.4f}±{fs['gamma_b_std']:.4f}"
                f" β_w={fs['beta_w']:.4f} β_b={fs['beta_b_mean']:.4f}±{fs['beta_b_std']:.4f}"
                f" txt_scale={fs['text_out_scale']:.4f} decay=[{decay_str}]"
            )
        print(epoch_line)

        if args.save_every_n > 0 and epoch % args.save_every_n == 0 and test_inputs is not None:
            ep_path = os.path.join(args.logdir, f"test_io_e{epoch}.pt")
            ep_mse = float(torch.mean((test_outputs - test_gts) ** 2).item())
            ep_inv_in = _inverse_transform_batch(test_dataset, test_inputs)
            ep_inv_out = _inverse_transform_y_batch(test_dataset, test_outputs)
            ep_inv_gt = _inverse_transform_y_batch(test_dataset, test_gts)
            torch.save({"inputs": test_inputs, "outputs": test_outputs, "ground_truths": test_gts,
                        "inputs_inv": ep_inv_in, "outputs_inv": ep_inv_out, "ground_truths_inv": ep_inv_gt},
                       ep_path)
            print(f"  [Saved test_io_e{epoch}.pt mse={ep_mse:.6f}]")

        improved = (cur_metric < best_val) if args.select_metric == "val_mse" else (cur_metric > best_val)
        if improved:
            best_val = cur_metric
            best_test_mse = test_mse
            best_test_close_mse = test_close_mse
            best_test_mae = test_mae
            best_test_dir_acc = test_dir_acc
            best_test_pnl = test_pnl
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_test_inputs = test_inputs
            best_test_outputs = test_outputs
            best_test_gts = test_gts
            best_test_inputs_inv = (
                _inverse_transform_batch(test_dataset, test_inputs)
                if test_inputs is not None
                else None
            )
            best_test_outputs_inv = (
                _inverse_transform_y_batch(test_dataset, test_outputs)
                if test_outputs is not None
                else None
            )
            best_test_gts_inv = (
                _inverse_transform_y_batch(test_dataset, test_gts)
                if test_gts is not None
                else None
            )
        else:
            patience_counter += 1
            # Phase-1 of two-stage: no early stopping, run all warmup epochs
            in_phase1 = two_stage and epoch <= args.two_stage_warmup
            if not in_phase1 and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best {args.select_metric}={best_val:.6f})")
                break

    print(
        f"[Best] {args.select_metric}={best_val:.6f} test_mse={best_test_mse:.6f} "
        f"test_close_mse={best_test_close_mse:.6f} test_mae={best_test_mae:.6f} "
        f"test_dir_acc={best_test_dir_acc:.4f} test_pnl={best_test_pnl:.6f}"
    )
    os.makedirs(args.logdir, exist_ok=True)
    if args.save_ckpt and best_model_state is not None:
        torch.save(best_model_state, os.path.join(args.logdir, "best_model.pt"))
    with open(os.path.join(args.logdir, "result.txt"), "w") as f:
        f.write(
            f"best_{args.select_metric}={best_val:.6f}\n"
            f"best_test_mse={best_test_mse:.6f}\n"
            f"best_test_close_mse={best_test_close_mse:.6f}\n"
            f"best_test_mae={best_test_mae:.6f}\n"
            f"best_test_dir_acc={best_test_dir_acc:.4f}\n"
            f"best_test_pnl={best_test_pnl:.6f}\n"
        )
    if best_test_inputs is not None and best_test_outputs is not None:
        torch.save(
            {
                "inputs": best_test_inputs,
                "outputs": best_test_outputs,
                "ground_truths": best_test_gts,
                "inputs_inv": best_test_inputs_inv,
                "outputs_inv": best_test_outputs_inv,
                "ground_truths_inv": best_test_gts_inv,
            },
            os.path.join(args.logdir, "test_io.pt"),
        )
        if best_test_outputs_inv is not None and best_test_gts_inv is not None:
            _save_pred_vs_actual_plot(
                best_test_outputs_inv,
                best_test_gts_inv,
                os.path.join(args.logdir, "pred_vs_actual.png"),
                predict_return=args.predict_return,
            )
            print(f"  [Saved pred_vs_actual.png]")

    # Also save last-epoch test predictions (useful when val/test distributions differ)
    if test_inputs is not None and test_outputs is not None:
        last_outputs_inv = (
            _inverse_transform_y_batch(test_dataset, test_outputs)
            if test_outputs is not None else None
        )
        last_gts_inv = (
            _inverse_transform_y_batch(test_dataset, test_gts)
            if test_gts is not None else None
        )
        last_inputs_inv = (
            _inverse_transform_batch(test_dataset, test_inputs)
            if test_inputs is not None else None
        )
        torch.save(
            {
                "inputs": test_inputs,
                "outputs": test_outputs,
                "ground_truths": test_gts,
                "inputs_inv": last_inputs_inv,
                "outputs_inv": last_outputs_inv,
                "ground_truths_inv": last_gts_inv,
            },
            os.path.join(args.logdir, "test_io_last.pt"),
        )
        last_mse = float(
            torch.mean((test_outputs - test_gts) ** 2).item()
        )
        last_dir_acc, last_pnl = _direction_pnl_metrics(
            test_inputs, test_outputs, test_gts, predict_return=args.predict_return
        )
        print(
            f"[Last epoch test_mse={last_mse:.6f} test_dir_acc={last_dir_acc:.4f} "
            f"test_pnl={last_pnl:.6f}] saved test_io_last.pt"
        )
