import pickle
import sys
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from layers.Autoformer_EncDec import series_decomp


class SimpleProjection:
    """Duck-types sklearn PCA: .mean_ and .components_. Shared with precompute_dimred."""
    def __init__(self, mean, components):
        import numpy as np
        self.mean_ = mean.astype(np.float32)
        self.components_ = components.astype(np.float32)

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


class _FixedUnpickler(pickle.Unpickler):
    """Redirects __main__.SimpleProjection → TextMoE.SimpleProjection."""
    def find_class(self, module, name):
        if name == "SimpleProjection":
            return SimpleProjection
        return super().find_class(module, name)


def _load_pca_pkl(path):
    with open(path, "rb") as fh:
        return _FixedUnpickler(fh).load()


def _last_text(text_x):
    """[B, T, n_levels, D] → 레벨별 마지막 non-zero 임베딩 → [B, n_levels, D]"""
    B, T, n_levels, D = text_x.shape
    mask = (text_x.abs().sum(dim=-1) > 1e-6).float()              # [B, T, n_levels]
    t_idx = torch.arange(T, device=text_x.device).float()
    weighted = mask * t_idx.unsqueeze(0).unsqueeze(2)              # [B, T, n_levels]
    last_t = weighted.argmax(dim=1)                                 # [B, n_levels]
    has_any = mask.any(dim=1)                                       # [B, n_levels]
    last_t_exp = last_t.unsqueeze(-1).unsqueeze(-1).expand(B, n_levels, 1, D)
    result = text_x.permute(0, 2, 1, 3).gather(2, last_t_exp).squeeze(2)  # [B, n_levels, D]
    return result * has_any.unsqueeze(-1)


def _pool_text(text_x, text_window=None):
    """[B, T, n_levels, D] → non-zero 날만 레벨별 평균 → [B, n_levels, D]
    text_window: 마지막 N일만 사용 (None이면 전체)
    """
    if text_window is not None:
        text_x = text_x[:, -text_window:, :, :]
    mask = (text_x.abs().sum(dim=-1) > 1e-6).float()          # [B, T, n_levels]
    text_sum   = (text_x * mask.unsqueeze(-1)).sum(dim=1)      # [B, n_levels, D]
    text_count = mask.sum(dim=1).unsqueeze(-1).clamp(min=1)    # [B, n_levels, 1]
    return text_sum / text_count                                # [B, n_levels, D]


class PriceCondTextAttn(nn.Module):
    """
    Price-conditioned temporal attention over text embeddings.

    Query : price 시퀀스의 mean+std → Linear → [B, d_attn]
    Key   : 각 날의 text embedding → Linear → [B, T, n_levels, d_attn]
    Value : text embedding 그대로   [B, T, n_levels, D]

    zero 임베딩 날은 attention에서 제외(-inf masking).
    출력: attention-weighted sum → [B, n_levels, D]
    """

    def __init__(self, text_dim: int, n_vars: int, d_attn: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(n_vars * 2, d_attn, bias=False)
        self.k_proj = nn.Linear(text_dim,   d_attn, bias=False)
        self.scale  = d_attn ** -0.5

    def forward(self, text_x, price_x, text_window=None):
        """
        text_x:  [B, T, n_levels, D]
        price_x: [B, seq_len, n_vars]  (정규화 전 원본)
        """
        if text_window is not None:
            text_x = text_x[:, -text_window:, :, :]

        # Query: price 통계 → [B, d_attn]
        p_mean = price_x.mean(dim=1)                              # [B, n_vars]
        p_std  = price_x.std(dim=1).clamp(min=1e-6)              # [B, n_vars]
        q = self.q_proj(torch.cat([p_mean, p_std], dim=-1))      # [B, d_attn]

        # Key: text → [B, T, n_levels, d_attn]
        k = self.k_proj(text_x)

        # Attention score: [B, T, n_levels, 1]
        scores = (q[:, None, None, :] * k).sum(-1, keepdim=True) * self.scale

        # zero 임베딩 마스킹
        zero_mask = (text_x.abs().sum(-1, keepdim=True) < 1e-6)  # [B, T, n_levels, 1]
        scores = scores.masked_fill(zero_mask, -1e9)

        weights = torch.softmax(scores, dim=1)                    # [B, T, n_levels, 1]
        return (text_x * weights).sum(dim=1)                      # [B, n_levels, D]


class LevelConditionedMoE(nn.Module):
    """
    공유 PatchTST encoder (price branch 1개) +
    레벨별 독립 text branch 4개 → 각 text_pred 합산 → price_pred에 더함.

    구조:
      enc_out → flatten → Linear(nf→pred_len) → price_pred  [B, n_vars, pred_len]

      macro_text   → Linear(384→pred_len) → text_pred_macro   [B, pred_len]
      sector_text  → Linear(384→pred_len) → text_pred_sector  [B, pred_len]
      company_text → Linear(384→pred_len) → text_pred_company [B, pred_len]
      lseg_text    → Linear(384→pred_len) → text_pred_lseg    [B, pred_len]

      output = price_pred + sum(text_preds)   broadcast over n_vars

    level_groups: {'macro': [0], 'sector': [1], 'company': [2,3,4], 'lseg': [5]}
    """

    def __init__(
        self,
        configs,
        text_dim: int = 384,
        fusion_mode: str = "dual_branch",  # kept for CLI compat, ignored
        level_groups: dict = None,
        text_window: int = None,
    ):
        super().__init__()
        self.pred_len    = configs.pred_len
        self.text_window = text_window
        self.predict_return = bool(getattr(configs, "predict_return", False))

        if level_groups is None:
            # used_col_prefixes가 있으면 그걸 기준으로 1레벨=1그룹 자동 구성
            prefixes = getattr(configs, "used_col_prefixes", None)
            if prefixes and len(prefixes) > 0:
                level_groups = {p: [i] for i, p in enumerate(prefixes)}
            else:
                level_groups = {
                    "macro":   [0],
                    "sector":  [1],
                    "company": [2, 3, 4],
                    "lseg":    [5],
                }
        self.level_groups  = level_groups
        self.level_indices = list(level_groups.values())
        self.n_experts     = len(level_groups)
        self.text_dim       = text_dim

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        nf        = configs.d_model * patch_num

        # 공유 price head (1개)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.price_head = nn.Linear(nf, configs.pred_len)

        # 레벨별 독립 text head
        self.text_heads = nn.ModuleList([
            nn.Linear(text_dim, configs.pred_len)
            for _ in self.level_indices
        ])

        # price-conditioned temporal attention for text pooling
        self.text_attn = PriceCondTextAttn(text_dim, configs.enc_in, d_attn=32)

        self.dropout = nn.Dropout(configs.dropout)

    def _encode(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)               # [B, n_vars, d_model, patch_num]
        return enc_out, means, stdev

    def _get_group_text(self, text_agg, indices):
        """[B, n_levels, D] → group indices 평균 → [B, D]"""
        parts = torch.stack([text_agg[:, i, :] for i in indices], dim=1)
        return parts.mean(dim=1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]
        enc_out, means, stdev = self._encode(x_enc)

        # 공유 price prediction
        price_pred = self.dropout(self.price_head(self.flatten(enc_out)))  # [B, n_vars, pred_len]

        if text_x is not None:
            text_agg = _pool_text(text_x, self.text_window)  # [B, n_levels, text_dim]
        else:
            text_agg = torch.zeros(
                B, max(max(v) for v in self.level_indices) + 1, self.text_dim,
                device=x_enc.device
            )

        # 레벨별 text prediction 합산
        text_total = torch.zeros(B, self.pred_len, device=x_enc.device)
        for text_head, indices in zip(self.text_heads, self.level_indices):
            group_text = self._get_group_text(text_agg, indices)           # [B, 384]
            has_text   = (group_text.abs().sum(dim=-1) > 1e-6).float()     # [B]
            text_pred  = self.dropout(text_head(group_text))               # [B, pred_len]
            text_total = text_total + text_pred * has_text.unsqueeze(1)

        dec_out = (price_pred + text_total.unsqueeze(1)).permute(0, 2, 1)  # [B, pred_len, n_vars]

        if not self.predict_return:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)

        return dec_out, torch.tensor(0.0, device=x_enc.device)


TextRoutedMoE = LevelConditionedMoE


class FinTextBaseline(nn.Module):
    """
    FinTexTS 논문 방식 재현 (gap=20 세팅):
      - 64일 context window의 마지막 날 텍스트만 사용
      - 6개 레벨 평균 풀링 → 단일 384-dim 벡터
      - PatchTST price prediction + Linear(384, pred_len) text prediction
      - 두 출력을 더함 (additive)
      - 텍스트 없는 날은 text branch 출력을 0으로 마스킹
    """

    def __init__(self, configs, text_dim: int = 384):
        super().__init__()
        self.pred_len = configs.pred_len
        self.predict_return = bool(getattr(configs, "predict_return", False))

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        patch_num      = int((configs.seq_len - patch_len) / stride + 2)
        nf             = configs.d_model * patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.text_head  = nn.Linear(text_dim, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        """
        text_x: [B, T, n_levels, text_dim]
        논문 방식: 마지막 날(text_x[:, -1, :, :])의 모든 레벨 평균
        """
        B = x_enc.shape[0]

        # PatchTST encoding
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)                 # [B, n_vars, d_model, patch_num]

        price_pred = self.dropout(self.price_head(self.flatten(enc_out)))  # [B, n_vars, pred_len]

        if text_x is not None:
            # 마지막 날 텍스트, 레벨 평균
            last_text = text_x[:, -1, :, :]                   # [B, n_levels, text_dim]
            text_emb  = last_text.mean(dim=1)                  # [B, text_dim]
            has_text  = (text_emb.abs().sum(dim=-1) > 1e-6).float()  # [B]
            text_pred = self.dropout(self.text_head(text_emb)) # [B, pred_len]
            text_pred = text_pred * has_text.unsqueeze(1)
        else:
            text_pred = torch.zeros(B, self.pred_len, device=x_enc.device)

        dec_out = (price_pred + text_pred.unsqueeze(1)).permute(0, 2, 1)  # [B, pred_len, n_vars]

        if not self.predict_return:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)

        return dec_out, torch.tensor(0.0, device=x_enc.device)


class PatchTSTWithPrefix(nn.Module):
    """
    PatchTST + 텍스트 prefix token.

    각 텍스트 레벨(e.g. macro, sector)을 d_model 차원으로 투영해
    patch 시퀀스 앞에 prepend → encoder가 [text₁, text₂, patch₁, ..., patchN]을 함께 처리.
    텍스트가 가격 패턴의 해석 방식을 self-attention으로 직접 조절.
    prediction head는 patch 토큰만 사용 (prefix 토큰 제외).
    """

    def __init__(self, configs, text_dim: int = 384):
        super().__init__()
        self.pred_len = configs.pred_len

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        nf        = configs.d_model * patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        # 레벨별 개별 projection (macro/sector 정보 성격이 다름)
        n_levels = len(getattr(configs, "used_col_prefixes", ["macro", "sector"]))
        self.n_levels = n_levels
        self.prefix_projs = nn.ModuleList([
            nn.Linear(text_dim, configs.d_model) for _ in range(n_levels)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        # Instance normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x     = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x     = x / stdev

        # Patch embedding: [B*n_vars, patch_num, d_model]
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        patch_num = enc_out.shape[1]

        # Prefix tokens from text
        n_prefix = 0
        if text_x is not None:
            text_agg = _pool_text(text_x)                        # [B, n_levels, text_dim]
            has_text = (text_agg.abs().sum(-1) > 1e-6).float()   # [B, n_levels]

            prefix_list = []
            for i, proj in enumerate(self.prefix_projs):
                tok = proj(text_agg[:, i, :])                    # [B, d_model]
                tok = tok * has_text[:, i].unsqueeze(-1)         # zero-mask
                prefix_list.append(tok.unsqueeze(1))             # [B, 1, d_model]

            prefix   = torch.cat(prefix_list, dim=1)             # [B, n_levels, d_model]
            n_prefix = prefix.shape[1]

            # broadcast across variables: [B*n_vars, n_levels, d_model]
            prefix = (prefix.unsqueeze(1)
                           .expand(-1, n_vars, -1, -1)
                           .reshape(B * n_vars, n_prefix, -1))

            enc_out = torch.cat([prefix, enc_out], dim=1)        # [B*n_vars, n_prefix+patch_num, d_model]

        # Transformer encoder
        enc_out, _ = self.encoder(enc_out)

        # prefix 토큰 제거, patch 토큰만 head에 전달
        if n_prefix > 0:
            enc_out = enc_out[:, n_prefix:, :]                   # [B*n_vars, patch_num, d_model]

        # [B, n_vars, d_model, patch_num]
        enc_out = enc_out.reshape(B, n_vars, patch_num, enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.dropout(self.price_head(self.flatten(enc_out)))  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                              # [B, pred_len, n_vars]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)

        return dec_out, torch.tensor(0.0, device=x_enc.device)


class PatchTSTWithPCAPrefix(nn.Module):
    """
    PatchTST + PCA-reduced text prefix tokens.

    PCA (384 → n_pca, frozen) → Linear(n_pca → d_model) → prefix prepend.
    PCA 차원 축소로 overfitting 방지 + prefix 방식으로 attention에 통합.
    """

    def __init__(self, configs, pca_path: str, n_pca: int = 32, text_dim: int = 384, pool_mode: str = "mean"):
        super().__init__()
        self.pred_len  = configs.pred_len
        self.pool_mode = pool_mode

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads,
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.price_head = nn.Linear(configs.d_model * patch_num, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        prefixes = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels = len(prefixes)
        self.prefixes = prefixes

        # Frozen PCA layers per level
        pca_models = _load_pca_pkl(pca_path)

        self.pca_means = nn.ParameterList()
        self.pca_projs = nn.ModuleList()
        for lv in prefixes:
            pca = pca_models.get(lv)
            if pca is not None:
                mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                proj = nn.Linear(text_dim, n_pca, bias=False)
                proj.weight = nn.Parameter(torch.FloatTensor(pca.components_), requires_grad=False)
            else:
                mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                proj = nn.Linear(text_dim, n_pca, bias=False)
                nn.init.zeros_(proj.weight)
                proj.weight.requires_grad = False
            self.pca_means.append(mean)
            self.pca_projs.append(proj)

        # Trainable: n_pca → d_model per level
        self.prefix_projs = nn.ModuleList([
            nn.Linear(n_pca, configs.d_model) for _ in range(self.n_levels)
        ])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        means = x_enc.mean(1, keepdim=True).detach()
        x     = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x     = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        patch_num = enc_out.shape[1]

        n_prefix = 0
        if text_x is not None:
            text_agg = _last_text(text_x) if self.pool_mode == "last" else _pool_text(text_x)
            has_text = (text_agg.abs().sum(-1) > 1e-6).float()        # [B, n_levels]

            prefix_list = []
            for i, (mean, pca_proj, prefix_proj) in enumerate(
                zip(self.pca_means, self.pca_projs, self.prefix_projs)
            ):
                lv_emb  = text_agg[:, i, :]                           # [B, 384]
                lv_pca  = pca_proj(lv_emb - mean.unsqueeze(0))        # [B, n_pca]
                tok     = prefix_proj(lv_pca)                          # [B, d_model]
                tok     = tok * has_text[:, i].unsqueeze(-1)
                prefix_list.append(tok.unsqueeze(1))                   # [B, 1, d_model]

            prefix   = torch.cat(prefix_list, dim=1)                  # [B, n_levels, d_model]
            n_prefix = prefix.shape[1]
            prefix   = (prefix.unsqueeze(1)
                              .expand(-1, n_vars, -1, -1)
                              .reshape(B * n_vars, n_prefix, -1))
            enc_out  = torch.cat([prefix, enc_out], dim=1)

        enc_out, _ = self.encoder(enc_out)

        if n_prefix > 0:
            enc_out = enc_out[:, n_prefix:, :]

        enc_out = enc_out.reshape(B, n_vars, patch_num, enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.dropout(self.price_head(self.flatten(enc_out)))
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)

        return dec_out, torch.tensor(0.0, device=x_enc.device)


class DLinearWithText(nn.Module):
    """
    DLinear (seasonal+trend decomp, linear heads) + additive text heads.

    price branch: seasonal/trend decompose → Linear → price_pred [B, pred_len, n_vars]
    text branch : nonzero-mean pool → level별 Linear(384, pred_len) → text_total [B, pred_len]
    output      : price_pred + text_total.unsqueeze(-1)   (n_vars에 broadcast)

    DLinear는 transformer가 없어 prefix 토큰 불가 → additive가 가장 자연스러운 결합.
    """

    def __init__(self, configs, text_dim: int = 384):
        super().__init__()
        self.pred_len = configs.pred_len
        self.seq_len  = configs.seq_len

        # DLinear price branch
        self.decomp          = series_decomp(configs.moving_avg)
        self.linear_seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.linear_trend    = nn.Linear(configs.seq_len, configs.pred_len)
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / configs.seq_len)
        nn.init.constant_(self.linear_trend.weight,    1.0 / configs.seq_len)

        # Text heads (레벨별 독립) — zero init: 초기엔 순수 DLinear, 점진적으로 text 학습
        prefixes = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels = len(prefixes)
        self.text_heads = nn.ModuleList([
            nn.Linear(text_dim, configs.pred_len) for _ in range(self.n_levels)
        ])
        for head in self.text_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        # DLinear price prediction (no normalization — DLinear works in original scale)
        seasonal, trend = self.decomp(x_enc)                            # [B, seq_len, n_vars]
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1))     # [B, n_vars, pred_len]
        trend    = self.linear_trend(trend.permute(0, 2, 1))
        price_pred = (seasonal + trend).permute(0, 2, 1)               # [B, pred_len, n_vars]

        if text_x is None:
            return price_pred, torch.tensor(0.0, device=x_enc.device)

        # Text: nonzero mean pool → level별 head → 합산
        text_agg   = _pool_text(text_x)                                # [B, n_levels, text_dim]
        has_text   = (text_agg.abs().sum(-1) > 1e-6).float()          # [B, n_levels]
        text_total = torch.zeros(B, self.pred_len, device=x_enc.device)

        for i, head in enumerate(self.text_heads):
            lv_pred    = self.dropout(head(text_agg[:, i, :]))        # [B, pred_len]
            text_total = text_total + lv_pred * has_text[:, i].unsqueeze(1)

        # broadcast text over n_vars
        dec_out = price_pred + text_total.unsqueeze(-1)               # [B, pred_len, n_vars]
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class DLinearWithNormProxy(nn.Module):
    """
    DLinear + embedding-norm time series as proxy text signal.

    Instead of projecting raw 384-dim embeddings (overfits with ~100 tickers),
    uses the scalar L2 norm of each level's embedding over the last `norm_window`
    days.  This gives `norm_window * n_levels` features — much lower-dimensional
    and robust to text sparsity.

    proxy_head: Linear(norm_window * n_levels, pred_len) — zero-initialized.
    """

    def __init__(self, configs, text_dim: int = 384, norm_window: int = 21):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.norm_window = norm_window

        self.decomp          = series_decomp(configs.moving_avg)
        self.linear_seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.linear_trend    = nn.Linear(configs.seq_len, configs.pred_len)
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / configs.seq_len)
        nn.init.constant_(self.linear_trend.weight,    1.0 / configs.seq_len)

        prefixes = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels  = len(prefixes)
        proxy_dim = norm_window * self.n_levels

        self.proxy_head = nn.Linear(proxy_dim, configs.pred_len)
        nn.init.zeros_(self.proxy_head.weight)
        nn.init.zeros_(self.proxy_head.bias)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        seasonal, trend = self.decomp(x_enc)
        seasonal   = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend      = self.linear_trend(trend.permute(0, 2, 1))
        price_pred = (seasonal + trend).permute(0, 2, 1)               # [B, pred_len, n_vars]

        if text_x is None:
            return price_pred, torch.tensor(0.0, device=x_enc.device)

        # Embedding norms over last norm_window days: [B, norm_window, n_levels]
        norms = text_x[:, -self.norm_window:, :, :].norm(dim=-1)      # [B, norm_window, n_levels]
        proxy_input = norms.reshape(B, -1)                              # [B, norm_window * n_levels]

        correction = self.proxy_head(proxy_input)                      # [B, pred_len]
        dec_out = price_pred + correction.unsqueeze(-1)
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class DLinearWithPCAText(nn.Module):
    """
    DLinear + PCA-reduced text embeddings.

    PCA (fitted offline on train split, frozen) reduces 384-dim → n_pca-dim
    per level.  Greatly reduces overfitting compared to full 384-dim heads.

    pca_path : path to pickle produced by precompute_pca.py
    n_pca    : number of PCA components (must match the pickle)
    pool_mode: "mean" (non-zero mean over seq_len) or "last" (most recent non-zero day)
    """

    def __init__(self, configs, pca_path: str, n_pca: int = 32, text_dim: int = 384, pool_mode: str = "mean"):
        super().__init__()
        self.pred_len  = configs.pred_len
        self.n_pca     = n_pca
        self.pool_mode = pool_mode

        # DLinear price branch
        self.decomp          = series_decomp(configs.moving_avg)
        self.linear_seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.linear_trend    = nn.Linear(configs.seq_len, configs.pred_len)
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / configs.seq_len)
        nn.init.constant_(self.linear_trend.weight,    1.0 / configs.seq_len)

        prefixes = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels  = len(prefixes)
        self.prefixes  = prefixes

        # Load PCA models and build frozen projection layers per level
        pca_models = _load_pca_pkl(pca_path)

        self.pca_means = nn.ParameterList()
        self.pca_projs = nn.ModuleList()
        for lv in prefixes:
            pca = pca_models.get(lv)
            if pca is not None:
                mean  = nn.Parameter(torch.FloatTensor(pca.mean_),        requires_grad=False)
                proj  = nn.Linear(text_dim, n_pca, bias=False)
                proj.weight = nn.Parameter(torch.FloatTensor(pca.components_), requires_grad=False)
            else:
                mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                proj = nn.Linear(text_dim, n_pca, bias=False)
                nn.init.zeros_(proj.weight)
                proj.weight.requires_grad = False
            self.pca_means.append(mean)
            self.pca_projs.append(proj)

        # Trainable text heads: PCA-dim → pred_len, zero-initialized
        self.text_heads = nn.ModuleList([
            nn.Linear(n_pca, configs.pred_len) for _ in range(self.n_levels)
        ])
        for head in self.text_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        seasonal, trend = self.decomp(x_enc)
        seasonal   = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend      = self.linear_trend(trend.permute(0, 2, 1))
        price_pred = (seasonal + trend).permute(0, 2, 1)               # [B, pred_len, n_vars]

        if text_x is None:
            return price_pred, torch.tensor(0.0, device=x_enc.device)

        # Nonzero mean pool: [B, n_levels, 384]
        text_agg = _last_text(text_x) if self.pool_mode == "last" else _pool_text(text_x)
        has_text = (text_agg.abs().sum(-1) > 1e-6).float()             # [B, n_levels]

        text_total = torch.zeros(B, self.pred_len, device=x_enc.device)
        for i, (mean, proj, head) in enumerate(zip(self.pca_means, self.pca_projs, self.text_heads)):
            lv_emb   = text_agg[:, i, :]                               # [B, 384]
            lv_pca   = proj(lv_emb - mean.unsqueeze(0))                # [B, n_pca]
            lv_pred  = self.dropout(head(lv_pca))                      # [B, pred_len]
            text_total = text_total + lv_pred * has_text[:, i].unsqueeze(1)

        dec_out = price_pred + text_total.unsqueeze(-1)
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class FinTextBaselineV2(nn.Module):
    """
    Improved FinTextBaseline:
    - Learnable temporal decay pooling per level (instead of last-day only)
    - PCA (frozen, 384→n_pca) per level to reduce overfitting
    - Modality dropout (p=modal_drop) during training for robustness
    """

    def __init__(self, configs, pca_path=None, n_pca: int = 32,
                 text_dim: int = 384, modal_drop: float = 0.3):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.modal_drop = modal_drop

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        patch_num      = int((configs.seq_len - patch_len) / stride + 2)
        nf             = configs.d_model * patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        prefixes       = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels  = len(prefixes)

        # Learnable log-decay per level (init 0 → exp(0)=1 → mild decay → uniform pooling initially)
        self.log_decay = nn.Parameter(torch.zeros(self.n_levels))

        # PCA projection layers (frozen)
        self.use_pca   = (pca_path is not None)
        proj_in        = n_pca if self.use_pca else text_dim
        if self.use_pca:
            pca_models = _load_pca_pkl(pca_path)
            self.pca_means = nn.ParameterList()
            self.pca_projs = nn.ModuleList()
            for lv in prefixes:
                pca = pca_models.get(lv)
                if pca is not None:
                    mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    proj.weight = nn.Parameter(
                        torch.FloatTensor(pca.components_), requires_grad=False)
                else:
                    mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    nn.init.zeros_(proj.weight)
                    proj.weight.requires_grad = False
                self.pca_means.append(mean)
                self.pca_projs.append(proj)

        # Zero-initialized text heads (one per level)
        self.text_heads = nn.ModuleList([
            nn.Linear(proj_in, configs.pred_len) for _ in range(self.n_levels)
        ])
        for h in self.text_heads:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def get_text_params(self):
        """Returns (price_params, text_params) for two-stage training."""
        text_ids = set()
        for m in [self.log_decay] + list(self.text_heads.parameters()):
            text_ids.add(id(m) if isinstance(m, nn.Parameter) else None)
        for h in self.text_heads.parameters():
            text_ids.add(id(h))
        text_ids.add(id(self.log_decay))
        text_params  = [self.log_decay] + list(self.text_heads.parameters())
        price_params = [p for p in self.parameters()
                        if id(p) not in {id(q) for q in text_params}
                        and p.requires_grad]
        return price_params, text_params

    def _decay_pool(self, text_x):
        """[B, T, n_levels, D] → [B, n_levels, D] via per-level exponential decay."""
        B, T, n_levels, D = text_x.shape
        mask = (text_x.abs().sum(-1) > 1e-6).float()               # [B, T, n_levels]

        t_idx      = torch.arange(T, device=text_x.device).float()
        decay_rate = self.log_decay.exp()                            # [n_levels]
        dist       = (T - 1 - t_idx).unsqueeze(1)                   # [T, 1]
        w          = torch.exp(-decay_rate.unsqueeze(0) * dist)      # [T, n_levels]

        w     = w.unsqueeze(0) * mask                                # [B, T, n_levels]
        w     = w / w.sum(1, keepdim=True).clamp(min=1e-8)          # normalize
        return (text_x * w.unsqueeze(-1)).sum(1)                     # [B, n_levels, D]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        # Instance normalization
        means  = x_enc.mean(1, keepdim=True).detach()
        x      = x_enc - means
        stdev  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x      = x / stdev

        # PatchTST encode
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _      = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)                       # [B, n_vars, d_model, patch_num]

        price_pred = self.dropout(self.price_head(self.flatten(enc_out)))  # [B, n_vars, pred_len]

        # Modality dropout
        use_text = (text_x is not None)
        if self.training and use_text and torch.rand(1).item() < self.modal_drop:
            use_text = False

        if use_text:
            text_agg = self._decay_pool(text_x)                     # [B, n_levels, D]
            has_text = (text_agg.abs().sum(-1) > 1e-6).float()      # [B, n_levels]

            text_total = torch.zeros(B, self.pred_len, device=x_enc.device)
            for i, head in enumerate(self.text_heads):
                if self.use_pca:
                    lv_emb = self.pca_projs[i](
                        text_agg[:, i, :] - self.pca_means[i].unsqueeze(0))
                else:
                    lv_emb = text_agg[:, i, :]
                lv_pred    = self.dropout(head(lv_emb))              # [B, pred_len]
                text_total = text_total + lv_pred * has_text[:, i].unsqueeze(1)
        else:
            text_total = torch.zeros(B, self.pred_len, device=x_enc.device)

        dec_out = (price_pred + text_total.unsqueeze(1)).permute(0, 2, 1)  # [B, pred_len, n_vars]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class PatchTSTWithCrossAttn(nn.Module):
    """
    PatchTST + gated cross-attention text fusion (MSGCA-inspired).

    1. PCA (frozen, 384→n_pca) + learnable temporal decay pooling per level
    2. text_proj: [B, n_levels, n_pca] → [B, n_levels, d_model]  (K, V)
    3. Cross-attention: price patches as Q, text tokens as K/V
    4. Gated residual: enc_out = norm(enc_out + sigmoid(gate) * cross_out)
       gate initialized to -3.0 → sigmoid ≈ 0.05 (near-zero at start)
    5. Modality dropout (p=modal_drop) during training
    """

    def __init__(self, configs, pca_path=None, n_pca: int = 32,
                 text_dim: int = 384, modal_drop: float = 0.3):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.modal_drop = modal_drop

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        patch_num      = int((configs.seq_len - patch_len) / stride + 2)
        nf             = configs.d_model * patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        prefixes      = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels = len(prefixes)

        # Learnable temporal decay per level
        self.log_decay = nn.Parameter(torch.zeros(self.n_levels))

        # PCA layers (frozen)
        self.use_pca = (pca_path is not None)
        proj_in      = n_pca if self.use_pca else text_dim
        if self.use_pca:
            pca_models = _load_pca_pkl(pca_path)
            self.pca_means = nn.ParameterList()
            self.pca_projs = nn.ModuleList()
            for lv in prefixes:
                pca = pca_models.get(lv)
                if pca is not None:
                    mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    proj.weight = nn.Parameter(
                        torch.FloatTensor(pca.components_), requires_grad=False)
                else:
                    mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    nn.init.zeros_(proj.weight)
                    proj.weight.requires_grad = False
                self.pca_means.append(mean)
                self.pca_projs.append(proj)

        # Text → d_model projection (Xavier init so text path gets real gradients)
        self.text_proj = nn.Linear(proj_in, configs.d_model)

        # Cross-attention module
        self.cross_attn = nn.MultiheadAttention(
            configs.d_model, configs.n_heads,
            dropout=configs.dropout, batch_first=True)

        # Gate starts at sigmoid(0)=0.5 so text contributes meaningfully at init
        # (sigmoid(-3)≈0.05 was too small; text_proj grad was ~0 and never learned)
        self.text_gate = nn.Parameter(torch.tensor(0.0))
        self.norm      = nn.LayerNorm(configs.d_model)

    def get_text_params(self):
        text_modules = [self.text_proj, self.cross_attn, self.norm]
        text_scalars = [self.log_decay, self.text_gate]
        text_params  = list(sum([list(m.parameters()) for m in text_modules], [])) + text_scalars
        text_id_set  = {id(p) for p in text_params}
        price_params = [p for p in self.parameters()
                        if id(p) not in text_id_set and p.requires_grad]
        return price_params, text_params

    def _decay_pool(self, text_x):
        """[B, T, n_levels, D] → [B, n_levels, D] via per-level exponential decay."""
        B, T, n_levels, D = text_x.shape
        mask = (text_x.abs().sum(-1) > 1e-6).float()               # [B, T, n_levels]

        t_idx      = torch.arange(T, device=text_x.device).float()
        decay_rate = self.log_decay.exp()                            # [n_levels]
        dist       = (T - 1 - t_idx).unsqueeze(1)                   # [T, 1]
        w          = torch.exp(-decay_rate.unsqueeze(0) * dist)      # [T, n_levels]

        w = w.unsqueeze(0) * mask                                    # [B, T, n_levels]
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        return (text_x * w.unsqueeze(-1)).sum(1)                     # [B, n_levels, D]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        # Instance normalization
        means  = x_enc.mean(1, keepdim=True).detach()
        x      = x_enc - means
        stdev  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x      = x / stdev

        # Patch embedding + encoder
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)                    # [B*n_vars, patch_num, d_model]
        enc_out, _      = self.encoder(enc_out)

        # Cross-attention with text
        use_text = (text_x is not None)
        if self.training and use_text and torch.rand(1).item() < self.modal_drop:
            use_text = False

        if use_text:
            text_agg = self._decay_pool(text_x)                      # [B, n_levels, D]
            has_text = (text_agg.abs().sum(-1) > 1e-6).float()       # [B, n_levels]

            if self.use_pca:
                text_pca = torch.stack([
                    self.pca_projs[i](text_agg[:, i, :] - self.pca_means[i].unsqueeze(0))
                    for i in range(self.n_levels)
                ], dim=1)                                             # [B, n_levels, n_pca]
            else:
                text_pca = text_agg                                   # [B, n_levels, text_dim]

            text_kv = self.text_proj(text_pca)                       # [B, n_levels, d_model]
            text_kv = text_kv * has_text.unsqueeze(-1)               # zero-mask missing levels

            # Expand across variables: [B*n_vars, n_levels, d_model]
            text_kv = (text_kv.unsqueeze(1)
                       .expand(-1, n_vars, -1, -1)
                       .reshape(B * n_vars, self.n_levels, -1))

            cross_out, _ = self.cross_attn(enc_out, text_kv, text_kv)  # [B*n_vars, patch_num, d_model]
            gate    = torch.sigmoid(self.text_gate)
            enc_out = self.norm(enc_out + gate * cross_out)

        # Predict
        patch_num = enc_out.shape[1]
        enc_out   = enc_out.reshape(B, n_vars, patch_num, enc_out.shape[-1])
        enc_out   = enc_out.permute(0, 1, 3, 2)                      # [B, n_vars, d_model, patch_num]

        dec_out = self.dropout(self.price_head(self.flatten(enc_out)))  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                            # [B, pred_len, n_vars]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class _AEEncoder(nn.Module):
    """Thin wrapper around the AE encoder so we can load/freeze uniformly."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        h1, h2 = 256, 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),        nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class PatchTSTWithFiLM(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) text fusion.

    Text features predict per-channel gamma/beta that modulate patch representations
    AFTER the PatchTST encoder. Guaranteed gradient flow: no multiplicative gate.

    gamma_init = 1 (identity scale), beta_init = 0 (no shift).
    Both are additively offset from text, so gradient is always active.

    Dimensionality reduction options (mutually exclusive):
      pca_path  → frozen (or unfrozen) PCA linear projection
      ae_path   → pretrained AE encoder (frozen or unfrozen via unfreeze_ae)
    """

    def __init__(self, configs, pca_path=None, n_pca: int = 32,
                 text_dim: int = 384, modal_drop: float = 0.3,
                 pool_mode: str = 'decay', text_window: int = None,
                 unfreeze_pca: bool = False,
                 ae_path: str = None, unfreeze_ae: bool = False,
                 unfreeze_ae_last_n: int = 0):
        super().__init__()
        self.pred_len    = configs.pred_len
        self.modal_drop  = modal_drop
        self.pool_mode   = pool_mode    # 'decay' | 'mean' | 'attn'
        self.text_window = text_window  # None = full seq_len
        # predict_return: 타깃이 가격 레벨이 아니라 변화율이라, 입력 가격
        # 윈도우의 mean/std로 출력을 de-normalize하면 단위가 안 맞아 망가짐.
        # 이 경우 최종 de-norm 단계를 건너뛴다 (입력 정규화는 그대로 유지).
        self.predict_return = bool(getattr(configs, "predict_return", False))

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        patch_num      = int((configs.seq_len - patch_len) / stride + 2)
        nf             = configs.d_model * patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        prefixes      = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels = len(prefixes)

        self.log_decay = nn.Parameter(torch.zeros(self.n_levels))

        # per-level learned temporal attention (pool_mode='level_attn')
        if pool_mode == 'level_attn':
            self.level_attn_logits = nn.Parameter(
                torch.zeros(self.n_levels, configs.seq_len))

        # ── dimensionality reduction ──────────────────────────────────────
        self.use_pca = (pca_path is not None) and (ae_path is None)
        self.use_ae  = (ae_path  is not None)

        if self.use_ae:
            ckpt      = torch.load(ae_path, map_location="cpu", weights_only=False)
            latent    = ckpt["latent_dim"]
            inp_dim   = ckpt.get("input_dim", text_dim)
            ae_mu     = ckpt["mu"]   # [text_dim]
            ae_sig    = ckpt["sig"]  # [text_dim]
            self.ae_mu  = nn.Parameter(ae_mu,  requires_grad=False)
            self.ae_sig = nn.Parameter(ae_sig, requires_grad=False)
            # shared AE encoder across all levels (same BERT space)
            self.ae_enc = _AEEncoder(inp_dim, latent)
            self.ae_enc.net.load_state_dict(ckpt["encoder_state_dict"])
            # freeze all first, then selectively unfreeze
            for p in self.ae_enc.parameters():
                p.requires_grad = False
            if unfreeze_ae:
                # full unfreeze
                for p in self.ae_enc.parameters():
                    p.requires_grad = True
            elif unfreeze_ae_last_n > 0:
                # partial unfreeze: last N linear layers only
                # net = [Linear, ReLU, Linear, ReLU, Linear]  →  linears at idx 0,2,4
                linears = [m for m in self.ae_enc.net if isinstance(m, nn.Linear)]
                for lin in linears[-unfreeze_ae_last_n:]:
                    for p in lin.parameters():
                        p.requires_grad = True
                n_trainable = sum(p.numel() for p in self.ae_enc.parameters() if p.requires_grad)
                print(f"[AE partial unfreeze] last {unfreeze_ae_last_n} linear(s), {n_trainable:,} trainable params")
            proj_in = latent

        elif self.use_pca:
            pca_models = _load_pca_pkl(pca_path)
            self.pca_means = nn.ParameterList()
            self.pca_projs = nn.ModuleList()
            for lv in prefixes:
                pca = pca_models.get(lv)
                if pca is not None:
                    mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    proj.weight = nn.Parameter(
                        torch.FloatTensor(pca.components_), requires_grad=unfreeze_pca)
                else:
                    mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    nn.init.zeros_(proj.weight)
                    proj.weight.requires_grad = unfreeze_pca
                self.pca_means.append(mean)
                self.pca_projs.append(proj)
            proj_in = n_pca

        else:
            proj_in = text_dim

        feat_dim = self.n_levels * proj_in

        # FiLM: gamma (scale) and beta (shift) from text
        # gamma starts at 1 (identity: no modulation), beta at 0
        self.film_gamma = nn.Linear(feat_dim, configs.d_model)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)   # gamma = 1 at init → identity

        self.film_beta = nn.Linear(feat_dim, configs.d_model)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)   # beta = 0 at init → no shift

        # Additive output correction from text (bypasses price head entirely)
        self.text_out = nn.Linear(feat_dim, configs.pred_len)
        nn.init.zeros_(self.text_out.weight)
        nn.init.zeros_(self.text_out.bias)

        self.text_out_scale = nn.Parameter(torch.tensor(-3.0))  # small init

        # Attention pooling: learned scalar score per timestep per level
        if pool_mode == 'attn':
            self.attn_pool_w = nn.Linear(text_dim, 1, bias=False)

    def get_text_params(self, ae_lr_scale: float = 1.0):
        """
        Returns (price_params, text_params, ae_params).
        ae_params is non-empty only when AE is unfrozen; they get a separate
        LR group so ae_lr_scale can be applied independently.
        """
        text_modules = [self.film_gamma, self.film_beta, self.text_out]
        if self.pool_mode == 'attn':
            text_modules.append(self.attn_pool_w)
        text_scalars = [self.log_decay, self.text_out_scale]
        if self.pool_mode == 'level_attn':
            text_scalars.append(self.level_attn_logits)
        text_params  = list(sum([list(m.parameters()) for m in text_modules], [])) + text_scalars

        ae_params = []
        if self.use_ae:
            ae_params = [p for p in self.ae_enc.parameters() if p.requires_grad]

        all_text_ids = {id(p) for p in text_params + ae_params}
        price_params = [p for p in self.parameters()
                        if id(p) not in all_text_ids and p.requires_grad]
        return price_params, text_params, ae_params

    def film_stats(self):
        """Return a dict of FiLM parameter stats for logging each epoch."""
        with torch.no_grad():
            gw_norm  = self.film_gamma.weight.norm().item()
            gb_mean  = self.film_gamma.bias.mean().item()   # init=1
            gb_std   = self.film_gamma.bias.std().item()
            bw_norm  = self.film_beta.weight.norm().item()
            bb_mean  = self.film_beta.bias.mean().item()    # init=0
            bb_std   = self.film_beta.bias.std().item()
            tw_norm  = self.text_out.weight.norm().item()
            tscale   = torch.sigmoid(self.text_out_scale).item()
            decay    = self.log_decay.exp().tolist()
        return dict(
            gamma_w=gw_norm, gamma_b_mean=gb_mean, gamma_b_std=gb_std,
            beta_w=bw_norm,  beta_b_mean=bb_mean,  beta_b_std=bb_std,
            text_out_w=tw_norm, text_out_scale=tscale,
            decay=decay,
        )

    def _pool_text(self, text_x):
        # text_window 슬라이싱
        if self.text_window is not None:
            text_x = text_x[:, -self.text_window:, :, :]
        B, T, n_levels, D = text_x.shape
        mask = (text_x.abs().sum(-1) > 1e-6).float()  # [B, T, n_levels]

        if self.pool_mode == 'mean':
            w = mask / mask.sum(1, keepdim=True).clamp(min=1e-8)

        elif self.pool_mode == 'attn':
            # [B, T, n_levels, 1] → [B, T, n_levels]
            scores = self.attn_pool_w(text_x).squeeze(-1)
            scores = scores.masked_fill(mask < 0.5, float('-inf'))
            w = torch.softmax(scores, dim=1)
            w = torch.nan_to_num(w, nan=0.0)

        elif self.pool_mode == 'level_attn':
            # per-level position-based attention: level_attn_logits [n_levels, seq_len]
            logits = self.level_attn_logits[:, :T]          # [n_levels, T]
            logits = logits.t().unsqueeze(0).expand(B, -1, -1)  # [B, T, n_levels]
            logits = logits.masked_fill(mask < 0.5, float('-inf'))
            w = torch.softmax(logits, dim=1)
            w = torch.nan_to_num(w, nan=0.0)

        else:  # 'decay' (default)
            t_idx      = torch.arange(T, device=text_x.device).float()
            decay_rate = self.log_decay.exp()
            dist       = (T - 1 - t_idx).unsqueeze(1)
            w          = torch.exp(-decay_rate.unsqueeze(0) * dist)
            w = w.unsqueeze(0) * mask
            w = w / w.sum(1, keepdim=True).clamp(min=1e-8)

        return (text_x * w.unsqueeze(-1)).sum(1)  # [B, n_levels, D]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        means  = x_enc.mean(1, keepdim=True).detach()
        x      = x_enc - means
        stdev  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x      = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _      = self.encoder(enc_out)

        use_text = (text_x is not None)
        if self.training and use_text and torch.rand(1).item() < self.modal_drop:
            use_text = False

        text_corr = None
        if use_text:
            text_agg = self._pool_text(text_x)                           # [B, n_levels, D]
            has_text = (text_agg.abs().sum(-1) > 1e-6).float()           # [B, n_levels]

            if self.use_ae:
                x_norm   = (text_agg - self.ae_mu) / self.ae_sig         # normalise
                text_pca = self.ae_enc(x_norm.reshape(-1, text_agg.shape[-1])).reshape(
                    B, self.n_levels, -1)                                 # [B, n_levels, latent]
            elif self.use_pca:
                text_pca = torch.stack([
                    self.pca_projs[i](text_agg[:, i, :] - self.pca_means[i].unsqueeze(0))
                    for i in range(self.n_levels)
                ], dim=1)
            else:
                text_pca = text_agg

            text_pca = text_pca * has_text.unsqueeze(-1)
            text_feat = text_pca.reshape(B, -1)                          # [B, n_levels*proj_in]

            # FiLM modulation of encoder output
            gamma = self.film_gamma(text_feat)                           # [B, d_model]
            beta  = self.film_beta(text_feat)                            # [B, d_model]

            gamma_exp = (gamma.unsqueeze(1)
                         .expand(-1, n_vars, -1)
                         .reshape(B * n_vars, 1, -1))                   # [B*n_vars, 1, d_model]
            beta_exp  = (beta.unsqueeze(1)
                         .expand(-1, n_vars, -1)
                         .reshape(B * n_vars, 1, -1))

            enc_out = gamma_exp * enc_out + beta_exp                     # [B*n_vars, patch_num, d_model]

            # Direct output correction from text
            scale = torch.sigmoid(self.text_out_scale)
            text_corr = scale * self.text_out(text_feat)                 # [B, pred_len]

        patch_num = enc_out.shape[1]
        enc_out   = enc_out.reshape(B, n_vars, patch_num, enc_out.shape[-1])
        enc_out   = enc_out.permute(0, 1, 3, 2)
        dec_out   = self.dropout(self.price_head(self.flatten(enc_out))) # [B, n_vars, pred_len]
        dec_out   = dec_out.permute(0, 2, 1)                            # [B, pred_len, n_vars]

        if text_corr is not None:
            dec_out = dec_out + text_corr.unsqueeze(-1)

        if not self.predict_return:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class PatchTSTWithFiLMDeep(nn.Module):
    """
    FiLM applied after EACH encoder layer (not just at the output).
    Text modulates attention features at every depth, allowing it to
    influence what the subsequent layers attend to.

    gamma_i init=1, beta_i init=0 per layer → identity at init.
    """

    def __init__(self, configs, pca_path=None, n_pca: int = 32,
                 text_dim: int = 384, modal_drop: float = 0.3):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.modal_drop = modal_drop
        self.e_layers   = configs.e_layers

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        patch_num       = int((configs.seq_len - patch_len) / stride + 2)
        nf              = configs.d_model * patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        prefixes      = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels = len(prefixes)
        self.log_decay = nn.Parameter(torch.zeros(self.n_levels))

        self.use_pca = (pca_path is not None)
        proj_in      = n_pca if self.use_pca else text_dim
        if self.use_pca:
            pca_models = _load_pca_pkl(pca_path)
            self.pca_means = nn.ParameterList()
            self.pca_projs = nn.ModuleList()
            for lv in prefixes:
                pca = pca_models.get(lv)
                if pca is not None:
                    mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    proj.weight = nn.Parameter(
                        torch.FloatTensor(pca.components_), requires_grad=False)
                else:
                    mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    nn.init.zeros_(proj.weight)
                    proj.weight.requires_grad = False
                self.pca_means.append(mean)
                self.pca_projs.append(proj)

        feat_dim = self.n_levels * proj_in

        # Per-layer FiLM: one gamma/beta MLP per encoder layer
        self.film_gammas = nn.ModuleList()
        self.film_betas  = nn.ModuleList()
        for _ in range(configs.e_layers):
            g = nn.Linear(feat_dim, configs.d_model)
            nn.init.zeros_(g.weight)
            nn.init.ones_(g.bias)   # gamma = 1 at init
            b = nn.Linear(feat_dim, configs.d_model)
            nn.init.zeros_(b.weight)
            nn.init.zeros_(b.bias)  # beta = 0 at init
            self.film_gammas.append(g)
            self.film_betas.append(b)

        self.text_out = nn.Linear(feat_dim, configs.pred_len)
        nn.init.zeros_(self.text_out.weight)
        nn.init.zeros_(self.text_out.bias)
        self.text_out_scale = nn.Parameter(torch.tensor(-3.0))

    def get_text_params(self):
        text_modules = list(self.film_gammas) + list(self.film_betas) + [self.text_out]
        text_scalars = [self.log_decay, self.text_out_scale]
        text_params  = list(sum([list(m.parameters()) for m in text_modules], [])) + text_scalars
        text_id_set  = {id(p) for p in text_params}
        price_params = [p for p in self.parameters()
                        if id(p) not in text_id_set and p.requires_grad]
        return price_params, text_params

    def _decay_pool(self, text_x):
        B, T, n_levels, D = text_x.shape
        mask = (text_x.abs().sum(-1) > 1e-6).float()
        t_idx      = torch.arange(T, device=text_x.device).float()
        decay_rate = self.log_decay.exp()
        dist       = (T - 1 - t_idx).unsqueeze(1)
        w          = torch.exp(-decay_rate.unsqueeze(0) * dist)
        w = w.unsqueeze(0) * mask
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        return (text_x * w.unsqueeze(-1)).sum(1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        means  = x_enc.mean(1, keepdim=True).detach()
        x      = x_enc - means
        stdev  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x      = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)

        use_text = (text_x is not None)
        if self.training and use_text and torch.rand(1).item() < self.modal_drop:
            use_text = False

        # Precompute text features once
        text_feat = None
        text_corr = None
        if use_text:
            text_agg = self._decay_pool(text_x)
            has_text = (text_agg.abs().sum(-1) > 1e-6).float()
            if self.use_pca:
                text_pca = torch.stack([
                    self.pca_projs[i](text_agg[:, i, :] - self.pca_means[i].unsqueeze(0))
                    for i in range(self.n_levels)
                ], dim=1)
            else:
                text_pca = text_agg
            text_pca  = text_pca * has_text.unsqueeze(-1)
            text_feat = text_pca.reshape(B, -1)   # [B, n_levels*proj_in]

            scale     = torch.sigmoid(self.text_out_scale)
            text_corr = scale * self.text_out(text_feat)

        # Run encoder layer by layer, applying FiLM after each
        for i, attn_layer in enumerate(self.encoder.attn_layers):
            enc_out, _ = attn_layer(enc_out)
            if use_text:
                gamma = self.film_gammas[i](text_feat)                        # [B, d_model]
                beta  = self.film_betas[i](text_feat)                         # [B, d_model]
                gamma_exp = (gamma.unsqueeze(1)
                             .expand(-1, n_vars, -1)
                             .reshape(B * n_vars, 1, -1))
                beta_exp  = (beta.unsqueeze(1)
                             .expand(-1, n_vars, -1)
                             .reshape(B * n_vars, 1, -1))
                enc_out = gamma_exp * enc_out + beta_exp

        if self.encoder.norm is not None:
            enc_out = self.encoder.norm(enc_out)

        patch_num = enc_out.shape[1]
        enc_out   = enc_out.reshape(B, n_vars, patch_num, enc_out.shape[-1])
        enc_out   = enc_out.permute(0, 1, 3, 2)
        dec_out   = self.dropout(self.price_head(self.flatten(enc_out)))
        dec_out   = dec_out.permute(0, 2, 1)

        if text_corr is not None:
            dec_out = dec_out + text_corr.unsqueeze(-1)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class PatchTSTWithFiLMLayered(nn.Module):
    """
    FiLM applied inside each encoder layer with *different* text level subsets per layer.

    Levels are split evenly across e_layers in order:
      e_layers=2, 4 levels → layer0: [0,1], layer1: [2,3]
      e_layers=2, 6 levels → layer0: [0,1,2], layer1: [3,4,5]

    This allows lower layers to receive broad market signals (macro/sector)
    and higher layers to receive company-specific signals (relatedCompany/filing).
    """

    def __init__(self, configs, pca_path=None, n_pca: int = 32,
                 text_dim: int = 384, modal_drop: float = 0.3):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.modal_drop = modal_drop
        self.e_layers   = configs.e_layers

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        patch_num       = int((configs.seq_len - patch_len) / stride + 2)
        nf              = configs.d_model * patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        prefixes      = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels = len(prefixes)
        self.log_decay = nn.Parameter(torch.zeros(self.n_levels))

        self.use_pca = (pca_path is not None)
        proj_in      = n_pca if self.use_pca else text_dim
        if self.use_pca:
            pca_models = _load_pca_pkl(pca_path)
            self.pca_means = nn.ParameterList()
            self.pca_projs = nn.ModuleList()
            for lv in prefixes:
                pca = pca_models.get(lv)
                if pca is not None:
                    mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    proj.weight = nn.Parameter(
                        torch.FloatTensor(pca.components_), requires_grad=False)
                else:
                    mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    nn.init.zeros_(proj.weight)
                    proj.weight.requires_grad = False
                self.pca_means.append(mean)
                self.pca_projs.append(proj)

        # Split levels evenly across layers
        # level_assignment[i] = list of level indices for layer i
        base  = self.n_levels // self.e_layers
        extra = self.n_levels  % self.e_layers
        self.level_assignment = []
        start = 0
        for i in range(self.e_layers):
            cnt = base + (1 if i < extra else 0)
            self.level_assignment.append(list(range(start, start + cnt)))
            start += cnt

        # Per-layer FiLM conditioned on assigned levels only
        self.film_gammas = nn.ModuleList()
        self.film_betas  = nn.ModuleList()
        for i in range(self.e_layers):
            n_assigned = len(self.level_assignment[i])
            feat = n_assigned * proj_in
            g = nn.Linear(feat, configs.d_model)
            nn.init.zeros_(g.weight); nn.init.ones_(g.bias)
            b = nn.Linear(feat, configs.d_model)
            nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)
            self.film_gammas.append(g)
            self.film_betas.append(b)

        # Direct correction uses all levels
        all_feat = self.n_levels * proj_in
        self.text_out = nn.Linear(all_feat, configs.pred_len)
        nn.init.zeros_(self.text_out.weight); nn.init.zeros_(self.text_out.bias)
        self.text_out_scale = nn.Parameter(torch.tensor(-3.0))

    def get_text_params(self):
        text_modules = list(self.film_gammas) + list(self.film_betas) + [self.text_out]
        text_scalars = [self.log_decay, self.text_out_scale]
        text_params  = list(sum([list(m.parameters()) for m in text_modules], [])) + text_scalars
        text_id_set  = {id(p) for p in text_params}
        price_params = [p for p in self.parameters()
                        if id(p) not in text_id_set and p.requires_grad]
        return price_params, text_params

    def _pool_text(self, text_x):
        B, T, n_levels, D = text_x.shape
        mask  = (text_x.abs().sum(-1) > 1e-6).float()
        t_idx = torch.arange(T, device=text_x.device).float()
        decay = self.log_decay.exp()
        dist  = (T - 1 - t_idx).unsqueeze(1)
        w     = torch.exp(-decay.unsqueeze(0) * dist)
        w     = w.unsqueeze(0) * mask
        w     = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        return (text_x * w.unsqueeze(-1)).sum(1)  # [B, n_levels, D]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x     = (x_enc - means) / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)

        use_text = (text_x is not None)
        if self.training and use_text and torch.rand(1).item() < self.modal_drop:
            use_text = False

        text_pca = None
        if use_text:
            text_agg = self._pool_text(text_x)                          # [B, n_levels, D]
            has_text = (text_agg.abs().sum(-1) > 1e-6).float()          # [B, n_levels]
            if self.use_pca:
                text_pca = torch.stack([
                    self.pca_projs[i](text_agg[:, i, :] - self.pca_means[i].unsqueeze(0))
                    for i in range(self.n_levels)
                ], dim=1)                                                # [B, n_levels, proj_in]
            else:
                text_pca = text_agg
            text_pca = text_pca * has_text.unsqueeze(-1)                # zero-out missing

        # Encoder layer-by-layer with per-layer FiLM
        for i, attn_layer in enumerate(self.encoder.attn_layers):
            enc_out, _ = attn_layer(enc_out)
            if use_text:
                lvs      = self.level_assignment[i]
                sub_feat = text_pca[:, lvs, :].reshape(B, -1)           # [B, n_assigned*proj_in]
                gamma    = self.film_gammas[i](sub_feat)                 # [B, d_model]
                beta     = self.film_betas[i](sub_feat)
                gamma_e  = gamma.unsqueeze(1).expand(-1, n_vars, -1).reshape(B * n_vars, 1, -1)
                beta_e   = beta.unsqueeze(1).expand(-1, n_vars, -1).reshape(B * n_vars, 1, -1)
                enc_out  = gamma_e * enc_out + beta_e

        if self.encoder.norm is not None:
            enc_out = self.encoder.norm(enc_out)

        patch_num = enc_out.shape[1]
        enc_out   = enc_out.reshape(B, n_vars, patch_num, enc_out.shape[-1])
        enc_out   = enc_out.permute(0, 1, 3, 2)
        dec_out   = self.dropout(self.price_head(self.flatten(enc_out)))
        dec_out   = dec_out.permute(0, 2, 1)

        if use_text:
            all_feat = text_pca.reshape(B, -1)
            scale    = torch.sigmoid(self.text_out_scale)
            dec_out  = dec_out + (scale * self.text_out(all_feat)).unsqueeze(-1)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class DLinearTextOnly(nn.Module):
    """
    텍스트만 사용하는 예측 모델 (가격 branch 없음).
    ablation: 텍스트 단독 예측력 측정용.

    text → pool → PCA(frozen) → Linear(n_pca, pred_len) → [B, pred_len, n_vars] broadcast
    """

    def __init__(self, configs, pca_path: str, n_pca: int = 32,
                 pool_mode: str = "last", text_dim: int = 384):
        super().__init__()
        self.pred_len  = configs.pred_len
        self.pool_mode = pool_mode

        prefixes = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels = len(prefixes)

        pca_models = _load_pca_pkl(pca_path)

        self.pca_means = nn.ParameterList()
        self.pca_projs = nn.ModuleList()
        for lv in prefixes:
            pca = pca_models.get(lv)
            if pca is not None:
                mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                proj = nn.Linear(text_dim, n_pca, bias=False)
                proj.weight = nn.Parameter(torch.FloatTensor(pca.components_), requires_grad=False)
            else:
                mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                proj = nn.Linear(text_dim, n_pca, bias=False)
                nn.init.zeros_(proj.weight)
                proj.weight.requires_grad = False
            self.pca_means.append(mean)
            self.pca_projs.append(proj)

        self.text_heads = nn.ModuleList([
            nn.Linear(n_pca, configs.pred_len) for _ in range(self.n_levels)
        ])
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B, _, n_vars = x_enc.shape

        if text_x is None:
            return torch.zeros(B, self.pred_len, n_vars, device=x_enc.device), \
                   torch.tensor(0.0, device=x_enc.device)

        text_agg  = _last_text(text_x) if self.pool_mode == "last" else _pool_text(text_x)
        has_text  = (text_agg.abs().sum(-1) > 1e-6).float()

        text_total = torch.zeros(B, self.pred_len, device=x_enc.device)
        for i, (mean, proj, head) in enumerate(zip(self.pca_means, self.pca_projs, self.text_heads)):
            lv_pca  = proj(text_agg[:, i, :] - mean.unsqueeze(0))
            lv_pred = self.dropout(head(lv_pca))
            text_total = text_total + lv_pred * has_text[:, i].unsqueeze(1)

        dec_out = text_total.unsqueeze(-1).expand(B, self.pred_len, n_vars).contiguous()
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class PatchTSTWithSoftGate(nn.Module):
    """
    MASTER-inspired soft gating: text reweights price patch importance.

    Instead of additive text correction, text derives a soft attention mask
    over price patches. This tells the model WHICH historical price patches
    are most relevant given the current news context.

    Architecture:
    1. PCA (frozen) + temporal decay pooling over all levels → [B, n_levels*n_pca]
    2. gate_proj: Linear(n_levels*n_pca, patch_num) → softmax → [B, patch_num]
    3. enc_out *= gate (broadcast over d_model and n_vars)
    4. Additive text correction (zero-init, small) on top

    The gate is initialized to uniform (1/patch_num * patch_num = 1.0) so at
    start the model behaves like vanilla PatchTST. Text gradually learns to
    reweight patches.
    """

    def __init__(self, configs, pca_path=None, n_pca: int = 32,
                 text_dim: int = 384, modal_drop: float = 0.3):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.modal_drop = modal_drop

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        nf              = configs.d_model * self.patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        prefixes      = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels = len(prefixes)

        # Learnable temporal decay per level
        self.log_decay = nn.Parameter(torch.zeros(self.n_levels))

        # PCA layers (frozen)
        self.use_pca = (pca_path is not None)
        proj_in      = n_pca if self.use_pca else text_dim
        if self.use_pca:
            pca_models = _load_pca_pkl(pca_path)
            self.pca_means = nn.ParameterList()
            self.pca_projs = nn.ModuleList()
            for lv in prefixes:
                pca = pca_models.get(lv)
                if pca is not None:
                    mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    proj.weight = nn.Parameter(
                        torch.FloatTensor(pca.components_), requires_grad=False)
                else:
                    mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    nn.init.zeros_(proj.weight)
                    proj.weight.requires_grad = False
                self.pca_means.append(mean)
                self.pca_projs.append(proj)

        # Soft gate: concatenated PCA features → patch attention weights
        gate_in = self.n_levels * proj_in
        self.gate_proj = nn.Linear(gate_in, self.patch_num)
        # Init: uniform gate → log(1/patch_num) before softmax
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

        # Small additive text correction on top of gated price pred (zero-init)
        self.text_head = nn.Linear(gate_in, configs.pred_len)
        nn.init.zeros_(self.text_head.weight)
        nn.init.zeros_(self.text_head.bias)

        self.gate_scale = nn.Parameter(torch.tensor(0.0))  # learned mixing weight

    def get_text_params(self):
        text_modules = [self.gate_proj, self.text_head]
        text_scalars = [self.log_decay, self.gate_scale]
        text_params  = list(sum([list(m.parameters()) for m in text_modules], [])) + text_scalars
        text_id_set  = {id(p) for p in text_params}
        price_params = [p for p in self.parameters()
                        if id(p) not in text_id_set and p.requires_grad]
        return price_params, text_params

    def _decay_pool(self, text_x):
        B, T, n_levels, D = text_x.shape
        mask = (text_x.abs().sum(-1) > 1e-6).float()
        t_idx      = torch.arange(T, device=text_x.device).float()
        decay_rate = self.log_decay.exp()
        dist       = (T - 1 - t_idx).unsqueeze(1)
        w          = torch.exp(-decay_rate.unsqueeze(0) * dist)
        w = w.unsqueeze(0) * mask
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        return (text_x * w.unsqueeze(-1)).sum(1)              # [B, n_levels, D]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        means  = x_enc.mean(1, keepdim=True).detach()
        x      = x_enc - means
        stdev  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x      = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)             # [B*n_vars, patch_num, d_model]
        enc_out, _      = self.encoder(enc_out)

        use_text = (text_x is not None)
        if self.training and use_text and torch.rand(1).item() < self.modal_drop:
            use_text = False

        if use_text:
            text_agg = self._decay_pool(text_x)               # [B, n_levels, D]
            has_any  = (text_agg.abs().sum(dim=(-1, -2)) > 1e-6).float()  # [B]

            if self.use_pca:
                text_pca = torch.stack([
                    self.pca_projs[i](text_agg[:, i, :] - self.pca_means[i].unsqueeze(0))
                    for i in range(self.n_levels)
                ], dim=1)                                      # [B, n_levels, n_pca]
            else:
                text_pca = text_agg

            text_feat = text_pca.reshape(B, -1)               # [B, n_levels*n_pca]

            # Soft gate over patches: [B, patch_num] (uniform when gate_proj=0)
            gate_logits = self.gate_proj(text_feat)            # [B, patch_num]
            gate = torch.softmax(gate_logits, dim=-1) * self.patch_num  # scale back to ~1
            gate = gate * has_any.unsqueeze(1) + (1 - has_any.unsqueeze(1))  # fallback=1 when no text

            # Apply gate across n_vars: [B*n_vars, patch_num, 1]
            gate_exp = (gate.unsqueeze(1)
                        .expand(-1, n_vars, -1)
                        .reshape(B * n_vars, self.patch_num, 1))

            gate_mix = torch.sigmoid(self.gate_scale)
            enc_out  = enc_out * (1 - gate_mix + gate_mix * gate_exp)

            # Additive correction
            text_corr = self.text_head(text_feat)              # [B, pred_len]
            text_corr = text_corr * has_any.unsqueeze(1)
        else:
            text_corr = None

        patch_num = enc_out.shape[1]
        enc_out   = enc_out.reshape(B, n_vars, patch_num, enc_out.shape[-1])
        enc_out   = enc_out.permute(0, 1, 3, 2)
        dec_out   = self.dropout(self.price_head(self.flatten(enc_out)))  # [B, n_vars, pred_len]
        dec_out   = dec_out.permute(0, 2, 1)                  # [B, pred_len, n_vars]

        if text_corr is not None:
            dec_out = dec_out + text_corr.unsqueeze(-1)

        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        return dec_out, torch.tensor(0.0, device=x_enc.device)


class PatchTSTWithTextRevIN(nn.Module):
    """
    PatchTST + text-conditioned denormalization bias.

    Key insight: scaler is fit on 2019-2021 data. In 2023 (bull market),
    prices systematically deviate from the trained norm. Text captures this
    regime shift and corrects the denormalization bias in original price space.

    mean_bias = text → Linear(feat_dim, pred_len), zero-init → no effect initially.
    Correction applied AFTER denorm → acts as regime-level price adjustment.
    """

    def __init__(self, configs, pca_path=None, n_pca: int = 32,
                 text_dim: int = 384, modal_drop: float = 0.3):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.modal_drop = modal_drop

        patch_len = getattr(configs, "patch_len", 16)
        stride    = patch_len // 2
        padding   = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=configs.activation,
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        patch_num       = int((configs.seq_len - patch_len) / stride + 2)
        nf              = configs.d_model * patch_num
        self.price_head = nn.Linear(nf, configs.pred_len)
        self.flatten    = nn.Flatten(start_dim=-2)
        self.dropout    = nn.Dropout(configs.dropout)

        prefixes       = getattr(configs, "used_col_prefixes", ["macro", "sector"])
        self.n_levels  = len(prefixes)
        self.log_decay = nn.Parameter(torch.zeros(self.n_levels))

        self.use_pca = (pca_path is not None)
        proj_in      = n_pca if self.use_pca else text_dim
        if self.use_pca:
            pca_models = _load_pca_pkl(pca_path)
            self.pca_means = nn.ParameterList()
            self.pca_projs = nn.ModuleList()
            for lv in prefixes:
                pca = pca_models.get(lv)
                if pca is not None:
                    mean = nn.Parameter(torch.FloatTensor(pca.mean_), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    proj.weight = nn.Parameter(
                        torch.FloatTensor(pca.components_), requires_grad=False)
                else:
                    mean = nn.Parameter(torch.zeros(text_dim), requires_grad=False)
                    proj = nn.Linear(text_dim, n_pca, bias=False)
                    nn.init.zeros_(proj.weight)
                    proj.weight.requires_grad = False
                self.pca_means.append(mean)
                self.pca_projs.append(proj)

        feat_dim = self.n_levels * proj_in
        self.text_mean_bias = nn.Linear(feat_dim, configs.pred_len)
        nn.init.zeros_(self.text_mean_bias.weight)
        nn.init.zeros_(self.text_mean_bias.bias)

    def get_text_params(self):
        text_params  = [self.log_decay] + list(self.text_mean_bias.parameters())
        text_id_set  = {id(p) for p in text_params}
        price_params = [p for p in self.parameters()
                        if id(p) not in text_id_set and p.requires_grad]
        return price_params, text_params

    def _decay_pool(self, text_x):
        B, T, n_levels, D = text_x.shape
        mask       = (text_x.abs().sum(-1) > 1e-6).float()
        t_idx      = torch.arange(T, device=text_x.device).float()
        decay_rate = self.log_decay.exp()
        dist       = (T - 1 - t_idx).unsqueeze(1)
        w          = torch.exp(-decay_rate.unsqueeze(0) * dist)
        w = w.unsqueeze(0) * mask
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        return (text_x * w.unsqueeze(-1)).sum(1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text_x=None):
        B = x_enc.shape[0]

        means  = x_enc.mean(1, keepdim=True).detach()
        x      = x_enc - means
        stdev  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x      = x / stdev

        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _      = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)

        price_pred = self.dropout(self.price_head(self.flatten(enc_out)))
        price_pred = price_pred.permute(0, 2, 1)                           # [B, pred_len, n_vars]

        # Standard denorm
        dec_out = price_pred * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)

        # Text correction in original price space
        use_text = (text_x is not None)
        if self.training and use_text and torch.rand(1).item() < self.modal_drop:
            use_text = False

        if use_text:
            text_agg = self._decay_pool(text_x)
            has_any  = (text_agg.abs().sum(dim=(-1, -2)) > 1e-6).float()

            if self.use_pca:
                text_pca = torch.stack([
                    self.pca_projs[i](text_agg[:, i, :] - self.pca_means[i].unsqueeze(0))
                    for i in range(self.n_levels)
                ], dim=1)
            else:
                text_pca = text_agg

            text_feat = text_pca.reshape(B, -1)
            mean_bias = self.text_mean_bias(text_feat) * has_any.unsqueeze(1)
            dec_out   = dec_out + mean_bias.unsqueeze(-1)

        return dec_out, torch.tensor(0.0, device=x_enc.device)
