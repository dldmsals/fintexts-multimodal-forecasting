import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class Dataset_Custom(Dataset):
    """
    경쟁 데이터용 Dataset.

    파일 구조:
      {root_path}/fintexts/{TICKER}_train.parquet  ← train/val
      {root_path}/fintexts/{TICKER}_test.parquet   ← inference

    컬럼: date, open, high, low, close, volume,
          {prefix}_emb0 .. {prefix}_emb383  (per level)

    gap: 4주 look-ahead 제약. context window 끝에서 gap일 후를 예측.
         competition: gap=28, FinTexTS 재현: gap=0
    """

    def __init__(
        self,
        seq_len,
        label_len,
        pred_len,
        flag,
        used_col_prefixes,
        root_path="data/",
        data_path="fintexts/AAPL_train.parquet",
        gap=0,
        emb_dim=384,
        predict_return=False,
    ):
        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.gap = gap
        self.root_path = root_path
        self.data_path = data_path
        self.used_col_prefixes = used_col_prefixes
        self.n_levels = len(used_col_prefixes)
        self.emb_dim  = emb_dim
        self.predict_return = predict_return

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_parquet(os.path.join(self.root_path, self.data_path))
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        # ── 가격 ─────────────────────────────────────────────────────
        pdf_raw = df_raw[["open", "high", "low", "close"]]

        # ── 텍스트: 레벨별 분리 [T, n_levels, emb_dim] ─────────────
        level_arrays = []
        for prefix in self.used_col_prefixes:
            cols = [f"{prefix}_emb{i}" for i in range(self.emb_dim)]
            arr = df_raw[cols].fillna(0).values.astype(np.float32)
            level_arrays.append(arr)
        text_array = np.stack(level_arrays, axis=1)  # [T, n_levels, emb_dim]

        # ── 날짜 피처 ────────────────────────────────────────────────
        ddf = df_raw[["date"]].copy()
        ddf["date"] = pd.to_datetime(ddf["date"])
        ddf["month"]   = ddf["date"].dt.month
        ddf["day"]     = ddf["date"].dt.day
        ddf["weekday"] = ddf["date"].dt.weekday
        stamp = ddf[["month", "day", "weekday"]].values

        # ── Train / Val / Test 날짜 기반 분할 ────────────────────────
        dates = pd.to_datetime(df_raw["date"])
        num_train = int((dates < "2022-01-01").sum())
        num_vali  = int(((dates >= "2022-01-01") & (dates < "2023-01-01")).sum())
        num_test  = int((dates >= "2023-01-01").sum())

        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [
            num_train,
            num_train + num_vali,
            len(df_raw),
        ]

        # val/test 구간이 없으면(train-only 파일) train 구간을 그대로 사용
        if num_vali == 0 and self.set_type == 1:
            border1s[1] = border1s[0]
            border2s[1] = border2s[0]
        if num_test == 0 and self.set_type == 2:
            border1s[2] = border1s[0]
            border2s[2] = border2s[0]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # scaler는 반드시 train 구간(2022년 이전)만으로 fit
        # 전체 파일로 fit하면 test 파일의 2023년 데이터가 포함되어 leakage 발생
        train_end = num_train if num_train > 0 else len(pdf_raw)
        self.scaler.fit(pdf_raw.values[:train_end])
        price_scaled = self.scaler.transform(pdf_raw.values)

        # ── 예측 타깃을 가격 레벨 대신 일별 변화율로 (외삽 방지) ──────
        # 가격 레벨은 비정상(non-stationary) 시계열이라 test 구간이 train 분포
        # 밖으로 표류하면 모델이 한 번도 못 본 값을 출력해야 함. 변화율은
        # 분포가 훨씬 안정적이라 이 외삽 흔들림을 줄여줌. 입력(seq_price_x)은
        # 맥락 정보 유지를 위해 그대로 가격 레벨 스케일을 사용.
        raw = pdf_raw.values
        returns_raw = np.zeros_like(raw)
        returns_raw[1:] = (raw[1:] - raw[:-1]) / np.clip(np.abs(raw[:-1]), 1e-6, None)
        self.return_scaler = StandardScaler()
        self.return_scaler.fit(returns_raw[:train_end])
        return_scaled = self.return_scaler.transform(returns_raw)

        self.data_price_x = price_scaled[border1:border2]
        self.data_text_x  = text_array[border1:border2]
        self.data_price_y = price_scaled[border1:border2]
        self.data_return_y = return_scaled[border1:border2]
        self.data_stamp   = stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end   = s_begin + self.seq_len

        # gap 적용: context 끝에서 gap일 후를 target 시작으로
        r_begin = s_end - self.label_len + self.gap
        r_end   = r_begin + self.label_len + self.pred_len

        seq_price_x = self.data_price_x[s_begin:s_end]
        y_source = self.data_return_y if self.predict_return else self.data_price_y
        seq_price_y = y_source[r_begin:r_end]
        seq_text_x  = self.data_text_x[s_begin:s_end]   # [seq_len, n_levels, 384]
        seq_x_mark  = self.data_stamp[s_begin:s_end]
        seq_y_mark  = self.data_stamp[r_begin:r_end]

        return seq_price_x, seq_price_y, seq_text_x, seq_x_mark, seq_y_mark, index

    def __len__(self):
        # gap이 있으면 그만큼 총 샘플 수 감소
        return len(self.data_price_x) - self.seq_len - self.gap - self.pred_len + 1

    def inverse_transform(self, data):
        """seq_price_x(입력, 항상 가격 레벨) 역변환용."""
        return self.scaler.inverse_transform(data)

    def inverse_transform_y(self, data):
        """seq_price_y(타깃) 역변환용. predict_return이면 변화율 스케일러 사용."""
        if self.predict_return:
            return self.return_scaler.inverse_transform(data)
        return self.scaler.inverse_transform(data)


def get_dataset_dataloader(
    seq_len,
    label_len,
    pred_len,
    flag,
    used_col_prefixes,
    data_path,
    root_path,
    batch_size,
    gap=0,
    emb_dim=384,
    predict_return=False,
):
    dataset = Dataset_Custom(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        flag=flag,
        used_col_prefixes=used_col_prefixes,
        data_path=data_path,
        root_path=root_path,
        gap=gap,
        emb_dim=emb_dim,
        predict_return=predict_return,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(flag == "train"),
        num_workers=1,
        drop_last=(flag == "train"),
    )
    return dataset, data_loader


def get_multi_ticker_dataloader(
    seq_len,
    label_len,
    pred_len,
    flag,
    used_col_prefixes,
    root_path,
    batch_size,
    gap=0,
    emb_dim=384,
    predict_return=False,
):
    """100개 ticker를 각자 정규화 후 하나의 DataLoader로 합침."""
    fintexts_dir = os.path.join(root_path, "fintexts")
    split = "train" if flag in ("train", "val") else "test"
    files = sorted(glob.glob(os.path.join(fintexts_dir, f"*_{split}.parquet")))

    sub_datasets = []
    for f in files:
        rel_path = os.path.relpath(f, root_path)
        try:
            ds = Dataset_Custom(
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                flag=flag,
                used_col_prefixes=used_col_prefixes,
                data_path=rel_path,
                root_path=root_path,
                gap=gap,
                emb_dim=emb_dim,
                predict_return=predict_return,
            )
            if len(ds) > 0:
                sub_datasets.append(ds)
        except Exception:
            pass

    combined = ConcatDataset(sub_datasets)
    data_loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=(flag == "train"),
        num_workers=4,
        drop_last=(flag == "train"),
    )
    # 대표 dataset(첫 번째)을 inverse_transform용으로 반환
    return sub_datasets[0], data_loader
