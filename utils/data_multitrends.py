import hashlib
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler
ImageFile.LOAD_TRUNCATED_IMAGES = True

_WEEK_COLS = [str(i) for i in range(12)]
_TEMPORAL_COLS = ["day", "week", "month", "year"]


def _tensor_for_dataloader(t: torch.Tensor) -> torch.Tensor:
    """
    default_collate 会对 batch 做 storage.resize_；部分来源（pandas/numpy 视图、torchvision、pin_memory）
    产生的张量存储不可 resize。用 empty_like + copy_ 得到独立、可堆叠的副本。
    """
    out = torch.empty_like(t, dtype=t.dtype, device=t.device, memory_format=torch.contiguous_format)
    out.copy_(t)
    return out


def collate_lazy_zeroshot_batch(batch):
    """
    LazyZeroShotDataset 返回 8 个 Tensor（含 recent_sales_2w）。不用 default_collate，改为逐字段 torch.stack，
    避免 PyTorch 2.x collate_tensor_fn 里 storage.resize_ 在 worker 中报
    RuntimeError: Trying to resize storage that is not resizable。
    """
    if not batch:
        raise ValueError("empty batch")
    n_fields = len(batch[0])
    stacked = []
    for i in range(n_fields):
        xs = [b[i] for b in batch]
        if not all(isinstance(x, torch.Tensor) for x in xs):
            raise TypeError(f"collate: field {i} must be all torch.Tensor")
        stacked.append(torch.stack(xs, dim=0))
    return tuple(stacked)


def _aux_temporal_scalar(row: pd.Series) -> float:
    """第 5 维：优先 restock；否则 extra（数值或哈希）；无则 0。与 DummyEmbedder 5 路输入对齐。"""
    if "restock" in row.index and pd.notna(row.get("restock", np.nan)):
        try:
            return float(row["restock"])
        except (TypeError, ValueError):
            pass
    if "extra" in row.index and pd.notna(row.get("extra", np.nan)):
        v = row["extra"]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        h = int(hashlib.md5(str(v).encode()).hexdigest()[:8], 16)
        return (h % 10000) / 10000.0
    return 0.0


def _aux_series_from_df(data: pd.DataFrame) -> np.ndarray:
    n = len(data)
    if "restock" in data.columns:
        return pd.to_numeric(data["restock"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if "extra" in data.columns:

        def _enc(x):
            if pd.isna(x):
                return 0.0
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return float(x)
            h = int(hashlib.md5(str(x).encode()).hexdigest()[:8], 16)
            return (h % 10000) / 10000.0

        return data["extra"].map(_enc).to_numpy(dtype=np.float32)
    return np.zeros(n, dtype=np.float32)


def _fixed_norm_trend(values: np.ndarray, trend_len: int) -> np.ndarray:
    """
    将单条趋势序列处理成固定长度 trend_len，并做 MinMax 归一化。
    - 超长：保留最近 trend_len 个点
    - 过短：左侧用首值补齐（空序列则补 0）
    """
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[-trend_len:]
    if arr.size == 0:
        arr = np.zeros(trend_len, dtype=np.float32)
    elif arr.size < trend_len:
        pad_val = float(arr[0])
        pad = np.full(trend_len - arr.size, pad_val, dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    arr = MinMaxScaler().fit_transform(arr.reshape(-1, 1)).reshape(-1).astype(np.float32)
    return arr


def _item_sales_from_row(row: pd.Series) -> torch.Tensor:
    """
    销量 12 维：
    - visuelle2 等：列名为 '0'..'11'（通常在表尾）
    - 旧版：drop 后前 12 列为周销量（见 train.csv 根目录格式）
    """
    if all(c in row.index for c in _WEEK_COLS):
        return torch.tensor(row[_WEEK_COLS].values.astype(np.float32), dtype=torch.float32)
    r_drop = row.to_frame().T
    r_drop.drop(
        ["external_code", "season", "release_date", "image_path"],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    return torch.tensor(r_drop.iloc[0, :12].values.astype(np.float32), dtype=torch.float32)


def _recent_sales_2w_from_row(row: pd.Series) -> torch.Tensor:
    """
    与 Visuelle 2-10 / item_sales 列语义一致：取 12 周序列中第 0、1 周（预测窗口紧前两周），形状 (2,)。
    """
    if all(c in row.index for c in _WEEK_COLS):
        return torch.tensor(
            [float(row["0"]), float(row["1"])], dtype=torch.float32
        )
    r_drop = row.to_frame().T
    r_drop.drop(
        ["external_code", "season", "release_date", "image_path"],
        axis=1,
        inplace=True,
        errors="ignore",
    )
    return torch.tensor(r_drop.iloc[0, :2].values.astype(np.float32), dtype=torch.float32)


def _temporal_from_row(row: pd.Series) -> torch.Tensor:
    """5 维：day, week, month, year + aux（restock / extra / 0）。DummyEmbedder 需 [:, 4]。"""
    aux = _aux_temporal_scalar(row)
    if all(c in row.index for c in _TEMPORAL_COLS):
        base = [float(row[c]) for c in _TEMPORAL_COLS]
        base.append(aux)
        return torch.tensor(base, dtype=torch.float32)
    dt = pd.to_datetime(row["release_date"])
    day = float(dt.dayofweek) / 6.0
    week = float(dt.isocalendar()[1]) / 52.0
    month = float(dt.month) / 12.0
    year = float(dt.year) / 2020.0
    return torch.tensor([day, week, month, year, aux], dtype=torch.float32)


class LazyZeroShotDataset(Dataset):
    """
    与 preprocess_data() 单样本逻辑一致，但在 __getitem__ 中按需计算 gtrends + 图像，
    避免一次性把所有样本缓存在内存（适合全量 train.csv 训练 / 导出 embedding）。
    """

    def __init__(self, data_df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len):
        self.data_df = data_df.reset_index(drop=True)
        self.img_root = img_root
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_transforms = Compose(
            [
                Resize((256, 256)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        cat, col, fab = row["category"], row["color"], row["fabric"]
        start_date = row["release_date"]
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.to_datetime(start_date)
        img_path = row["image_path"]

        gtrend_start = start_date - pd.DateOffset(weeks=52)
        cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat].values
        col_gtrend = self.gtrends.loc[gtrend_start:start_date][col].values
        fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab].values

        cat_gtrend = _fixed_norm_trend(cat_gtrend, self.trend_len)
        col_gtrend = _fixed_norm_trend(col_gtrend, self.trend_len)
        fab_gtrend = _fixed_norm_trend(fab_gtrend, self.trend_len)
        multitrends = np.vstack([cat_gtrend, col_gtrend, fab_gtrend])

        img = Image.open(os.path.join(self.img_root, img_path)).convert("RGB")
        image_features = self.img_transforms(img)

        item_sales = _item_sales_from_row(row)
        recent_sales_2w = _recent_sales_2w_from_row(row)
        temporal_features = _temporal_from_row(row)
        categories = torch.tensor(self.cat_dict[cat], dtype=torch.long)
        colors = torch.tensor(self.col_dict[col], dtype=torch.long)
        fabrics = torch.tensor(self.fab_dict[fab], dtype=torch.long)
        gtrends_t = torch.tensor(multitrends, dtype=torch.float32)

        return (
            _tensor_for_dataloader(item_sales),
            _tensor_for_dataloader(recent_sales_2w),
            _tensor_for_dataloader(categories),
            _tensor_for_dataloader(colors),
            _tensor_for_dataloader(fabrics),
            _tensor_for_dataloader(temporal_features),
            _tensor_for_dataloader(gtrends_t),
            _tensor_for_dataloader(image_features),
        )


class ZeroShotDataset():
    def __init__(self, data_df, img_root, gtrends, cat_dict, col_dict, fab_dict, trend_len):
        self.data_df = data_df
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_root = img_root

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        return self.data_df.iloc[idx, :]

    def preprocess_data(self):
        data = self.data_df

        # Get the Gtrends time series associated with each product
        # Read the images (extracted image features) as well
        gtrends, image_features = [], []
        img_transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        for (idx, row) in tqdm(data.iterrows(), total=len(data), ascii=True):
            cat, col, fab = row["category"], row["color"], row["fabric"]
            start_date = row["release_date"]
            img_path = row["image_path"]

            # Get the gtrend signal up to the previous year (52 weeks) of the release date
            gtrend_start = start_date - pd.DateOffset(weeks=52)
            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat].values
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col].values
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab].values

            cat_gtrend = _fixed_norm_trend(cat_gtrend, self.trend_len)
            col_gtrend = _fixed_norm_trend(col_gtrend, self.trend_len)
            fab_gtrend = _fixed_norm_trend(fab_gtrend, self.trend_len)
            multitrends =  np.vstack([cat_gtrend, col_gtrend, fab_gtrend])


            # Read images
            img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')

            # Append them to the lists
            gtrends.append(multitrends)
            image_features.append(img_transforms(img))

        # Convert to numpy arrays
        gtrends = np.array(gtrends)

        # 销量与时间：兼容 visuelle2（'0'..'11' 在表尾、无 day/week/month/year）与旧版 CSV
        if all(c in data.columns for c in _WEEK_COLS):
            item_sales = torch.FloatTensor(data[_WEEK_COLS].values.astype(np.float32))
            recent_sales_2w = torch.FloatTensor(data[["0", "1"]].values.astype(np.float32))
        else:
            tmp = data.drop(
                ["external_code", "season", "release_date", "image_path"],
                axis=1,
                errors="ignore",
            )
            item_sales = torch.FloatTensor(tmp.iloc[:, :12].values.astype(np.float32))
            recent_sales_2w = torch.FloatTensor(tmp.iloc[:, :2].values.astype(np.float32))

        if all(c in data.columns for c in _TEMPORAL_COLS):
            base = data[_TEMPORAL_COLS].values.astype(np.float32)
            aux = _aux_series_from_df(data).reshape(-1, 1)
            temporal_features = torch.tensor(
                np.hstack([base, aux]).astype(np.float32), dtype=torch.float32
            )
        else:
            dts = pd.to_datetime(data["release_date"])
            day = (dts.dt.dayofweek.astype(np.float32) / 6.0).to_numpy()
            iso = dts.dt.isocalendar()
            week = (iso["week"].astype(np.float32) / 52.0).to_numpy()
            month = (dts.dt.month.astype(np.float32) / 12.0).to_numpy()
            year = (dts.dt.year.astype(np.float32) / 2020.0).to_numpy()
            aux = _aux_series_from_df(data)
            temporal_features = torch.tensor(
                np.stack([day, week, month, year, aux], axis=1).astype(np.float32),
                dtype=torch.float32,
            )

        categories, colors, fabrics = [self.cat_dict[val] for val in data.iloc[:].category.values], \
                                       [self.col_dict[val] for val in data.iloc[:].color.values], \
                                       [self.fab_dict[val] for val in data.iloc[:].fabric.values]

        
        categories, colors, fabrics = torch.LongTensor(categories), torch.LongTensor(colors), torch.LongTensor(fabrics)
        gtrends = torch.FloatTensor(gtrends)
        images = torch.stack(image_features)

        return TensorDataset(
            item_sales,
            recent_sales_2w,
            categories,
            colors,
            fabrics,
            temporal_features,
            gtrends,
            images,
        )

    def get_loader(self, batch_size, train=True, lazy=False):
        """
        lazy=False: 原行为，preprocess_data() 一次性缓存全部分子到内存（易 OOM）。
        lazy=True: 使用 LazyZeroShotDataset，按 batch 读图算 gtrends，与单样本预处理逻辑一致，适合全量训练。
        """
        print("Starting dataset creation process...")
        if lazy:
            dataset = LazyZeroShotDataset(
                self.data_df.copy(),
                self.img_root,
                self.gtrends,
                self.cat_dict,
                self.col_dict,
                self.fab_dict,
                self.trend_len,
            )
            nw = min(4, os.cpu_count() or 1)
            pin = torch.cuda.is_available()
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=train,
                num_workers=nw,
                pin_memory=pin,
                collate_fn=collate_lazy_zeroshot_batch,
            )
        else:
            data_with_gtrends = self.preprocess_data()
            if train:
                data_loader = DataLoader(
                    data_with_gtrends, batch_size=batch_size, shuffle=True, num_workers=4
                )
            else:
                data_loader = DataLoader(
                    data_with_gtrends, batch_size=1, shuffle=False, num_workers=4
                )
        print("Done.")

        return data_loader

