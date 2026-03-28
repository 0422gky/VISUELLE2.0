import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from sklearn.preprocessing import MinMaxScaler
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _ensure_trend_length_1d(values, target_len: int) -> np.ndarray:
    """
    将单条趋势拉平为长度 target_len，避免不同商品切片长度不一致导致
    np.array(gtrends_list) 报 inhomogeneous shape。
    不足时用最后一个有效值向后填充，全空则为零。
    """
    a = np.asarray(values, dtype=np.float64).ravel()
    if a.size == 0:
        return np.zeros(target_len, dtype=np.float64)
    if a.size >= target_len:
        return a[:target_len].astype(np.float64, copy=False)
    out = np.empty(target_len, dtype=np.float64)
    out[: a.size] = a
    out[a.size :] = a[-1]
    return out


def _ensure_multitrends_matrix(m: np.ndarray, num_streams: int, trend_len: int) -> np.ndarray:
    """
    保证每条样本的多路趋势为固定 (num_streams, trend_len)，供 np.stack 堆叠。
    若上游仍出现不齐宽（例如异常 NaN/标量），在此处截断或按列填充。
    """
    m = np.asarray(m, dtype=np.float64)
    if m.ndim != 2:
        m = np.atleast_2d(m)
    if m.shape[0] != num_streams and m.shape[1] == num_streams:
        m = m.T
    if m.shape[0] != num_streams:
        raise ValueError(
            f"multitrends 行数应为 {num_streams}，实际 shape={m.shape}（检查 category/color/fabric 三路趋势）"
        )
    L = m.shape[1]
    if L == trend_len:
        return m.astype(np.float32, copy=False)
    if L > trend_len:
        return m[:, :trend_len].astype(np.float32, copy=False)
    if L == 0:
        return np.zeros((num_streams, trend_len), dtype=np.float32)
    pad = trend_len - L
    tail = np.tile(m[:, -1:], (1, pad))
    return np.hstack([m, tail]).astype(np.float32, copy=False)


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
            cat, col, fab, start_date, img_path = row['category'], row['color'], row['fabric'], \
                row['release_date'], row['image_path'] # remove row extra 

            # Get the gtrend signal up to the previous year (52 weeks) of the release date
            gtrend_start = start_date - pd.DateOffset(weeks=52)
            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[: self.trend_len]
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[: self.trend_len]
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[: self.trend_len]

            cat_gtrend = _ensure_trend_length_1d(cat_gtrend, self.trend_len)
            col_gtrend = _ensure_trend_length_1d(col_gtrend, self.trend_len)
            fab_gtrend = _ensure_trend_length_1d(fab_gtrend, self.trend_len)

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1, 1)).flatten()
            col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1, 1)).flatten()
            fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1, 1)).flatten()
            multitrends = np.vstack([cat_gtrend, col_gtrend, fab_gtrend])
            multitrends = _ensure_multitrends_matrix(multitrends, num_streams=3, trend_len=self.trend_len)

            # Read images
            img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')

            # Append them to the lists
            gtrends.append(multitrends)
            image_features.append(img_transforms(img))

        # 使用 stack 而非 array：若仍有不齐会立刻报错；正常情况下均为 (3, trend_len)
        gtrends = np.stack(gtrends, axis=0)

        # DummyEmbedder expects 5 numeric inputs: day, week, month, year, restock (each embedded via Linear(1, dim); see DummyEmbedder).
        # New CSVs keep category/color/fabric as strings and use release_date instead of precomputed d/w/m/y.
        rd = pd.to_datetime(data['release_date'])
        if hasattr(rd.dt, 'isocalendar'):
            week_num = rd.dt.isocalendar().week.to_numpy(dtype=np.float32)
        else:
            week_num = rd.dt.week.to_numpy(dtype=np.float32)
        if 'restock' in data.columns:
            restock_vals = pd.to_numeric(data['restock'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        else:
            restock_vals = np.zeros(len(data), dtype=np.float32)
        temporal_np = np.column_stack([
            rd.dt.day.to_numpy(dtype=np.float32),
            week_num,
            rd.dt.month.to_numpy(dtype=np.float32),
            rd.dt.year.to_numpy(dtype=np.float32),
            restock_vals,
        ])

        # Remove non-numerical information (restock is copied into temporal_np above)
        drop_cols = ['external_code', 'season', 'release_date', 'image_path', 'retail', 'restock']
        data.drop([c for c in drop_cols if c in data.columns], axis=1, inplace=True)

        # Last 12 numeric columns = weekly sales (e.g. columns "0".."11"); exclude strings like category/color/fabric.
        num = data.select_dtypes(include=[np.number])
        if num.shape[1] < 12:
            raise ValueError(
                f'Need at least 12 numeric sales columns at the end; got {num.shape[1]} numeric columns.'
            )
        item_sales = torch.tensor(num.iloc[:, -12:].to_numpy(dtype=np.float32))
        temporal_features = torch.tensor(temporal_np, dtype=torch.float32)
        categories, colors, fabrics = [self.cat_dict[val] for val in data.iloc[:].category.values], \
                                       [self.col_dict[val] for val in data.iloc[:].color.values], \
                                       [self.fab_dict[val] for val in data.iloc[:].fabric.values]

        
        categories, colors, fabrics = torch.LongTensor(categories), torch.LongTensor(colors), torch.LongTensor(fabrics)
        gtrends = torch.FloatTensor(gtrends)
        images = torch.stack(image_features)

        return TensorDataset(item_sales, categories, colors, fabrics, temporal_features, gtrends, images)

    def get_loader(self, batch_size, train=True):
        print('Starting dataset creation process...')
        data_with_gtrends = self.preprocess_data()
        data_loader = None
        if train:
            data_loader = DataLoader(data_with_gtrends, batch_size=batch_size, shuffle=True, num_workers=4)
        else:
            data_loader = DataLoader(data_with_gtrends, batch_size=1, shuffle=False, num_workers=4)
        print('Done.')

        return data_loader

