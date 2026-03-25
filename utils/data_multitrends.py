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
            cat_gtrend = self.gtrends.loc[gtrend_start:start_date][cat][-52:].values[:self.trend_len]
            col_gtrend = self.gtrends.loc[gtrend_start:start_date][col][-52:].values[:self.trend_len]
            fab_gtrend = self.gtrends.loc[gtrend_start:start_date][fab][-52:].values[:self.trend_len]

            cat_gtrend = MinMaxScaler().fit_transform(cat_gtrend.reshape(-1,1)).flatten()
            col_gtrend = MinMaxScaler().fit_transform(col_gtrend.reshape(-1,1)).flatten()
            fab_gtrend = MinMaxScaler().fit_transform(fab_gtrend.reshape(-1,1)).flatten()
            multitrends =  np.vstack([cat_gtrend, col_gtrend, fab_gtrend])


            # Read images
            img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')

            # Append them to the lists
            gtrends.append(multitrends)
            image_features.append(img_transforms(img))

        # Convert to numpy arrays
        gtrends = np.array(gtrends)

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

