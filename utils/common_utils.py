from __future__ import annotations

from pathlib import Path

import pandas as pd


def torch_load_trusted(path: str | Path):
    """
    PyTorch 2.6+ 默认 torch.load(weights_only=True) 会拒绝加载包含 numpy/非纯权重对象的 .pt。
    本项目读取 checkpoint / label dict 时默认信任来源，显式关闭 weights_only。
    """
    import torch

    path_str = str(path)
    try:
        return torch.load(path_str, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容老版本 PyTorch（没有 weights_only 参数）
        return torch.load(path_str, map_location="cpu")


def sample_train_df(df: pd.DataFrame, train_frac: float, seed: int) -> pd.DataFrame:
    """与 train.py 一致：仅对 train 子集抽样。"""
    if not (0.0 < train_frac <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")
    if train_frac < 1.0:
        return df.sample(frac=train_frac, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)
