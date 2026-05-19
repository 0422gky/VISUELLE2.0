from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch


def trusted_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """
    Load trusted local checkpoints / label dictionaries across PyTorch versions.

    PyTorch 2.6+ defaults to weights_only=True, which rejects files containing
    Python objects such as the category/color/fabric dictionaries used here.
    """
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def load_label_dicts(data_folder: str | Path) -> tuple[dict, dict, dict]:
    data_dir = Path(data_folder)
    return (
        trusted_torch_load(data_dir / "category_labels.pt"),
        trusted_torch_load(data_dir / "color_labels.pt"),
        trusted_torch_load(data_dir / "fabric_labels.pt"),
    )


def read_split_csv(data_folder: str | Path, split: str, **read_csv_kwargs) -> pd.DataFrame:
    read_csv_kwargs.setdefault("parse_dates", ["release_date"])
    return pd.read_csv(Path(data_folder) / f"{split}.csv", **read_csv_kwargs)


def read_gtrends(data_folder: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_folder) / "gtrends.csv", index_col=[0], parse_dates=True)


def sample_train_df(df: pd.DataFrame, train_frac: float, seed: int) -> pd.DataFrame:
    if not (0.0 < train_frac <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")
    if train_frac < 1.0:
        return df.sample(frac=train_frac, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def extract_hparams(ckpt: Any) -> dict:
    if not isinstance(ckpt, dict):
        return {}
    if "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
        return ckpt["hyper_parameters"]
    if "hparams" in ckpt and isinstance(ckpt["hparams"], dict):
        return ckpt["hparams"]
    return {}


def load_model_state_dict(model: torch.nn.Module, checkpoint_path: str | Path, strict: bool = False) -> None:
    ckpt = trusted_torch_load(checkpoint_path)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=strict)
