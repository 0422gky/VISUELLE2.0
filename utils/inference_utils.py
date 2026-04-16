from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from models.model_factory import build_model_from_args
from utils.common_utils import torch_load_trusted


def load_model_from_checkpoint(
    args: Namespace,
    *,
    cat_dict: dict[Any, Any],
    col_dict: dict[Any, Any],
    fab_dict: dict[Any, Any],
    ckpt_path: Path,
):
    model = build_model_from_args(
        args,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
    )
    ckpt = torch_load_trusted(ckpt_path)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model


def run_forecast_inference(
    *,
    model,
    test_loader,
    device: torch.device,
    model_type: str,
    output_dim: int,
    collect_full_12w: bool = False,
):
    model.to(device)
    model.eval()

    gt, forecasts = [], []
    gt12_norm, pred12_norm = [], []

    for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            test_data = [tensor.to(device) for tensor in test_data]
            (
                item_sales,
                recent_sales_2w,
                category,
                color,
                textures,
                temporal_features,
                gtrends,
                images,
            ) = test_data

            if model_type == "FCN":
                y_pred = model(category, color, textures, temporal_features, gtrends, images)
            else:
                y_pred = model(
                    category,
                    color,
                    textures,
                    temporal_features,
                    gtrends,
                    images,
                    recent_sales_2w=recent_sales_2w,
                )

            item_sales_flat = item_sales.detach().cpu().numpy().flatten()
            y_pred_flat = y_pred.detach().cpu().numpy().flatten()

            forecasts.append(y_pred_flat[:output_dim])
            gt.append(item_sales_flat[:output_dim])

            if collect_full_12w:
                if item_sales_flat.size < 12:
                    raise ValueError(
                        f"item_sales 长度应为 12（与数据集一致），当前为 {item_sales_flat.size}；"
                        "请检查 test.csv 与 ZeroShotDataset。"
                    )
                gt12_norm.append(item_sales_flat[:12].astype(np.float64, copy=False))
                pred_12w = np.full(12, np.nan, dtype=np.float64)
                pred_12w[:output_dim] = y_pred_flat[:output_dim]
                pred12_norm.append(pred_12w)

    forecasts = np.array(forecasts)
    gt = np.array(gt)

    if not collect_full_12w:
        return forecasts, gt

    return forecasts, gt, np.stack(gt12_norm, axis=0), np.stack(pred12_norm, axis=0)
