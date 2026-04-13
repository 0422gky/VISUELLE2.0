"""
销量归一化（与论文仓库 issue 说明一致）:
- 训练集上所有周、所有样本的销量取 **全局最大值** global_max。
- 输入/目标均为:  raw_sales / global_max  （Maximum scaling）。
- 还原到原始销量尺度:  normalized * global_max 。

本仓库中 `normalization_scale.npy` 即该尺度（常为标量或与周数等长的广播形式，与训练时保存方式一致）。
新数据集请先在 **train** 上算 global_max，对 train/test 用同一除数；推理时再乘同一数以还原。
"""

from __future__ import annotations

import argparse
import json
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from models.GTM import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from utils.curve_eval_metrics import avg_pearson_avg_dtw


def _torch_load_trusted(path: Path):
    """
    PyTorch 2.6+ 默认 torch.load(weights_only=True) 会拒绝加载包含 numpy/非纯权重对象的 .pt。
    本脚本用于加载 checkpoint/data dict，因此显式关闭 weights_only。
    """
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def compute_max_scaling_from_train_csv(train_csv: Path, num_week_cols: int = 12) -> float:
    """
    论文 / issue: 用训练集上「所有店、所有周」销量的全局最大值做分母。
    此处对 train.csv 前 num_week_cols 列（与 ZeroShotDataset 的 12 周销量列一致）取 nanmax。

    注意: 若 train.csv 里已是 raw/global_max 后的比例（本仓库公开数据常如此），则 nanmax≈1，
    不能当作「原始件数」再乘一次；此时应使用训练时保存的 normalization_scale.npy 或原始未归一化 train。
    """
    df = pd.read_csv(train_csv)
    if df.shape[1] < num_week_cols:
        raise ValueError(f"{train_csv} 列数不足 {num_week_cols}，无法取周销量最大值。")
    mat = df.iloc[:, :num_week_cols].to_numpy(dtype=np.float64)
    m = float(np.nanmax(mat))
    if m <= 0:
        raise ValueError(
            f"{train_csv} 前 {num_week_cols} 列全局最大值 {m} 无效。"
            "若 CSV 已是 raw/global_max 的比例，则 max≈1，请改用 normalization_scale.npy 或 --rescale_max=原始训练集真实 global max。"
        )
    return m


def load_rescale_values(
    data_folder: Path,
    *,
    rescale_max: float | None,
    scale_from_train_max: bool,
    train_csv_name: str,
    num_week_cols: int,
) -> np.ndarray:
    """
    返回用于 ×scale 还原的数组（标量或向量），与 np.load(normalization_scale.npy) 语义一致。
    优先级: --rescale_max > --scale_from_train_max > normalization_scale.npy
    """
    if rescale_max is not None:
        if rescale_max <= 0:
            raise ValueError("--rescale_max 必须为正数。")
        print(f"Using --rescale_max = {rescale_max} (manual global max for denormalization)")
        return np.array([rescale_max], dtype=np.float64)
    if scale_from_train_max:
        train_path = data_folder / train_csv_name
        m = compute_max_scaling_from_train_csv(train_path, num_week_cols=num_week_cols)
        print(
            f"Using global max from {train_path} over first {num_week_cols} week columns: {m}"
        )
        return np.array([m], dtype=np.float64)
    npy_path = data_folder / "normalization_scale.npy"
    arr = np.load(str(npy_path))
    print(f"Loaded rescale from {npy_path} shape={np.asarray(arr).shape}")
    return np.asarray(arr, dtype=np.float64)


def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)

    return round(mae, 3), round(wape, 3)
    

def print_error_metrics(
    y_test,
    y_hat,
    rescaled_y_test,
    rescaled_y_hat,
    *,
    wsl: int = 0,
    tw: int | None = None,
):
    mae, wape = cal_error_metrics(y_test, y_hat)
    rescaled_mae, rescaled_wape = cal_error_metrics(rescaled_y_test, rescaled_y_hat)
    print(mae, wape, rescaled_mae, rescaled_wape)
    ap, adtw = avg_pearson_avg_dtw(rescaled_y_test, rescaled_y_hat, wsl=wsl, tw=tw)
    print(f"Avg_Pearson {ap:.4f}  Avg_DTW {adtw:.4f}  (window [{wsl}:{tw if tw is not None else rescaled_y_test.shape[1]}), rescaled sales)")


def _rescale_matrix_rows(mat: np.ndarray, scale) -> np.ndarray:
    """
    将 (N, L) 矩阵每行乘以 scale，即 Maximum scaling 的逆变换: normalized * global_max。
    scale 可为标量（论文中的单一 global max）、或与 L 等长、或较短（不足部分用最后一个值向右填充）。
    """
    m = np.asarray(mat, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64).ravel()
    L = m.shape[1]
    if s.size == 1:
        return m * float(s[0])
    if s.size >= L:
        return m * s[:L]
    s_full = np.empty(L, dtype=np.float64)
    s_full[: s.size] = s
    s_full[s.size :] = s[-1]
    return m * s_full


def write_forecast_csv(
    path: Path,
    external_codes: np.ndarray,
    retail: np.ndarray,
    y_test: np.ndarray,
    y_hat: np.ndarray,
    sales_original_12w_normalized: np.ndarray,
    sales_predicted_12w_normalized: np.ndarray,
    wide_sales_weeks: bool = False,
) -> None:
    """
    导出预测结果 CSV。

    与 test.csv / train.csv 的关系（与 forecast.py 一致）:
    - DataLoader 的 item_sales = CSV 前 12 列，已是 raw / train_global_max（Maximum scaling，数值多在 ~0.01 量级）。
    - 模型输出 y_pred 也在同一归一化空间。

    列含义:
    - y_test / y_hat: JSON，**× normalization_scale 之后** 的「原始销量尺度」，仅前 output_dim 周（与 forecast.py 打印的 rescaled 一致）。
    - sales_original_12w_restored / sales_predicted_12w_restored: JSON 列名沿用旧称，但内容为 **未乘 scale**
      的归一化曲线，与 test.csv、train.csv 前 12 列量纲一致（约 0.01）；预测列在后若干周为 null。

    wide_sales_weeks=True 时 true_sales_wk_* / pred_sales_wk_* 与上述两列 JSON 相同（归一化空间）。
    """
    n = len(external_codes)
    if not (
        len(retail) == n
        and len(y_test) == n
        and len(y_hat) == n
        and len(sales_original_12w_normalized) == n
        and len(sales_predicted_12w_normalized) == n
    ):
        raise ValueError(
            f"行数不一致: codes={n}, retail={len(retail)}, y_test={len(y_test)}, "
            f"y_hat={len(y_hat)}, orig12={len(sales_original_12w_normalized)}, pred12={len(sales_predicted_12w_normalized)}"
        )

    def _json_12w(row: np.ndarray) -> str:
        out = []
        for x in np.asarray(row, dtype=np.float64).ravel():
            if np.isnan(x):
                out.append(None)
            else:
                out.append(float(x))
        return json.dumps(out)

    y_test_json = [json.dumps(row.tolist()) for row in y_test]
    y_hat_json = [json.dumps(row.tolist()) for row in y_hat]
    orig12_json = [_json_12w(row) for row in sales_original_12w_normalized]
    pred12_json = [_json_12w(row) for row in sales_predicted_12w_normalized]

    df = pd.DataFrame(
        {
            "external_code": external_codes,
            "retail": retail,
            "y_test": y_test_json,
            "y_hat": y_hat_json,
            # 列名保留兼容下游；数值为归一化空间（与 train/test CSV 前 12 列一致）
            "sales_original_12w_restored": orig12_json,
            "sales_predicted_12w_restored": pred12_json,
        }
    )
    if wide_sales_weeks:
        o = np.asarray(sales_original_12w_normalized, dtype=np.float64)
        p = np.asarray(sales_predicted_12w_normalized, dtype=np.float64)
        if o.shape[1] != 12 or p.shape[1] != 12:
            raise ValueError("wide_sales_weeks 要求 12 周归一化矩阵形状为 (N, 12)")
        for w in range(12):
            df[f"true_sales_wk_{w}"] = o[:, w]
            df[f"pred_sales_wk_{w}"] = p[:, w]

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Forecast CSV written: {path}")


def run(args):
    print(args)
    
    # Set up CUDA
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Seeds for reproducibility
    pl.seed_everything(args.seed)

    # Load sales data（与 forecast.py 一致；可选 nrows 仅用于调试）
    read_csv_kw = {"parse_dates": ["release_date"]}
    if args.nrows is not None:
        read_csv_kw["nrows"] = args.nrows
    test_df = pd.read_csv(Path(args.data_folder + "test.csv"), **read_csv_kw)
    item_codes = test_df['external_code'].values
    if 'retail' not in test_df.columns:
        raise ValueError("test.csv 缺少列 `retail`，无法导出 forecast CSV（与 export_item_embeddings 元数据列一致）。")
    retail_vals = test_df['retail'].values

     # Load category and color encodings
    cat_dict = _torch_load_trusted(Path(args.data_folder + 'category_labels.pt'))
    col_dict = _torch_load_trusted(Path(args.data_folder + 'color_labels.pt'))
    fab_dict = _torch_load_trusted(Path(args.data_folder + 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)
    
    test_loader = ZeroShotDataset(test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict, \
            fab_dict, args.trend_len).get_loader(batch_size=1, train=False, lazy=bool(args.lazy_loader))


    model_savename = f'{args.wandb_run}_{args.output_dim}'
    
    # Create model
    model = None
    if args.model_type == 'FCN':
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_trends=args.use_trends,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num
        )
    else:
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num,
            use_hist_sales=args.use_hist_sales,
        )
    
    ckpt = _torch_load_trusted(Path(args.ckpt_path))
    model.load_state_dict(ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt, strict=False)

    # Forecast the testing set
    model.to(device)
    model.eval()
    gt, forecasts = [], []
    gt12_norm, pred12_norm = [], []
    for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            test_data = [tensor.to(device) for tensor in test_data]
            item_sales, recent_sales_2w, category, color, textures, temporal_features, gtrends, images = test_data
            if args.model_type == 'FCN':
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
            is_flat = item_sales.detach().cpu().numpy().flatten()
            y_flat = y_pred.detach().cpu().numpy().flatten()
            if is_flat.size < 12:
                raise ValueError(
                    f"item_sales 长度应为 12（与数据集一致），当前为 {is_flat.size}；请检查 test.csv 与 ZeroShotDataset。"
                )
            forecasts.append(y_flat[: args.output_dim])
            gt.append(is_flat[: args.output_dim])
            gt12_norm.append(is_flat[:12].astype(np.float64, copy=False))
            p12 = np.full(12, np.nan, dtype=np.float64)
            p12[: args.output_dim] = y_flat[: args.output_dim]
            pred12_norm.append(p12)
    forecasts = np.array(forecasts)
    gt = np.array(gt)
    gt12_norm = np.stack(gt12_norm, axis=0)
    pred12_norm = np.stack(pred12_norm, axis=0)

    data_folder = Path(args.data_folder)
    rescale_vals = load_rescale_values(
        data_folder,
        rescale_max=args.rescale_max,
        scale_from_train_max=args.scale_from_train_max,
        train_csv_name=args.train_csv,
        num_week_cols=args.num_week_cols,
    )
    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals
    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    out_csv = Path(args.output_csv) if args.output_csv else Path('results') / f'{model_savename}_forecast.csv'
    # 12 周 JSON：与 test/train.csv 前 12 列相同（Maximum scaling，未 × normalization_scale）
    write_forecast_csv(
        out_csv,
        item_codes,
        retail_vals,
        rescaled_gt,
        rescaled_forecasts,
        gt12_norm,
        pred12_norm,
        wide_sales_weeks=args.wide_sales_weeks,
    )

    torch.save(
        {'results': forecasts * rescale_vals, 'gts': gt * rescale_vals, 'codes': item_codes.tolist()},
        Path('results/' + model_savename + '.pth'),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument(
        '--lazy_loader',
        type=int,
        default=1,
        help='1=惰性加载 gtrends+图像（大 test 省内存）；0=原 preload',
    )
    parser.add_argument('--ckpt_path', type=str, default='log/path-to-model.ckpt')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=21)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument(
        '--use_hist_sales',
        type=int,
        default=0,
        help='1=与训练时一致，传入 recent_sales_2w；checkpoint 需与 --use_hist_sales 匹配',
    )
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    
    # wandb arguments
    parser.add_argument('--wandb_run', type=str, default='Run1')
    parser.add_argument(
        '--output_csv',
        type=str,
        default='',
        help='预测明细 CSV 路径；默认 results/<wandb_run>_<output_dim>_forecast.csv',
    )
    parser.add_argument(
        '--nrows',
        type=int,
        default=None,
        help='仅读取 test.csv 前 n 行（默认全部，用于快速调试）',
    )
    parser.add_argument(
        '--wide_sales_weeks',
        action='store_true',
        help='额外输出 true_sales_wk_0..11、pred_sales_wk_0..11（归一化空间，与 train/test CSV 前 12 列一致）',
    )
    parser.add_argument(
        '--rescale_max',
        type=float,
        default=None,
        help='手动指定训练集全局最大销量（Maximum scaling 的分母）；指定后不再读 normalization_scale.npy',
    )
    parser.add_argument(
        '--scale_from_train_max',
        action='store_true',
        help='从 data_folder/train_csv 前 num_week_cols 列计算全局 max，用作还原尺度（与论文 issue 一致）',
    )
    parser.add_argument(
        '--train_csv',
        type=str,
        default='train.csv',
        help='与 --scale_from_train_max 联用，相对 data_folder',
    )
    parser.add_argument(
        '--num_week_cols',
        type=int,
        default=12,
        help='train.csv 前若干列为周销量，与 ZeroShotDataset 一致（默认 12）',
    )

    args = parser.parse_args()
    run(args)