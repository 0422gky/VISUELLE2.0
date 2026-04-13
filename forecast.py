from __future__ import annotations

import argparse
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
from pathlib import Path

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

def run(args):
    print(args)
    
    # Set up CUDA
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Seeds for reproducibility
    pl.seed_everything(args.seed)

    # Load sales data    
    test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])
    item_codes = test_df['external_code'].values

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
            forecasts.append(y_pred.detach().cpu().numpy().flatten()[:args.output_dim])
            gt.append(item_sales.detach().cpu().numpy().flatten()[:args.output_dim])
    forecasts = np.array(forecasts)
    gt = np.array(gt)

    # 销量 Maximum scaling: normalized = raw / train_global_max；此处 × normalization_scale 即还原 raw。
    # 见论文仓库 issue: global max over training set.
    rescale_vals = np.load(args.data_folder + 'normalization_scale.npy')
    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals
    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    
    torch.save({'results': forecasts* rescale_vals, 'gts': gt* rescale_vals, 'codes': item_codes.tolist()}, Path('results/' + model_savename+'.pth'))


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
        help='GTM：1=与训练一致，使用 recent_sales_2w；FCN 忽略该字段',
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

    args = parser.parse_args()
    run(args)
