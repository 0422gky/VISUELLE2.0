from __future__ import annotations

import argparse
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from utils.data_multitrends import ZeroShotDataset
from pathlib import Path

from utils.inference_utils import (
    load_model_from_checkpoint,
    run_forecast_inference,
    torch_load_trusted,
)
from utils.forecast_metrics import print_error_metrics

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
    cat_dict = torch_load_trusted(Path(args.data_folder + 'category_labels.pt'))
    col_dict = torch_load_trusted(Path(args.data_folder + 'color_labels.pt'))
    fab_dict = torch_load_trusted(Path(args.data_folder + 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)
    
    test_loader = ZeroShotDataset(test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict, \
            fab_dict, args.trend_len).get_loader(batch_size=1, train=False, lazy=bool(args.lazy_loader))


    model_savename = f'{args.wandb_run}_{args.output_dim}'
    
    # Create model
    model = load_model_from_checkpoint(
        args,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        ckpt_path=Path(args.ckpt_path),
    )

    # Forecast the testing set
    forecasts, gt = run_forecast_inference(
        model=model,
        test_loader=test_loader,
        device=device,
        model_type=args.model_type,
        output_dim=args.output_dim,
    )

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
    parser.add_argument('--use_trends', type=int, default=1, help='FCN 使用；GTM 固定使用 trends memory')
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
    parser.add_argument('--num_attn_heads', type=int, default=4, help='CLI 桥接到 GTM 构造参数 num_heads')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='CLI 桥接到 GTM 构造参数 num_layers')
    
    # wandb arguments
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)
