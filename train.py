import os
import argparse
import torch
import pandas as pd
import pytorch_lightning as pl
import inspect
import numpy as np
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from utils.data_multitrends import ZeroShotDataset
from utils.io_helpers import load_label_dicts, read_gtrends, read_split_csv, sample_train_df

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _select_scale_slice(scale, start_idx: int, length: int) -> np.ndarray:
    s = np.asarray(scale, dtype=np.float64).ravel()
    if s.size == 1:
        return s
    if s.size >= start_idx + length:
        return s[start_idx : start_idx + length]
    if s.size >= length:
        return s[:length]
    return s


def run(args):
    import wandb

    print(args)
    # Seeds for reproducibility (By default we use the number 21)
    pl.seed_everything(args.seed)

    # Load sales data
    train_df = sample_train_df(read_split_csv(args.data_folder, "train"), args.train_frac, args.seed)
    test_df = read_split_csv(args.data_folder, "test")

    # Load category and color encodings
    cat_dict, col_dict, fab_dict = load_label_dicts(args.data_folder)

    # Load Google trends
    gtrends = read_gtrends(args.data_folder)

    lazy = bool(args.lazy_loader)
    train_loader = ZeroShotDataset(train_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict,
                                   fab_dict, args.trend_len).get_loader(batch_size=args.batch_size, train=True, lazy=lazy)
    test_loader = ZeroShotDataset(test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict,
                                  fab_dict, args.trend_len).get_loader(batch_size=1, train=False, lazy=lazy)

    # Create model
    if args.model_type == 'FCN':
        from models.FCN import FCN

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
    elif args.model_type == 'MMTS':
        from models.MMTS import MMTS

        rescale_values = _select_scale_slice(
            np.load(Path(args.data_folder) / "normalization_scale.npy"),
            2,
            10,
        )
        model = MMTS(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=args.use_text,
            use_img=args.use_img,
            gpu_num=args.gpu_num,
            forecast_mode=args.forecast_mode,
            contrastive_loss_weight=args.contrastive_loss_weight,
            contrastive_temperature=args.contrastive_temperature,
            num_attn_heads=args.num_attn_heads,
            rescale_values=rescale_values.tolist(),
        )
    elif args.model_type == 'StaticQKVGTM':
        from models.StaticQKVGTM import StaticQKVGTM

        model = StaticQKVGTM(
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
            gpu_num=args.gpu_num,
            use_hist_sales=args.use_hist_sales,
            contrastive_loss_weight=args.contrastive_loss_weight,
            contrastive_temperature=args.contrastive_temperature,
        )
    elif args.model_type == 'Simple':
        from models.Simple import Simple

        model = Simple(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            gpu_num=args.gpu_num,
        )
    else:
        from models.GTM import GTM

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

    # Model Training
    # Define model saving procedure
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    model_savename = args.model_type + '_' + args.wandb_run

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.log_dir + '/'+args.model_type,
        filename=model_savename+'---{epoch}---'+dt_string,
        monitor='val_mae',
        mode='min',
        save_top_k=1
    )

    wandb.init(entity=args.wandb_entity, project=args.wandb_proj, name=args.wandb_run)
    wandb_logger = pl_loggers.WandbLogger()
    wandb_logger.watch(model)

    # If you wish to use Tensorboard you can change the logger to:
    # tb_logger = pl_loggers.TensorBoardLogger(args.log_dir+'/', name=model_savename)
    # Lightning API 兼容：旧版本用 gpus=，新版本用 accelerator/devices=
    trainer_kwargs = dict(
        max_epochs=args.epochs,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer_sig = inspect.signature(pl.Trainer)
    if "gpus" in trainer_sig.parameters:
        trainer_kwargs["gpus"] = [args.gpu_num]
    else:
        # 让 devices=1 映射到你指定的卡号
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
            trainer_kwargs["accelerator"] = "gpu"
            trainer_kwargs["devices"] = 1
        else:
            trainer_kwargs["accelerator"] = "cpu"
            trainer_kwargs["devices"] = 1

    trainer = pl.Trainer(**trainer_kwargs)

    # Fit model
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=test_loader)

    # Print out path of best model
    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='visuelle2/')
    parser.add_argument(
        '--lazy_loader',
        type=int,
        default=1,
        help='1=惰性加载每条样本的 gtrends+图像（全量 train 省内存，与原版单样本数学一致）；0=原 preload 全表',
    )
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=0)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM, FCN, MMTS, StaticQKVGTM, or Simple')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument(
        '--use_hist_sales',
        type=int,
        default=0,
        help='1=将两周销量 recent_sales_2w 经 Sales2WeekEmbedder 接入 FusionNetwork（与 use_img/use_text 并列）',
    )
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument(
        '--train_frac',
        type=float,
        default=1.0,
        help='Random fraction of train.csv rows to use (0,1], e.g. 0.25 for 25%%. Uses args.seed for reproducibility.',
    )
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument(
        '--forecast_mode',
        type=str,
        default='direct_2_10',
        choices=['direct_2_10', 'rolling_2_1'],
        help='MMTS only: direct 2-to-10 prediction or 2-week-to-1-week rolling prediction.',
    )
    parser.add_argument(
        '--contrastive_loss_weight',
        type=float,
        default=0.1,
        help='MMTS/StaticQKVGTM: weight for CLIP-style InfoNCE alignment loss.',
    )
    parser.add_argument(
        '--contrastive_temperature',
        type=float,
        default=0.07,
        help='MMTS/StaticQKVGTM: temperature for InfoNCE logits.',
    )

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='username-here')
    parser.add_argument('--wandb_proj', type=str, default='GTM')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError(f"--train_frac must be in (0, 1], got {args.train_frac}")
    run(args)
