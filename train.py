import os
import argparse
import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
import inspect
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path
from datetime import datetime
from models.model_factory import build_model_from_args
from utils.common_utils import torch_load_trusted
from utils.data_multitrends import ZeroShotDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(args):
    print(args)
    # Seeds for reproducibility (By default we use the number 21)
    pl.seed_everything(args.seed)

    # Load sales data
    train_df = pd.read_csv(Path(args.data_folder + 'train.csv'), parse_dates=['release_date'])
    if args.train_frac < 1.0:
        train_df = train_df.sample(frac=args.train_frac, random_state=args.seed).reset_index(drop=True)
    test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])

    # Load category and color encodings
    cat_dict = torch_load_trusted(Path(args.data_folder + 'category_labels.pt'))
    col_dict = torch_load_trusted(Path(args.data_folder + 'color_labels.pt'))
    fab_dict = torch_load_trusted(Path(args.data_folder + 'fabric_labels.pt'))

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)

    lazy = bool(args.lazy_loader)
    train_loader = ZeroShotDataset(train_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict,
                                   fab_dict, args.trend_len).get_loader(batch_size=args.batch_size, train=True, lazy=lazy)
    test_loader = ZeroShotDataset(test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict,
                                  fab_dict, args.trend_len).get_loader(batch_size=1, train=False, lazy=lazy)

    # Create model
    model = build_model_from_args(
        args,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
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
    parser.add_argument('--data_folder', type=str, default='dataset/')
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
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1, help='FCN 使用；GTM 固定使用 trends memory')
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
    parser.add_argument('--num_attn_heads', type=int, default=4, help='CLI 桥接到 GTM 构造参数 num_heads')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='CLI 桥接到 GTM 构造参数 num_layers')

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='username-here')
    parser.add_argument('--wandb_proj', type=str, default='GTM')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError(f"--train_frac must be in (0, 1], got {args.train_frac}")
    run(args)
