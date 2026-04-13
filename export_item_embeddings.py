import argparse
import os
import sys
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.GTM import GTM
from utils.data_multitrends import (
    LazyZeroShotDataset,
    ZeroShotDataset,
    collate_lazy_zeroshot_batch,
)


def _torch_load_trusted(path: str):
    """
    PyTorch 2.6+ 默认 torch.load(weights_only=True) 会拒绝加载包含 numpy/非纯权重对象的 .pt。
    本脚本加载的是数据字典/ckpt（你一般是信任来源的），因此显式 weights_only=False。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容老版本 PyTorch（没有 weights_only 参数）
        return torch.load(path, map_location="cpu")


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = _torch_load_trusted(checkpoint_path)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)


def _extract_hparams(ckpt: Dict) -> Dict:
    if not isinstance(ckpt, dict):
        return {}
    if "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
        return ckpt["hyper_parameters"]
    if "hparams" in ckpt and isinstance(ckpt["hparams"], dict):
        return ckpt["hparams"]
    return {}


def _read_split_df(data_folder: str, split: str) -> pd.DataFrame:
    csv_path = os.path.join(data_folder, f"{split}.csv")
    # parse_dates 用于满足 ZeroShotDataset.preprocess_data() 里的时间运算
    df = pd.read_csv(csv_path, parse_dates=["release_date"])
    return df


def sample_train_df(df: pd.DataFrame, train_frac: float, seed: int) -> pd.DataFrame:
    """与 train.py 一致：仅对 train 子集抽样。"""
    if not (0.0 < train_frac <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")
    if train_frac < 1.0:
        return df.sample(frac=train_frac, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def _release_date_slash(val) -> str:
    ts = pd.to_datetime(val)
    return f"{int(ts.year)}/{int(ts.month)}/{int(ts.day)}"


def _build_compact_dataframe(
    df: pd.DataFrame,
    sales_matrix: np.ndarray,
    emb_matrix: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    """紧凑列：train_* / test_*；split_name=='all' 时为 split + code + …"""
    n = len(df)
    if sales_matrix.shape != (n, 12):
        raise ValueError(f"sales_matrix shape {sales_matrix.shape} != ({n}, 12)")
    if emb_matrix.shape[0] != n:
        raise ValueError(f"emb_matrix rows {emb_matrix.shape[0]} != {n}")
    if "external_code" not in df.columns:
        raise ValueError("compact format requires column 'external_code' in CSV")

    codes = df["external_code"].values
    if "retail" in df.columns:
        retail = df["retail"].values
    else:
        retail = np.full(n, np.nan, dtype=np.float64)
    if "restock" in df.columns:
        restock = df["restock"].values
    else:
        restock = np.full(n, np.nan, dtype=np.float64)

    release_dates = df["release_date"].map(_release_date_slash)
    # 每行一列：Python list[float]，非 JSON 字符串（DataFrame 列为 object）
    curves = [sales_matrix[i].astype(np.float64, copy=False).tolist() for i in range(n)]
    embs = [emb_matrix[i].astype(np.float64, copy=False).tolist() for i in range(n)]

    if split_name == "all":
        if "_compact_split" not in df.columns:
            raise ValueError("split=all compact export requires internal column _compact_split")
        return pd.DataFrame(
            {
                "split": df["_compact_split"].astype(str).values,
                "code": codes,
                "retail": retail,
                "curve": curves,
                "restock": restock,
                "release_date": release_dates,
                "emb": embs,
            }
        )

    prefix = split_name
    return pd.DataFrame(
        {
            f"{prefix}_code": codes,
            f"{prefix}_retail": retail,
            f"{prefix}_curve": curves,
            f"{prefix}_restock": restock,
            f"{prefix}_release_date": release_dates,
            f"{prefix}_emb": embs,
        }
    )


_WEEK_COLS = [str(i) for i in range(12)]


def _prepare_metadata_df(df: pd.DataFrame, split_value: str) -> pd.DataFrame:
    """
    导出元数据：与数据 CSV 一致，只去掉 12 周销量列 '0'..'11'（避免与导出的 sales_wk_* 重复）。
    visuelle2：external_code, retail, season, category, color, image_path, fabric, release_date, restock。
    旧版 train：可仍含 day/week/month/year/extra（若 CSV 里有）。
    """
    meta_cols = [
        c
        for c in df.columns
        if c not in _WEEK_COLS and c != "_compact_split"
    ]
    if not meta_cols:
        raise ValueError("No metadata columns after excluding week columns '0'..'11'")

    meta_df = df[meta_cols].copy()
    if "release_date" in meta_df.columns and np.issubdtype(
        meta_df["release_date"].dtype, np.datetime64
    ):
        meta_df["release_date"] = meta_df["release_date"].dt.strftime("%Y-%m-%d")
    meta_df.insert(0, "split", split_value)
    meta_df = meta_df.reset_index(drop=True)
    return meta_df


def _ensure_item_embedding(
    fused_feature: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    fused_feature 运行时实际 shape 可能是：
    - (B, 1, H)
    - (B, T, H)
    - (T, B, H)  (则需要 transpose)
    """
    shape = tuple(fused_feature.shape)
    if fused_feature.dim() == 2:
        # (B, H)
        if fused_feature.shape[0] != batch_size:
            raise ValueError(f"Unexpected fused_feature shape: {shape}, batch_size={batch_size}")
        item_embedding = fused_feature
        return item_embedding, shape

    if fused_feature.dim() != 3:
        raise ValueError(f"Unexpected fused_feature dim={fused_feature.dim()}, shape={shape}")

    B = batch_size
    # 尝试把 fused_feature 统一成 (B, T, H)
    if fused_feature.shape[0] == B:
        # already (B, T, H) or (B, 1, H)
        bf = fused_feature
    elif fused_feature.shape[1] == B:
        # (T, B, H) -> (B, T, H)
        bf = fused_feature.transpose(0, 1)
    else:
        raise ValueError(f"Cannot infer batch dim from fused_feature shape={shape} with batch_size={B}")

    # bf: (B, T, H)
    T = bf.shape[1]
    if T == 1:
        item_embedding = bf[:, 0, :]
    else:
        item_embedding = bf.mean(dim=1)

    return item_embedding, shape


@torch.no_grad()
def export_for_df(
    split_name: str,
    df: pd.DataFrame,
    args: argparse.Namespace,
    model: GTM,
) -> None:
    output_format = getattr(args, "output_format", "compact")

    # 用于模型的 df：preprocess_data 会 inplace drop 列，因此用副本
    df_for_model = df.copy(deep=True)

    use_lazy = bool(getattr(args, "lazy_loader", 1))
    if use_lazy:
        # 与 preprocess_data 逐样本一致，峰值内存随 batch 而非全表
        ds = LazyZeroShotDataset(
            data_df=df_for_model,
            img_root=os.path.join(args.data_folder, "images"),
            gtrends=args.gtrends,
            cat_dict=args.cat_dict,
            col_dict=args.col_dict,
            fab_dict=args.fab_dict,
            trend_len=args.trend_len,
        )
        # default_collate 的 resize_ 在部分环境下会报 storage not resizable；用 stack collate
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            collate_fn=collate_lazy_zeroshot_batch,
        )
    else:
        dataset = ZeroShotDataset(
            data_df=df_for_model,
            img_root=os.path.join(args.data_folder, "images"),
            gtrends=args.gtrends,
            cat_dict=args.cat_dict,
            col_dict=args.col_dict,
            fab_dict=args.fab_dict,
            trend_len=args.trend_len,
        )
        tensor_dataset = dataset.preprocess_data()
        loader = DataLoader(
            tensor_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    n = len(df_for_model)
    emb_matrix = None  # (N, H)
    sales_matrix = np.zeros((n, 12), dtype=np.float32)

    offset = 0
    model.eval()
    first_print_done = False
    ablate_print_done = False
    ablate2_print_done = False

    for batch in loader:
        # TensorDataset item order:
        # item_sales, recent_sales_2w, categories, colors, fabrics, temporal_features, gtrends, images
        item_sales, recent_sales_2w, category, color, fabric, temporal_features, gtrends, images = batch

        # Move to device
        item_sales = item_sales.to(args.device)
        recent_sales_2w = recent_sales_2w.to(args.device)
        category = category.to(args.device)
        color = color.to(args.device)
        fabric = fabric.to(args.device)
        temporal_features = temporal_features.to(args.device)
        gtrends = gtrends.to(args.device)
        images = images.to(args.device)

        if args.ablation2_ccf_img:
            if not ablate2_print_done:
                print(
                    f"[{split_name}] ablation2_ccf_img=1: keep only category/color/fabric + image signal; "
                    "set gtrends/temporal_features/recent_sales_2w to zeros before forward"
                )
                ablate2_print_done = True
            gtrends = torch.zeros_like(gtrends)
            temporal_features = torch.zeros_like(temporal_features)
            recent_sales_2w = torch.zeros_like(recent_sales_2w)
        elif args.ablate_trends:
            if not ablate_print_done:
                print(f"[{split_name}] ablate_trends=1: set gtrends tensor to zeros before forward")
                ablate_print_done = True
            gtrends = torch.zeros_like(gtrends)

        pred, fused_feature = model(
            category,
            color,
            fabric,
            temporal_features,
            gtrends,
            images,
            recent_sales_2w=recent_sales_2w,
            return_embedding=True,
        )
        # fused_feature: decoder cross-attn output
        B = item_sales.shape[0]
        item_embedding, fused_shape = _ensure_item_embedding(fused_feature, batch_size=B)

        # runtime check / logging
        if not first_print_done:
            print(f"[{split_name}] fused_feature.shape(runtime)={fused_shape}, item_embedding.shape={tuple(item_embedding.shape)}")
            first_print_done = True

        item_emb_np = item_embedding.detach().cpu().numpy().astype(np.float32)
        sales_np = item_sales.detach().cpu().numpy().astype(np.float32)

        if emb_matrix is None:
            H = item_emb_np.shape[1]
            emb_matrix = np.zeros((n, H), dtype=np.float32)

        bs = item_emb_np.shape[0]
        emb_matrix[offset : offset + bs] = item_emb_np
        sales_matrix[offset : offset + bs] = sales_np
        offset += bs

    if emb_matrix is None:
        raise RuntimeError("No batches processed; cannot export embeddings.")
    if offset != n:
        raise RuntimeError(f"Export alignment error: processed_rows={offset}, expected_rows={n}")

    os.makedirs(args.output_dir, exist_ok=True)
    base = f"{split_name}_item_embeddings"
    out_dir = args.output_dir
    npy_path = os.path.join(out_dir, f"{base}.npy")
    np.save(npy_path, emb_matrix)

    if output_format in ("compact", "both"):
        compact_df = _build_compact_dataframe(df, sales_matrix, emb_matrix, split_name)
        storage = getattr(args, "compact_storage", "parquet")
        if storage == "parquet":
            compact_path = os.path.join(out_dir, f"{base}.parquet")
            try:
                compact_df.to_parquet(compact_path, index=False, engine="pyarrow")
            except ImportError as e:
                raise RuntimeError(
                    "紧凑导出 Parquet 需要 pyarrow：pip install pyarrow；或改用 --compact_storage csv"
                ) from e
            print(f"[{split_name}] wrote compact Parquet: {compact_path}")
        else:
            compact_path = os.path.join(out_dir, f"{base}.csv")
            compact_df.to_csv(compact_path, index=False)
            print(f"[{split_name}] wrote compact CSV: {compact_path}")

    if output_format in ("wide", "both"):
        meta_df = _prepare_metadata_df(df, split_name)
        sales_cols = [f"sales_wk_{i}" for i in range(12)]
        sales_df = pd.DataFrame(sales_matrix, columns=sales_cols)
        H = emb_matrix.shape[1]
        emb_cols = [f"emb_{i:03d}" for i in range(H)]
        emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)
        final_wide = pd.concat([meta_df, sales_df, emb_df], axis=1)
        wide_csv = os.path.join(out_dir, f"{base}_wide.csv")
        meta_csv = os.path.join(out_dir, f"{base}_meta.csv")
        final_wide.to_csv(wide_csv, index=False)
        pd.concat([meta_df, sales_df], axis=1).to_csv(meta_csv, index=False)
        print(f"[{split_name}] wrote wide CSV: {wide_csv}")


@torch.no_grad()
def export_for_split(
    split_name: str,
    args: argparse.Namespace,
    model: GTM,
) -> None:
    df = _read_split_df(args.data_folder, split_name)
    if split_name == "train":
        df = sample_train_df(df, args.train_frac, args.seed)
    export_for_df(split_name=split_name, df=df, args=args, model=model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path/to/model.ckpt or .pth")
    parser.add_argument("--data_folder", type=str, default="dataset/", help="dataset root folder")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu (NOTE: this project masks are hard-coded to cuda)")

    # 当 checkpoint 不包含 hyper_parameters 时，这些用于兜底
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument(
        "--use_hist_sales",
        type=int,
        default=0,
        help="与训练 GTM 一致；checkpoint 含 hyper_parameters 时以 ckpt 为准",
    )
    parser.add_argument(
        "--lazy_loader",
        type=int,
        default=1,
        help="1=按样本惰性加载 gtrends+图像（大表省内存，与 preprocess 单样本逻辑一致）；0=原 preload 全表",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=1.0,
        help="与 train.py 一致：仅对 train.csv 抽样后再导出；test 不受影响。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=21,
        help="train_frac 抽样用 random_state，与 train.py --seed 一致。",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="compact",
        choices=["compact", "wide", "both"],
        help="compact=紧凑表+np y（默认）；wide=原宽表 emb_000+sales_wk；both=两种都写",
    )
    parser.add_argument(
        "--compact_storage",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="紧凑表落盘格式：parquet（默认，保留 list 列类型，需 pyarrow）；csv 为文本表",
    )
    parser.add_argument(
        "--ablate_trends",
        type=int,
        default=0,
        help="1=快速近似消融：导出 embedding 时将每个 batch 的 gtrends 置零后再前向；0=正常输入",
    )
    parser.add_argument(
        "--ablation2_ccf_img",
        type=int,
        default=0,
        help="1=快速近似消融2：仅保留 category/color/fabric + image 信号；"
        "导出 embedding 时将 gtrends/temporal_features/recent_sales_2w 置零；0=关闭",
    )

    args = parser.parse_args()
    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError(f"--train_frac must be in (0, 1], got {args.train_frac}")

    # device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, but this project generates masks on cuda. Please run with GPU or adjust model code.")
    device_str = f"cuda:{args.gpu_num}" if args.device.startswith("cuda") else "cpu"
    args.device = torch.device(device_str)

    # load csv embeddings metadata (and dicts/models inputs)
    cat_dict = _torch_load_trusted(os.path.join(args.data_folder, "category_labels.pt"))
    col_dict = _torch_load_trusted(os.path.join(args.data_folder, "color_labels.pt"))
    fab_dict = _torch_load_trusted(os.path.join(args.data_folder, "fabric_labels.pt"))

    args.cat_dict = cat_dict
    args.col_dict = col_dict
    args.fab_dict = fab_dict

    # Load gtrends
    args.gtrends = pd.read_csv(
        os.path.join(args.data_folder, "gtrends.csv"),
        index_col=[0],
        parse_dates=True,
    )

    # Load checkpoint + hyper parameters (if available)
    ckpt = _torch_load_trusted(args.checkpoint)
    hparams = _extract_hparams(ckpt)

    def get_h(k, default):
        return hparams.get(k, default)

    if hparams.get("train_frac") is not None and abs(
        float(hparams["train_frac"]) - float(args.train_frac)
    ) > 1e-9:
        print(
            f"Note: checkpoint hyper_parameters train_frac={hparams['train_frac']} "
            f"!= CLI --train_frac={args.train_frac}"
        )
    if hparams.get("seed") is not None and int(hparams["seed"]) != int(args.seed):
        print(f"Note: checkpoint hyper_parameters seed={hparams['seed']} != CLI --seed={args.seed}")

    model = GTM(
        embedding_dim=get_h("embedding_dim", args.embedding_dim),
        hidden_dim=get_h("hidden_dim", args.hidden_dim),
        output_dim=get_h("output_dim", args.output_dim),
        num_heads=get_h("num_heads", args.num_attn_heads),
        num_layers=get_h("num_layers", args.num_hidden_layers),
        use_text=get_h("use_text", args.use_text),
        use_img=get_h("use_img", args.use_img),
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=get_h("trend_len", args.trend_len),
        num_trends=get_h("num_trends", args.num_trends),
        gpu_num=args.gpu_num,
        use_encoder_mask=get_h("use_encoder_mask", args.use_encoder_mask),
        autoregressive=get_h("autoregressive", args.autoregressive),
        use_hist_sales=get_h("use_hist_sales", args.use_hist_sales),
    )
    model.to(args.device)
    _load_checkpoint(model, args.checkpoint)

    # Ensure trend_len is available for dataset
    args.trend_len = get_h("trend_len", args.trend_len)

    splits_to_run: List[str]
    if args.split == "all":
        splits_to_run = ["train", "test", "all"]
    else:
        splits_to_run = [args.split]

    for sp in splits_to_run:
        if sp == "all":
            train_df = _read_split_df(args.data_folder, "train")
            train_df = sample_train_df(train_df, args.train_frac, args.seed)
            test_df = _read_split_df(args.data_folder, "test")
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df["_compact_split"] = "train"
            test_df["_compact_split"] = "test"
            df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            export_for_df(split_name="all", df=df_all, args=args, model=model)
        else:
            export_for_split(sp, args, model)

    print("Export finished.")


if __name__ == "__main__":
    main()

