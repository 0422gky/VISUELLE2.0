"""
Train a metric-learning projector on frozen train item embeddings.
Supervision: MSE between cosine(z_i, z_j) and Pearson(curve_i, curve_j).
Uses train split only (no test leakage).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.curve_metric_projector import (
    CurveMetricProjector,
    fit_pca_on_train,
    pearson_r_batch,
    transform_with_pca,
)


def _read_tabular(path: str, nrows: int | None = None) -> pd.DataFrame:
    """读取 CSV 或 Parquet（与 export compact 一致）。"""
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.head(int(nrows)).reset_index(drop=True)
        return df
    read_kw: dict = {}
    if nrows is not None:
        read_kw["nrows"] = int(nrows)
    return pd.read_csv(path, **read_kw)


def sample_train_df(df: pd.DataFrame, train_frac: float, seed: int) -> pd.DataFrame:
    """与 export_item_embeddings / train.py 一致：仅对 train 子集抽样。"""
    if not (0.0 < train_frac <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")
    if train_frac < 1.0:
        return df.sample(frac=train_frac, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def _stack_curve_series(series: pd.Series) -> np.ndarray:
    """compact 导出中 curve 列为每行 list[float]，堆成 (N, 12)。"""
    arr = np.stack(series.apply(lambda x: np.asarray(x, dtype=np.float32)).values)
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise ValueError(f"Expected curve column -> (N, 12), got {arr.shape}")
    return arr


def _curves_array_from_df(df: pd.DataFrame) -> np.ndarray:
    """从 DataFrame 提取 (N, 12) 销量曲线：curve / *_curve / sales_wk_* / '0'..'11' / 前 12 列。"""
    if "curve" in df.columns:
        return _stack_curve_series(df["curve"])
    curve_cols = [c for c in df.columns if c.endswith("_curve")]
    if len(curve_cols) == 1:
        return _stack_curve_series(df[curve_cols[0]])
    if len(curve_cols) > 1:
        raise ValueError(f"Multiple *_curve columns, pick one: {curve_cols}")

    sales = [c for c in df.columns if c.startswith("sales_wk_")]
    if len(sales) >= 12:
        sales = sorted(sales, key=lambda x: int(x.replace("sales_wk_", "")))[:12]
        return df[sales].to_numpy(dtype=np.float32)
    str_cols = [str(i) for i in range(12)]
    if all(c in df.columns for c in str_cols):
        return df[str_cols].to_numpy(dtype=np.float32)
    if df.shape[1] >= 12:
        return df.iloc[:, :12].to_numpy(dtype=np.float32)
    raise ValueError("Cannot infer 12 week columns from dataframe")


def load_curves_12w(csv_path: str, nrows: int | None = None) -> np.ndarray:
    """Load (N, 12) sales curves; supports Parquet/CSV, compact curve 列或宽表周列。"""
    df = _read_tabular(csv_path, nrows=nrows)
    return _curves_array_from_df(df)


def load_train_curves_for_projector(
    csv_path: str,
    train_frac: float,
    train_sample_seed: int,
    max_train_rows: int | None,
) -> np.ndarray:
    """
    与 export_item_embeddings 对齐：
    - train_frac<1 时先全表读入再 sample（需与导出 npy 使用相同 frac/seed）；
    - train_frac==1 且 max_train_rows 时可用 nrows 只读前 N 行（省内存）。
    """
    if train_frac < 1.0:
        df = _read_tabular(csv_path, nrows=None)
        df = sample_train_df(df, train_frac, train_sample_seed)
        if max_train_rows is not None:
            df = df.iloc[: int(max_train_rows)].reset_index(drop=True)
    else:
        df = _read_tabular(csv_path, nrows=max_train_rows)
    return _curves_array_from_df(df)


def sample_disjoint_pairs(rng: np.random.Generator, n: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Random pairs (i, j) with i != j."""
    idx_i = rng.integers(0, n, size=batch, dtype=np.int64)
    idx_j = rng.integers(0, n, size=batch, dtype=np.int64)
    for t in range(batch):
        while idx_j[t] == idx_i[t]:
            idx_j[t] = int(rng.integers(0, n))
    return idx_i, idx_j


def train_loop(
    model: CurveMetricProjector,
    X: np.ndarray,
    curves: np.ndarray,
    device: torch.device,
    epochs: int,
    steps_per_epoch: int,
    batch_pairs: int,
    lr: float,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    opt = optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    losses: list[float] = []

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        for _ in tqdm(range(steps_per_epoch), desc=f"epoch {ep+1}/{epochs}", leave=False):
            idx_i, idx_j = sample_disjoint_pairs(rng, n, batch_pairs)
            sim_curve = pearson_r_batch(curves, idx_i, idx_j)
            valid = np.isfinite(sim_curve)
            if not np.any(valid):
                continue

            xi = torch.from_numpy(X[idx_i]).float().to(device)
            xj = torch.from_numpy(X[idx_j]).float().to(device)
            sc = torch.tensor(sim_curve, dtype=torch.float32, device=device)

            opt.zero_grad()
            zi = model(xi)
            zj = model(xj)
            sim_emb = (zi * zj).sum(dim=-1)
            err = (sim_emb - sc) ** 2
            valid_t = torch.from_numpy(valid).to(device)
            loss = err[valid_t].mean()
            if torch.isnan(loss):
                continue
            loss.backward()
            opt.step()

            ep_loss += float(loss.item())
            n_batches += 1

        avg = ep_loss / max(n_batches, 1)
        losses.append(avg)
        print(f"epoch {ep+1} mean loss: {avg:.6f}")

    return losses


def main():
    parser = argparse.ArgumentParser(description="Train curve-aligned metric projector")
    parser.add_argument(
        "--train_embeddings_npy",
        type=str,
        required=True,
        help="Path to train_item_embeddings.npy (N, D), frozen GTM",
    )
    parser.add_argument(
        "--train_curves_csv",
        type=str,
        required=True,
        help="与 npy 行对齐的表：CSV 或 Parquet（如 export 的 train_item_embeddings.csv / .parquet，含 curve 或 train_curve 列）",
    )
    parser.add_argument("--output_dir", type=str, default="results/curve_projector", help="Save projector + PCA + config")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--batch_pairs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument(
        "--pca_components",
        type=int,
        default=0,
        help="If >0, fit PCA on train embeddings with this many components before MLP",
    )
    parser.add_argument(
        "--max_train_rows",
        type=int,
        default=None,
        help="在 train_frac 抽样之后（若未抽样则在全表上）再只取前 N 行；与 npy 行数需一致。快速测试示例: 5000",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=1.0,
        help="与 export_item_embeddings --train_frac 一致：仅当 train_curves_csv 为原始 train 表且需与导出 npy 对齐时使用；已导出的对齐 CSV 请保持 1.0",
    )
    parser.add_argument(
        "--train_sample_seed",
        type=int,
        default=21,
        help="与 export_item_embeddings --seed 一致（train_frac<1 时的 random_state）",
    )
    args = parser.parse_args()

    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError(f"--train_frac must be in (0, 1], got {args.train_frac}")

    emb = np.load(args.train_embeddings_npy)
    curves = load_train_curves_for_projector(
        args.train_curves_csv,
        train_frac=args.train_frac,
        train_sample_seed=args.train_sample_seed,
        max_train_rows=args.max_train_rows,
    )
    if args.train_frac < 1.0 or args.max_train_rows is not None:
        print(
            f"Curves rows={curves.shape[0]} (train_frac={args.train_frac}, "
            f"train_sample_seed={args.train_sample_seed}, max_train_rows={args.max_train_rows})"
        )
    if emb.shape[0] != curves.shape[0]:
        raise ValueError(
            f"Row count mismatch: embeddings {emb.shape[0]} vs curves {curves.shape[0]} "
            f"(请确认 train_embeddings_npy 与 train_curves_csv 使用同一套 train_frac/seed，或改用 export 产出的对齐 CSV)"
        )
    if curves.shape[1] != 12:
        raise ValueError(f"Expected curves (N, 12), got {curves.shape}")

    raw_dim = emb.shape[1]
    pca = None
    pca_n = args.pca_components
    if pca_n > 0:
        pca = fit_pca_on_train(emb, n_components=pca_n, random_state=args.seed)
        if pca is None:
            raise RuntimeError("PCA fit failed")
        X = transform_with_pca(pca, emb)
        input_dim = X.shape[1]
        print(f"PCA: raw_dim={raw_dim} -> {input_dim}")
    else:
        X = emb.astype(np.float32)
        input_dim = raw_dim

    device = torch.device(args.device)
    model = CurveMetricProjector(input_dim=input_dim, hidden_dim=args.hidden_dim, output_dim=args.out_dim)
    model.to(device)

    train_loop(
        model=model,
        X=X,
        curves=curves,
        device=device,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_pairs=args.batch_pairs,
        lr=args.lr,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, out_dir / "projector.pt")
    if pca is not None:
        joblib.dump(pca, out_dir / "pca_model.joblib")

    config = {
        "raw_embedding_dim": raw_dim,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.out_dim,
        "use_pca": pca is not None,
        "pca_n_components": int(pca.n_components_) if pca is not None else 0,
        "train_frac": float(args.train_frac),
        "train_sample_seed": int(args.train_sample_seed),
        "max_train_rows": args.max_train_rows,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
