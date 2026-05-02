"""
Train a metric-learning projector on frozen train item embeddings.

Composite loss (optional):
  loss = topk_loss_coef * L_topk + lambda_metric * L_metric

- L_metric: MSE between cosine(z_i,z_j) and Pearson(curve_i, curve_j) (random pairs).
- L_topk: aligns with similarity_wape_pipeline static forecast — cosine similarity to a
  random train pool, take Top-K (torch.topk), normalize weights like weighted_forecast(),
  weighted mean of neighbor curves, MSE vs query curve (week slice aligns WAPE start_week).

Does NOT compute GTM decoder FC loss; that stays in GTM.train.
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
import torch.nn.functional as F
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
from utils.curve_eval_metrics import avg_pearson_avg_dtw
from utils.ref_group_curve_stats import (
    ref_group_flat_means_from_summary,
    ref_group_stats_from_train_export_table,
    summary_stats_to_jsonable,
    train_export_ref_group_skip_reason,
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


def load_train_df_for_projector(
    csv_path: str,
    train_frac: float,
    train_sample_seed: int,
    max_train_rows: int | None,
) -> pd.DataFrame:
    """
    与 export_item_embeddings 对齐，返回与 npy 行对齐的 DataFrame（再由此取 curves）。
    """
    if train_frac < 1.0:
        df = _read_tabular(csv_path, nrows=None)
        df = sample_train_df(df, train_frac, train_sample_seed)
        if max_train_rows is not None:
            df = df.iloc[: int(max_train_rows)].reset_index(drop=True)
    else:
        df = _read_tabular(csv_path, nrows=max_train_rows)
    return df.reset_index(drop=True)


def load_train_curves_for_projector(
    csv_path: str,
    train_frac: float,
    train_sample_seed: int,
    max_train_rows: int | None,
) -> np.ndarray:
    """见 load_train_df_for_projector；仅返回 (N, 12) 曲线矩阵。"""
    df = load_train_df_for_projector(csv_path, train_frac, train_sample_seed, max_train_rows)
    return _curves_array_from_df(df)


def sample_disjoint_pairs(rng: np.random.Generator, n: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Random pairs (i, j) with i != j."""
    idx_i = rng.integers(0, n, size=batch, dtype=np.int64)
    idx_j = rng.integers(0, n, size=batch, dtype=np.int64)
    for t in range(batch):
        while idx_j[t] == idx_i[t]:
            idx_j[t] = int(rng.integers(0, n))
    return idx_i, idx_j


def sample_query_and_pool_indices(
    rng: np.random.Generator,
    n: int,
    batch_queries: int,
    pool_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Query rows (pseudo test) and support pool rows (pseudo train archive). May overlap; masked in loss."""
    b = min(int(batch_queries), n)
    m = min(int(pool_size), n)
    if b < 1 or m < 1:
        raise ValueError("batch_queries and pool_size must allow at least one row each")
    query_idx = rng.choice(n, size=b, replace=False).astype(np.int64)
    pool_idx = rng.choice(n, size=m, replace=False).astype(np.int64)
    return query_idx, pool_idx


def topk_weighted_curve_mse(
    z_q: torch.Tensor,
    z_pool: torch.Tensor,
    curves_pool: torch.Tensor,
    curves_query: torch.Tensor,
    pool_global_idx: torch.Tensor,
    query_global_idx: torch.Tensor,
    top_k: int,
    start_idx: int,
) -> torch.Tensor:
    """
    Same structure as similarity_wape_pipeline.build_similarity_refs (cosine) +
    weighted_forecast (normalize weights, weighted average of curves).

    z_q: (B, D), z_pool: (M, D) L2-normalized; cosine sim = z_q @ z_pool.T
    start_idx: 0-based column index; default 1 matches calc_mae_wape(..., start_week=2).
    """
    sim = z_q @ z_pool.T
    mask_self = pool_global_idx.unsqueeze(0) == query_global_idx.unsqueeze(1)
    sim = sim.masked_fill(mask_self, float("-inf"))
    k_eff = min(int(top_k), sim.shape[1])
    vals, idx = torch.topk(sim, k=k_eff, dim=1)
    weights = vals
    w_sum = weights.sum(dim=1, keepdim=True)
    uniform = torch.full_like(weights, 1.0 / float(k_eff))
    weights = torch.where(w_sum > 0, weights / w_sum.clamp(min=1e-12), uniform)

    c_top = curves_pool[idx]
    pred = (weights.unsqueeze(-1) * c_top).sum(dim=1)
    target = curves_query
    if start_idx > 0:
        pred = pred[:, start_idx:]
        target = target[:, start_idx:]
    return F.mse_loss(pred, target)


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
    topk_loss_coef: float = 0.0,
    lambda_metric: float = 1.0,
    top_k: int = 20,
    pool_size: int = 512,
    topk_query_batch: int = 32,
    curve_loss_start_idx: int = 1,
) -> tuple[list[float], list[float], list[float]]:
    """
    Returns (total_losses, metric_losses, topk_losses) per epoch (mean over steps).
    topk_loss matches similarity_wape_pipeline: Top-K by cosine, weighted_forecast blend, MSE on curve tail.
    """
    rng = np.random.default_rng(seed)
    opt = optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]

    total_losses: list[float] = []
    metric_losses_e: list[float] = []
    topk_losses_e: list[float] = []

    use_topk = topk_loss_coef > 0.0
    use_metric = lambda_metric > 0.0
    if not use_topk and not use_metric:
        raise ValueError("At least one of topk_loss_coef or lambda_metric must be positive")

    model.train()
    for ep in range(epochs):
        ep_sum = 0.0
        ep_metric = 0.0
        ep_topk = 0.0
        n_batches = 0

        for _ in tqdm(range(steps_per_epoch), desc=f"epoch {ep+1}/{epochs}", leave=False):
            l_metric_t = torch.tensor(0.0, device=device)
            l_topk_t = torch.tensor(0.0, device=device)
            parts: list[torch.Tensor] = []

            if use_metric:
                idx_i, idx_j = sample_disjoint_pairs(rng, n, batch_pairs)
                sim_curve = pearson_r_batch(curves, idx_i, idx_j)
                valid = np.isfinite(sim_curve)
                if np.any(valid):
                    xi = torch.from_numpy(X[idx_i]).float().to(device)
                    xj = torch.from_numpy(X[idx_j]).float().to(device)
                    sc = torch.tensor(sim_curve, dtype=torch.float32, device=device)
                    zi = model(xi)
                    zj = model(xj)
                    sim_emb = (zi * zj).sum(dim=-1)
                    err = (sim_emb - sc) ** 2
                    valid_t = torch.from_numpy(valid).to(device)
                    l_metric_t = err[valid_t].mean()
                    parts.append(lambda_metric * l_metric_t)

            if use_topk:
                q_idx, p_idx = sample_query_and_pool_indices(
                    rng, n, topk_query_batch, pool_size
                )
                x_q = torch.from_numpy(X[q_idx]).float().to(device)
                x_p = torch.from_numpy(X[p_idx]).float().to(device)
                z_q = model(x_q)
                z_p = model(x_p)
                cq = torch.from_numpy(curves[q_idx]).float().to(device)
                cp = torch.from_numpy(curves[p_idx]).float().to(device)
                qg = torch.tensor(q_idx, device=device, dtype=torch.long)
                pg = torch.tensor(p_idx, device=device, dtype=torch.long)
                l_topk_t = topk_weighted_curve_mse(
                    z_q,
                    z_p,
                    cp,
                    cq,
                    pg,
                    qg,
                    top_k=top_k,
                    start_idx=curve_loss_start_idx,
                )
                parts.append(topk_loss_coef * l_topk_t)

            if not parts:
                continue

            loss = sum(parts)
            if torch.isnan(loss):
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_sum += float(loss.item())
            ep_metric += float(l_metric_t.detach().item())
            ep_topk += float(l_topk_t.detach().item())
            n_batches += 1

        denom = max(n_batches, 1)
        total_losses.append(ep_sum / denom)
        metric_losses_e.append(ep_metric / denom)
        topk_losses_e.append(ep_topk / denom)
        print(
            f"epoch {ep+1} mean loss: {total_losses[-1]:.6f}  "
            f"(metric: {metric_losses_e[-1]:.6f}, topk_wape_pipe: {topk_losses_e[-1]:.6f})"
        )

    return total_losses, metric_losses_e, topk_losses_e


@torch.no_grad()
def encode_all_embeddings(
    model: CurveMetricProjector,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """将训练集输入矩阵投影为 L2 归一化后的 (N, out_dim)。"""
    model.eval()
    x_np = np.asarray(X, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for i0 in range(0, x_np.shape[0], batch_size):
        chunk = torch.from_numpy(x_np[i0 : i0 + batch_size]).to(device)
        chunks.append(model(chunk).float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def nn_indices_max_cosine(Z: np.ndarray, batch_rows: int) -> np.ndarray:
    """
    Z: (N, D) 已为 L2 归一化。返回每个 i 在 j!=i 上余弦相似度最大的 j。
    用于「用 embedding 最近邻的 train 曲线」近似检索质量，与 Avg_Pearson / Avg_DTW 配套。
    """
    z = np.asarray(Z, dtype=np.float32)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    n = z.shape[0]
    nn_idx = np.empty(n, dtype=np.int64)
    zt = z.T
    for i0 in range(0, n, batch_rows):
        i1 = min(n, i0 + batch_rows)
        sim = z[i0:i1] @ zt
        loc = np.arange(i0, i1)
        sim[np.arange(loc.size), loc] = -np.inf
        nn_idx[i0:i1] = np.argmax(sim, axis=1).astype(np.int64)
    return nn_idx


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
    parser.add_argument(
        "--eval_wsl",
        type=int,
        default=2,
        help="训练结束后 1-NN 曲线评估：窗口起点索引（含），与 curve_eval_metrics 一致",
    )
    parser.add_argument(
        "--eval_tw",
        type=int,
        default=None,
        help="窗口终点索引（不含）；默认 None 表示用满 12 周",
    )
    parser.add_argument(
        "--no_nn_curve_metrics",
        action="store_true",
        help="关闭训练集上的 embedding 1-NN 邻居曲线 vs 自身的 Avg_Pearson / Avg_DTW",
    )
    parser.add_argument("--encode_batch_size", type=int, default=4096, help="全量投影时的 batch 大小")
    parser.add_argument(
        "--nn_sim_batch_rows",
        type=int,
        default=512,
        help="计算逐样本余弦最近邻时的行块大小（控制峰值内存）",
    )
    parser.add_argument(
        "--train_export_ref_group_metrics",
        action="store_true",
        help="极少用：在「一行一 SKU」的 train 导出表上算组统计（多数量弱）。"
        "与 exploration 一致的 Avg_sim/Avg_corr/CV 应对 similarity 的 final_ref 算，见 similarity_wape_pipeline --save_prefix。",
    )
    parser.add_argument(
        "--topk_loss_coef",
        type=float,
        default=0.0,
        help="similarity_wape_pipeline 风格：随机 train 子集为 pool，对 query 做 cos Top-K + weighted_forecast 加权曲线，"
        "与自身曲线算 MSE；0 表示关闭（仅 Pearson metric）。",
    )
    parser.add_argument(
        "--lambda_metric",
        type=float,
        default=1.0,
        help="Pearson–cosine MSE 项权重；(cos-r)^2。可与 topk_loss_coef 组合；二者均为 0 非法。",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="与 similarity_wape_pipeline / build_similarity_refs 中 Top-K 邻居数一致（训练时每步在 pool 内 topk）。",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=512,
        help="每步从 train 中无放回抽样的候选池大小（模拟检索库子集；越大越接近全库但更慢）。",
    )
    parser.add_argument(
        "--topk_query_batch",
        type=int,
        default=32,
        help="每步伪 query 条数（并行计算 Top-K 加权损失）。",
    )
    parser.add_argument(
        "--curve_loss_start_idx",
        type=int,
        default=1,
        help="曲线 MSE 起始列（0-based）。默认 1 对应 calc_mae_wape(..., start_week=2) 的周切片。",
    )
    args = parser.parse_args()

    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError(f"--train_frac must be in (0, 1], got {args.train_frac}")
    if args.topk_loss_coef <= 0 and args.lambda_metric <= 0:
        raise ValueError("Require --topk_loss_coef > 0 or --lambda_metric > 0")

    emb = np.load(args.train_embeddings_npy)
    train_df = load_train_df_for_projector(
        args.train_curves_csv,
        train_frac=args.train_frac,
        train_sample_seed=args.train_sample_seed,
        max_train_rows=args.max_train_rows,
    )
    curves = _curves_array_from_df(train_df)

    ref_group_stats_json: dict[str, dict[str, float | None]] | None = None
    ref_group_n_groups: int | None = None
    if args.train_export_ref_group_metrics:
        rg = ref_group_stats_from_train_export_table(
            train_df, wsl=int(args.eval_wsl), tw=args.eval_tw
        )
        if rg is not None:
            rg_df, rg_summary = rg
            ref_group_stats_json = summary_stats_to_jsonable(rg_summary)
            ref_group_n_groups = int(len(rg_df))
            print("[ref_group/train_export] 汇总（写入 config.json）")
            print(rg_summary.to_string())
        else:
            msg = train_export_ref_group_skip_reason(train_df)
            print(f"[ref_group/train_export] 跳过: {msg or '未知原因'}")
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
        topk_loss_coef=float(args.topk_loss_coef),
        lambda_metric=float(args.lambda_metric),
        top_k=int(args.top_k),
        pool_size=int(args.pool_size),
        topk_query_batch=int(args.topk_query_batch),
        curve_loss_start_idx=int(args.curve_loss_start_idx),
    )

    nn_avg_pearson: float | None = None
    nn_avg_dtw: float | None = None
    if not args.no_nn_curve_metrics:
        Z = encode_all_embeddings(
            model, X, device, batch_size=int(args.encode_batch_size)
        )
        nn_j = nn_indices_max_cosine(Z, batch_rows=int(args.nn_sim_batch_rows))
        pred_c = curves[nn_j].astype(np.float64, copy=False)
        gt_c = curves.astype(np.float64, copy=False)
        nn_avg_pearson, nn_avg_dtw = avg_pearson_avg_dtw(
            gt_c, pred_c, wsl=int(args.eval_wsl), tw=args.eval_tw
        )
        tw_disp = args.eval_tw if args.eval_tw is not None else curves.shape[1]
        print(
            f"[train 1-NN curve vs self] Avg_Pearson={nn_avg_pearson:.4f}  "
            f"Avg_DTW={nn_avg_dtw:.4f}  (window [{args.eval_wsl}:{tw_disp}), N={curves.shape[0]})"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, out_dir / "projector.pt")
    if pca is not None:
        joblib.dump(pca, out_dir / "pca_model.joblib")

    flat_ref_means = (
        ref_group_flat_means_from_summary(ref_group_stats_json)
        if ref_group_stats_json
        else {}
    )
    config: dict = {
        "raw_embedding_dim": raw_dim,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "output_dim": args.out_dim,
        "use_pca": pca is not None,
        "pca_n_components": int(pca.n_components_) if pca is not None else 0,
        "topk_loss_coef": float(args.topk_loss_coef),
        "lambda_metric": float(args.lambda_metric),
        "top_k": int(args.top_k),
        "pool_size": int(args.pool_size),
        "topk_query_batch": int(args.topk_query_batch),
        "curve_loss_start_idx": int(args.curve_loss_start_idx),
        "train_frac": float(args.train_frac),
        "train_sample_seed": int(args.train_sample_seed),
        "max_train_rows": args.max_train_rows,
        "nn_curve_eval_wsl": int(args.eval_wsl),
        "nn_curve_eval_tw": args.eval_tw,
        "nn_curve_avg_pearson": nn_avg_pearson,
        "nn_curve_avg_dtw": nn_avg_dtw,
        "neighbor_ref_group_metrics": (
            "在 similarity_wape_pipeline 对 test 检索 train 后，由 final_ref_df 计算；"
            "运行: python similarity_wape_pipeline.py ... --save_prefix <前缀>，"
            "将生成 <前缀>_ref_group_summary.json 与 <前缀>_ref_group_per_sku.csv"
        ),
    }
    if ref_group_stats_json is not None:
        config["ref_group_on_train_export"] = True
        config["ref_group_stats_wsl"] = int(args.eval_wsl)
        config["ref_group_stats_tw"] = args.eval_tw
        config["ref_group_n_groups"] = ref_group_n_groups
        config.update(flat_ref_means)
        config["ref_group_stats"] = ref_group_stats_json
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
