"""
从 exploratio_v0.ipynb 中整理的“通过相似度计算 WAPE”核心流程。

流程:
1) 用 embedding 计算 test 对 train 的余弦相似度
2) 为每个 test 样本选 Top-K 相似邻居
3) 用相似度作为权重，对邻居曲线做加权平均，得到预测曲线
4) 按周切片后计算 MAE / WAPE

输入要求:
- train_df: 至少包含 [external_code, retail, embedding, curve]
- test_df:  至少包含 [external_code, retail, embedding, curve]
- embedding: list/np.ndarray，形状一致
- curve:     list/np.ndarray，例如 12 周曲线
"""

from __future__ import annotations

import argparse
import ast
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.curve_eval_metrics import avg_pearson_avg_dtw
from utils.ref_group_curve_stats import (
    ref_group_flat_means_from_summary,
    ref_group_stats_from_final_ref_df,
    save_ref_group_artifacts,
    summary_stats_to_jsonable,
)


def _to_2d_array(series: pd.Series, name: str) -> np.ndarray:
    """将 DataFrame 某列(list/ndarray)安全堆叠为二维矩阵。"""
    try:
        arr = np.stack(series.values)
    except Exception as exc:
        raise ValueError(f"列 `{name}` 无法堆叠为二维数组，请检查每行长度是否一致。") from exc
    return arr


def build_similarity_refs(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    top_k: int = 20,
    embedding_col: str = "embedding",
    curve_col: str = "curve",
) -> pd.DataFrame:
    """
    计算 test->train 的 Top-K 相似邻居，并返回明细表 final_ref_df。

    返回列包含:
    [external_code, retail, curve, sim_score, test_code, test_retail, test_curve]
    """
    required = {"external_code", "retail", embedding_col, curve_col}
    miss_train = required - set(train_df.columns)
    miss_test = required - set(test_df.columns)
    if miss_train:
        raise ValueError(f"train_df 缺少列: {sorted(miss_train)}")
    if miss_test:
        raise ValueError(f"test_df 缺少列: {sorted(miss_test)}")

    test_emb = _to_2d_array(test_df[embedding_col], embedding_col)
    train_emb = _to_2d_array(train_df[embedding_col], embedding_col)
    sim_matrix = cosine_similarity(test_emb, train_emb)

    k = min(top_k, len(train_df))
    top_indices = np.argsort(sim_matrix, axis=1)[:, :-k - 1 : -1]

    results = []
    for i in range(len(test_df)):
        ref_idx = top_indices[i]
        refs = train_df.iloc[ref_idx][["external_code", "retail", curve_col]].copy()

        t_row = test_df.iloc[i]
        refs["sim_score"] = sim_matrix[i, ref_idx]
        refs["test_code"] = t_row["external_code"]
        refs["test_retail"] = t_row["retail"]
        refs["test_curve"] = [t_row[curve_col]] * len(refs)
        refs.rename(columns={curve_col: "curve"}, inplace=True)
        results.append(refs)

    if not results:
        return pd.DataFrame(
            columns=[
                "external_code",
                "retail",
                "curve",
                "sim_score",
                "test_code",
                "test_retail",
                "test_curve",
            ]
        )

    final_ref_df = pd.concat(results, ignore_index=True)
    return final_ref_df


def weighted_forecast(final_ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 final_ref_df 生成每个测试样本的预测曲线。
    输出列: [test_code, test_retail, pred_curve, true_curve]
    """
    required = {"test_code", "test_retail", "curve", "sim_score", "test_curve"}
    miss = required - set(final_ref_df.columns)
    if miss:
        raise ValueError(f"final_ref_df 缺少列: {sorted(miss)}")

    def weighted_mean(group: pd.DataFrame) -> pd.Series:
        curves = np.stack(group["curve"].values)
        weights = group["sim_score"].to_numpy(dtype=float)

        # 权重归一化，防止全零权重
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights), dtype=float) / len(weights)

        pred_curve = np.average(curves, axis=0, weights=weights)
        return pd.Series(
            {
                "pred_curve": pred_curve.tolist(),
                "true_curve": group["test_curve"].iloc[0],
            }
        )

    forecast_df = (
        final_ref_df.groupby(["test_code", "test_retail"], group_keys=False)[
            ["curve", "sim_score", "test_curve"]
        ]
        .apply(weighted_mean)
        .reset_index()
    )
    return forecast_df


def calc_mae_wape(
    gt_list: Iterable[Iterable[float]],
    pred_list: Iterable[Iterable[float]],
    start_week: int = 2,
) -> Tuple[float, float]:
    """
    计算从指定周开始(默认第2周)的全局 MAE 与 WAPE(%)
    - start_week=2 对应切片索引 1:
    """
    gt = np.array(list(gt_list), dtype=float)
    pred = np.array(list(pred_list), dtype=float)

    if gt.shape != pred.shape:
        raise ValueError(f"真实值与预测值 shape 不一致: {gt.shape} vs {pred.shape}")
    if gt.ndim != 2:
        raise ValueError("gt_list/pred_list 应为二维结构: (样本数, 周数)")
    if start_week < 1 or start_week > gt.shape[1]:
        raise ValueError(f"start_week 超范围: {start_week}, 周数={gt.shape[1]}")

    start_idx = start_week - 1
    gt = gt[:, start_idx:]
    pred = pred[:, start_idx:]

    abs_err = np.abs(gt - pred)
    mae = float(np.mean(abs_err))

    total_abs_error = float(np.sum(abs_err))
    total_actual = float(np.sum(gt))
    if total_actual == 0:
        wape = 0.0 if total_abs_error == 0 else 100.0
    else:
        wape = 100.0 * (total_abs_error / total_actual)

    return round(mae, 3), round(wape, 3)


def curve_shape_metrics_from_forecast_df(
    forecast_df: pd.DataFrame,
    start_week: int = 2,
) -> tuple[float, float]:
    """
    与 calc_mae_wape 相同时间窗：从第 start_week 周（1-based）起至序列末尾，
    对 true_curve vs pred_curve 计算 Avg_Pearson、Avg_DTW。
    """
    if forecast_df.empty:
        return 0.0, 0.0
    gt = np.stack([np.asarray(x, dtype=np.float64) for x in forecast_df["true_curve"]])
    pred = np.stack([np.asarray(x, dtype=np.float64) for x in forecast_df["pred_curve"]])
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError(f"forecast_df 曲线 shape 异常: gt {gt.shape}, pred {pred.shape}")
    start_idx = start_week - 1
    if start_week < 1 or start_idx >= gt.shape[1]:
        raise ValueError(f"start_week 超范围: {start_week}, 周数={gt.shape[1]}")
    return avg_pearson_avg_dtw(gt, pred, wsl=start_idx, tw=None)


def run_similarity_wape(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    top_k: int = 20,
    start_week: int = 2,
    embedding_col: str = "embedding",
    curve_col: str = "curve",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    float,
    float,
    float,
    float,
    pd.DataFrame | None,
    pd.DataFrame | None,
]:
    """
    一站式执行:
    - final_ref_df: 邻居明细
    - forecast_df: 预测结果
    - mae, wape: 指标
    - avg_pearson, avg_dtw: 与 mae/wape 同一时间窗的曲线形状指标
    - ref_group_per_sku_df, ref_group_summary_stats: 按每个 test 样本聚合的邻居参考曲线组统计
      （与 exploration test_codes_df 语义一致）；无 final_ref 时为 (None, None)
    """
    final_ref_df = build_similarity_refs(
        test_df=test_df,
        train_df=train_df,
        top_k=top_k,
        embedding_col=embedding_col,
        curve_col=curve_col,
    )
    forecast_df = weighted_forecast(final_ref_df)
    mae, wape = calc_mae_wape(
        gt_list=forecast_df["true_curve"],
        pred_list=forecast_df["pred_curve"],
        start_week=start_week,
    )
    avg_pearson, avg_dtw = curve_shape_metrics_from_forecast_df(forecast_df, start_week=start_week)

    wsl = start_week - 1
    rg = ref_group_stats_from_final_ref_df(final_ref_df, wsl=wsl, tw=None)
    if rg is None:
        ref_group_per_sku_df, ref_group_summary_stats = None, None
    else:
        ref_group_per_sku_df, ref_group_summary_stats = rg

    return (
        final_ref_df,
        forecast_df,
        mae,
        wape,
        avg_pearson,
        avg_dtw,
        ref_group_per_sku_df,
        ref_group_summary_stats,
    )


def _parse_embedding_cell(value) -> np.ndarray:
    """将 embedding 单元格解析为 np.ndarray。"""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=float)
    raise ValueError(f"无法解析 embedding 值类型: {type(value)}")


def attach_embeddings_from_npy(df: pd.DataFrame, emb_npy: str | np.ndarray) -> pd.DataFrame:
    """
    Replace `embedding` column with rows from a (N, D) array; row order must match df.
    Use after prepare_input_from_csv to inject metric-learning projected embeddings.
    If the array has more rows than df (e.g. df was truncated with nrows), uses emb[: len(df)].
    """
    emb = np.load(emb_npy) if isinstance(emb_npy, str) else np.asarray(emb_npy)
    n_df = len(df)
    if len(emb) < n_df:
        raise ValueError(
            f"Row count mismatch: dataframe has {n_df} rows, embedding array has only {len(emb)}"
        )
    if len(emb) > n_df:
        emb = emb[:n_df]
    out = df.copy()
    out["embedding"] = [emb[i].astype(np.float64, copy=False) for i in range(n_df)]
    return out


def _parse_curve_cell(value) -> np.ndarray:
    """将 curve 单元格解析为 np.ndarray。"""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=float)
    raise ValueError(f"无法解析 curve 值类型: {type(value)}")


def sample_train_df(df: pd.DataFrame, train_frac: float, seed: int) -> pd.DataFrame:
    """与 export_item_embeddings / train.py 一致：仅对 train 子集抽样。"""
    if not (0.0 < train_frac <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")
    if train_frac < 1.0:
        return df.sample(frac=train_frac, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def _read_table(path: str, nrows: int | None = None) -> pd.DataFrame:
    read_kw: dict = {}
    if nrows is not None:
        read_kw["nrows"] = int(nrows)
    if path.lower().endswith(".parquet"):
        if nrows is not None:
            # pandas 的 read_parquet 不支持 nrows；这里先全读后截断
            return pd.read_parquet(path).head(int(nrows))
        return pd.read_parquet(path)
    return pd.read_csv(path, **read_kw)


def _pick_prefixed_cols(df: pd.DataFrame) -> tuple[str, str, str, str] | None:
    """
    兼容 compact split 表:
    - train_code, train_retail, train_curve, train_emb
    - test_code, test_retail, test_curve, test_emb
    """
    for col in df.columns:
        if not col.endswith("_code"):
            continue
        prefix = col[: -len("_code")]
        c_code = f"{prefix}_code"
        c_retail = f"{prefix}_retail"
        c_curve = f"{prefix}_curve"
        c_emb = f"{prefix}_emb"
        if all(c in df.columns for c in [c_code, c_retail, c_curve, c_emb]):
            return c_code, c_retail, c_curve, c_emb
    return None


def prepare_input_from_file(
    path: str,
    embedding_col: str = "embedding",
    sales_prefix: str = "sales_wk_",
    nrows: int | None = None,
    sample_frac: float | None = None,
    sample_seed: int = 21,
) -> pd.DataFrame:
    """
    读取 embedding CSV，并生成标准输入列:
    [external_code, retail, embedding, curve]
    nrows: 若给定，在抽样之后（若有）再只保留前 n 行（便于快速测试）。
    sample_frac: 若 <1，先全表读入再按 frac 抽样（需与 export_item_embeddings 的 train_frac/seed 一致）；
                 已导出的对齐表请保持 None 或 1.0。
    """
    if sample_frac is not None and sample_frac < 1.0:
        df = _read_table(path, nrows=None)
        df = sample_train_df(df, sample_frac, sample_seed)
        if nrows is not None:
            df = df.head(int(nrows)).reset_index(drop=True)
    else:
        df = _read_table(path, nrows=nrows)

    out = df.copy()

    # 1) 标准列 / compact(all) 列
    if {"external_code", "retail"}.issubset(out.columns):
        code_col, retail_col = "external_code", "retail"
        if embedding_col in out.columns:
            emb_col = embedding_col
        elif "emb" in out.columns:
            emb_col = "emb"
        else:
            emb_cols = [c for c in out.columns if c.startswith("emb_")]
            if not emb_cols:
                raise ValueError(f"{path} 缺少 embedding 列（`{embedding_col}` / `emb` / `emb_*`）。")
            emb_cols = sorted(emb_cols)
            out["embedding"] = out[emb_cols].astype(float).values.tolist()
            emb_col = "embedding"

        if "curve" in out.columns:
            curve_col = "curve"
        else:
            week_cols = sorted(
                [c for c in out.columns if c.startswith(sales_prefix)],
                key=lambda x: int(x.replace(sales_prefix, "")),
            )
            if not week_cols:
                raise ValueError(f"{path} 缺少 curve 列（`curve` / `{sales_prefix}*`）。")
            out["curve"] = out[week_cols].astype(float).values.tolist()
            curve_col = "curve"

        out["embedding"] = out[emb_col].apply(_parse_embedding_cell)
        out["curve"] = out[curve_col].apply(_parse_curve_cell)
        out = out[[code_col, retail_col, "embedding", "curve"]].copy()
        out.rename(columns={code_col: "external_code", retail_col: "retail"}, inplace=True)
        return out

    # 2) compact split 前缀列 (train_*/test_*)
    picked = _pick_prefixed_cols(out)
    if picked is not None:
        code_col, retail_col, curve_col, emb_col = picked
        out["embedding"] = out[emb_col].apply(_parse_embedding_cell)
        out["curve"] = out[curve_col].apply(_parse_curve_cell)
        out = out[[code_col, retail_col, "embedding", "curve"]].copy()
        out.rename(columns={code_col: "external_code", retail_col: "retail"}, inplace=True)
        return out

    raise ValueError(
        f"{path} 的列结构不受支持。需要以下之一："
        "1) external_code/retail + (embedding|emb|emb_*) + (curve|sales_wk_*)；"
        "2) train_code/train_retail/train_curve/train_emb 或 test_*。"
    )


def _parse_topk_list(raw: str) -> list[int]:
    """将 '1,5,20' 解析为 [1, 5, 20]。"""
    if not raw.strip():
        return []
    values = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        v = int(x)
        if v <= 0:
            raise ValueError("top_k 列表中的值必须为正整数。")
        values.append(v)
    # 去重并排序，输出更稳定
    return sorted(set(values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="相似度检索并计算 WAPE")
    parser.add_argument(
        "--train_csv",
        default="train_item_embeddings.csv",
        help="训练集 embedding 文件路径（CSV/Parquet 均可）",
    )
    parser.add_argument(
        "--test_csv",
        default="test_item_embeddings.csv",
        help="测试集 embedding 文件路径（CSV/Parquet 均可）",
    )
    parser.add_argument("--top_k", type=int, default=20, help="每个测试样本选取的邻居数量")
    parser.add_argument(
        "--start_week",
        type=int,
        default=2,
        help="从第几周开始计算指标(1-based，默认2表示W2开始)",
    )
    parser.add_argument(
        "--save_prefix",
        default="",
        help="若提供前缀，将额外保存明细和预测结果CSV",
    )
    parser.add_argument(
        "--compare_topk",
        default="1,5,20",
        help="输出对比的 Top-K 列表，逗号分隔，例如: 1,5,20",
    )
    parser.add_argument(
        "--train_emb_npy",
        default="",
        help="可选: (N,D) 投影后的 train embedding，行序须与 train_csv 一致，将覆盖 CSV 中的 embedding 列",
    )
    parser.add_argument(
        "--test_emb_npy",
        default="",
        help="可选: (M,D) 投影后的 test embedding，行序须与 test_csv 一致",
    )
    parser.add_argument(
        "--train_nrows",
        type=int,
        default=None,
        help="只读 train 文件前 n 行（默认全部）。快速测试示例: 5000",
    )
    parser.add_argument(
        "--test_nrows",
        type=int,
        default=None,
        help="只读 test 文件前 n 行（默认全部）。快速测试示例: 500",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=1.0,
        help="仅 train：与 export_item_embeddings --train_frac 一致；用已导出对齐表时请保持 1.0",
    )
    parser.add_argument(
        "--train_sample_seed",
        type=int,
        default=21,
        help="与 export_item_embeddings --seed 一致（train_frac<1 时）",
    )
    args = parser.parse_args()

    if not (0.0 < args.train_frac <= 1.0):
        raise ValueError(f"--train_frac must be in (0, 1], got {args.train_frac}")

    train_df = prepare_input_from_file(
        args.train_csv,
        nrows=args.train_nrows,
        sample_frac=args.train_frac if args.train_frac < 1.0 else None,
        sample_seed=args.train_sample_seed,
    )
    test_df = prepare_input_from_file(args.test_csv, nrows=args.test_nrows)
    if args.train_nrows is not None or args.test_nrows is not None:
        print(
            f"Subset rows: train={len(train_df)} (train_nrows={args.train_nrows}), "
            f"test={len(test_df)} (test_nrows={args.test_nrows})"
        )
    if args.train_emb_npy:
        train_df = attach_embeddings_from_npy(train_df, args.train_emb_npy)
    if args.test_emb_npy:
        test_df = attach_embeddings_from_npy(test_df, args.test_emb_npy)

    (
        final_ref_df,
        forecast_df,
        mae,
        wape,
        avg_pearson,
        avg_dtw,
        ref_group_per_sku_df,
        ref_group_summary_stats,
    ) = run_similarity_wape(
        test_df=test_df,
        train_df=train_df,
        top_k=args.top_k,
        start_week=args.start_week,
    )

    print("=== Similarity Forecast Metrics ===")
    print(f"Samples: {len(forecast_df)}")
    print(f"Top-K: {args.top_k}")
    print(f"Global MAE (W{args.start_week}-W12): {mae}")
    print(f"Global WAPE (W{args.start_week}-W12): {wape}%")
    print(f"Avg_Pearson (W{args.start_week}-W12): {avg_pearson:.4f}")
    print(f"Avg_DTW (W{args.start_week}-W12): {avg_dtw:.4f}")

    if ref_group_summary_stats is not None:
        print("\n=== Neighbor ref-curve group stats (per test_code×test_retail, W{}+) ===".format(args.start_week))
        print(ref_group_summary_stats.to_string())
        flat = ref_group_flat_means_from_summary(summary_stats_to_jsonable(ref_group_summary_stats))
        print(f"flat_means: {flat}")

    compare_topk_list = _parse_topk_list(args.compare_topk)
    if compare_topk_list:
        print("\n=== WAPE Compare ===")
        for k in compare_topk_list:
            _, fc_k, mae_k, wape_k, ap_k, dtw_k, _, _ = run_similarity_wape(
                test_df=test_df,
                train_df=train_df,
                top_k=k,
                start_week=args.start_week,
            )
            print(
                f"Top{k:<2} -> Samples: {len(fc_k):>3}, "
                f"MAE: {mae_k:.3f}, WAPE: {wape_k:.3f}%, "
                f"Avg_P: {ap_k:.4f}, Avg_DTW: {dtw_k:.4f}"
            )

    if args.save_prefix:
        ref_path = f"{args.save_prefix}_final_ref.csv"
        forecast_path = f"{args.save_prefix}_forecast.csv"
        final_ref_df.to_csv(ref_path, index=False)
        forecast_df.to_csv(forecast_path, index=False)
        print(f"Saved: {ref_path}")
        print(f"Saved: {forecast_path}")
        if ref_group_per_sku_df is not None and ref_group_summary_stats is not None:
            save_ref_group_artifacts(args.save_prefix, ref_group_per_sku_df, ref_group_summary_stats)
