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


def run_similarity_wape(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    top_k: int = 20,
    start_week: int = 2,
    embedding_col: str = "embedding",
    curve_col: str = "curve",
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    一站式执行:
    - final_ref_df: 邻居明细
    - forecast_df: 预测结果
    - mae, wape: 指标
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
    return final_ref_df, forecast_df, mae, wape


def _parse_embedding_cell(value) -> np.ndarray:
    """将 embedding 单元格解析为 np.ndarray。"""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=float)
    raise ValueError(f"无法解析 embedding 值类型: {type(value)}")


def prepare_input_from_csv(
    csv_path: str,
    embedding_col: str = "embedding",
    sales_prefix: str = "sales_wk_",
) -> pd.DataFrame:
    """
    读取 embedding CSV，并生成标准输入列:
    [external_code, retail, embedding, curve]
    """
    df = pd.read_csv(csv_path)

    required = {"external_code", "retail", embedding_col}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{csv_path} 缺少列: {sorted(miss)}")

    week_cols = sorted(
        [c for c in df.columns if c.startswith(sales_prefix)],
        key=lambda x: int(x.replace(sales_prefix, "")),
    )
    if not week_cols:
        raise ValueError(f"{csv_path} 中未找到 `{sales_prefix}*` 列。")

    out = df.copy()
    out["embedding"] = out[embedding_col].apply(_parse_embedding_cell)
    out["curve"] = out[week_cols].astype(float).values.tolist()
    out = out[["external_code", "retail", "embedding", "curve"]].copy()
    return out


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
        help="训练集 embedding CSV 路径",
    )
    parser.add_argument(
        "--test_csv",
        default="test_item_embeddings.csv",
        help="测试集 embedding CSV 路径",
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
    args = parser.parse_args()

    train_df = prepare_input_from_csv(args.train_csv)
    test_df = prepare_input_from_csv(args.test_csv)

    final_ref_df, forecast_df, mae, wape = run_similarity_wape(
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

    compare_topk_list = _parse_topk_list(args.compare_topk)
    if compare_topk_list:
        print("\n=== WAPE Compare ===")
        for k in compare_topk_list:
            _, fc_k, mae_k, wape_k = run_similarity_wape(
                test_df=test_df,
                train_df=train_df,
                top_k=k,
                start_week=args.start_week,
            )
            print(
                f"Top{k:<2} -> Samples: {len(fc_k):>3}, "
                f"MAE: {mae_k:.3f}, WAPE: {wape_k:.3f}%"
            )

    if args.save_prefix:
        ref_path = f"{args.save_prefix}_final_ref.csv"
        forecast_path = f"{args.save_prefix}_forecast.csv"
        final_ref_df.to_csv(ref_path, index=False)
        forecast_df.to_csv(forecast_path, index=False)
        print(f"Saved: {ref_path}")
        print(f"Saved: {forecast_path}")
