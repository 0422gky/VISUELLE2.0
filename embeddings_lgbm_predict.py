from pathlib import Path
import argparse
import itertools
import json
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("请先安装 lightgbm：pip install lightgbm") from exc

from similarity_wape_pipeline import (
    build_similarity_refs,
    curve_shape_metrics_from_forecast_df,
    prepare_input_from_file,
)
from utils.ref_group_curve_stats import (
    ref_group_flat_means_from_summary,
    ref_group_stats_from_final_ref_df,
    summary_stats_to_jsonable,
)

WEEK_COLS = [str(i) for i in range(12)]
WEEK_SINCE_LAUNCH = 2
TOTAL_WEEK = 12
PRED_WEEKS = list(range(WEEK_SINCE_LAUNCH, TOTAL_WEEK))


def calc_mae_wape(
    gt_list,
    pred_list,
    start_week: int = 2,
):
    """对齐 similarity_wape_pipeline.py 的全局 MAE / WAPE 计算方式。"""
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


def compute_hit_capture_rate(
    results_df: pd.DataFrame,
    hit_ratio: float = 0.1,
    actual_col: str = "Actual_sum_i",
    pred_col: str = "Pred_sum_i",
) -> Tuple[float, int, int]:
    """
    计算爆款捕捉率。

    规则：
    - 先按实际销量和预测销量分别取前 top_k 个样本
    - top_k = max(1, ceil(n_valid * hit_ratio))
    - 爆款捕捉率 = 两个 top_k 集合的交集 / top_k

    返回：
    - hit_capture_rate: 爆款捕捉率
    - top_k: 截断数量
    - n_valid: 样本数
    """
    if results_df.empty:
        return 0.0, 0, 0

    n_valid = len(results_df)
    top_k = max(1, int(np.ceil(n_valid * hit_ratio)))
    print(f"长度:{n_valid},爆款捕捉top-k:{top_k}")
    actual_top = set(results_df.nlargest(top_k, actual_col).index)
    pred_top = set(results_df.nlargest(top_k, pred_col).index)
    hit_capture_rate = len(actual_top & pred_top) / top_k
    print(f"爆款捕捉率{hit_capture_rate}")
    return float(hit_capture_rate), top_k, n_valid


def build_forecast_df_from_pred_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_curve": pred_df["curve"].values,
            "pred_curve": pred_df["lgbm_pred_curve"].values,
        }
    )


def compute_ref_group_flat_means(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k: int,
    start_week: int,
) -> Tuple[Dict[str, Optional[float]], Optional[pd.DataFrame]]:
    final_ref_df = build_similarity_refs(
        test_df=test_df,
        train_df=train_df,
        top_k=top_k,
        embedding_col="embedding",
        curve_col="curve",
    )
    rg = ref_group_stats_from_final_ref_df(final_ref_df, wsl=start_week - 1, tw=None)
    if rg is None:
        empty: Dict[str, Optional[float]] = {
            "avg_pairwise_corr": None,
            "avg_cv_total": None,
            "avg_cv_week": None,
            "avg_sim": None,
            "avg_neigh": None,
        }
        return empty, None
    _per_sku_df, summary_stats = rg
    flat = ref_group_flat_means_from_summary(summary_stats_to_jsonable(summary_stats))
    return flat, summary_stats


def add_sum_columns(results_df: pd.DataFrame, start_week: int) -> pd.DataFrame:
    start_idx = start_week - 1
    out = results_df.copy()
    out["Actual_sum_i"] = out["curve"].apply(
        lambda curve: float(np.sum(np.asarray(curve, dtype="float64")[start_idx:]))
    )
    out["Pred_sum_i"] = out["lgbm_pred_curve"].apply(
        lambda curve: float(np.sum(np.asarray(curve, dtype="float64")[start_idx:]))
    )
    return out


def _predict_static_long(
    model,
    long_df: pd.DataFrame,
    feature_cols,
    retail_categories,
) -> pd.DataFrame:
    X = long_df[feature_cols].copy()
    X["retail"] = pd.Categorical(X["retail"], categories=retail_categories)

    scored_long = long_df.copy()
    scored_long["pred"] = np.maximum(0.0, model.predict(X))
    return scored_long


def _predict_rolling_2w1w(
    model,
    src_df: pd.DataFrame,
    feature_cols,
    retail_categories,
    teacher_forcing: bool,
) -> pd.DataFrame:
    emb_cols = [c for c in feature_cols if c.startswith("emb_")]
    emb_mat = np.stack(src_df["embedding"].apply(lambda x: np.asarray(x, dtype="float64")).values)
    true_curves = np.stack(src_df["curve"].apply(lambda x: np.asarray(x, dtype="float64")).values)
    pred_curves = true_curves.copy()
    retail_values = src_df["retail"].to_numpy()

    if emb_mat.shape[1] != len(emb_cols):
        raise ValueError(
            f"embedding 维度与特征列数量不一致: emb_dim={emb_mat.shape[1]}, emb_cols={len(emb_cols)}"
        )

    emb_df = pd.DataFrame(emb_mat, columns=emb_cols)
    for week_idx in PRED_WEEKS:
        hist_src = true_curves if teacher_forcing else pred_curves
        w0 = hist_src[:, week_idx - 2].astype("float64")
        w1 = hist_src[:, week_idx - 1].astype("float64")

        X = emb_df.copy()
        X["sales_w0"] = w0
        X["sales_w1"] = w1
        X["sales_first2_sum"] = w0 + w1
        X["sales_first2_mean"] = (w0 + w1) / 2.0
        X["sales_first2_delta"] = w1 - w0
        X["week_idx"] = int(week_idx)
        X["retail"] = pd.Categorical(retail_values, categories=retail_categories)
        X = X[feature_cols]

        pred_curves[:, week_idx] = np.maximum(0.0, model.predict(X))

    pred_df = src_df[["external_code", "retail"]].copy()
    pred_df["curve"] = [row.tolist() for row in true_curves]
    pred_df["lgbm_pred_curve"] = [row.tolist() for row in pred_curves]
    for week_idx in PRED_WEEKS:
        pred_df[f"pred_{week_idx}"] = pred_curves[:, week_idx]
    return pred_df


def build_pred_df_from_model(
    model,
    long_df: pd.DataFrame,
    src_df: pd.DataFrame,
    feature_cols,
    retail_categories,
    start_week: int,
    forecast_mode: str = "static",
    rolling_teacher_forcing: bool = False,
) -> pd.DataFrame:
    if forecast_mode == "rolling_2w1w":
        pred_df = _predict_rolling_2w1w(
            model=model,
            src_df=src_df,
            feature_cols=feature_cols,
            retail_categories=retail_categories,
            teacher_forcing=rolling_teacher_forcing,
        )
    else:
        scored_long = _predict_static_long(
            model=model,
            long_df=long_df,
            feature_cols=feature_cols,
            retail_categories=retail_categories,
        )
        pred_df = assemble_pred_curve(src_df, scored_long)

    pred_df = add_sum_columns(pred_df, start_week=start_week)
    return pred_df


def parse_grid_values(raw_value: str, cast_type):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(cast_type(item))
    return values


def build_long_from_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """将 embedding 数据展平成监督学习的 long 格式。
    输出包含 embedding 展开列、retail、week_idx 及 target（对应 week_idx 的销量）
    要求 df 至少包含: external_code, retail, embedding, curve
    """
    req = {"external_code", "retail", "embedding", "curve"}
    if not req.issubset(df.columns):
        raise ValueError(f"输入表缺少列: {req - set(df.columns)}")

    n = len(df)
    emb_mat = np.stack(df["embedding"].values)
    D = emb_mat.shape[1]

    # 将 embedding 拆成列
    emb_cols = {f"emb_{i}": emb_mat[:, i] for i in range(D)}
    base = df[["external_code", "retail", "curve"]].copy()
    base = pd.concat([base.reset_index(drop=True), pd.DataFrame(emb_cols)], axis=1)

    # 从 curve 中提取前两周统计量
    base["sales_w0"] = base["curve"].apply(lambda x: float(x[0]))
    base["sales_w1"] = base["curve"].apply(lambda x: float(x[1]))
    base["sales_first2_sum"] = base["sales_w0"] + base["sales_w1"]
    base["sales_first2_mean"] = base["sales_first2_sum"] / 2.0
    base["sales_first2_delta"] = base["sales_w1"] - base["sales_w0"]

    rows = []
    for week_idx in PRED_WEEKS:
        part = base.drop(columns=["curve"]).copy()
        part["week_idx"] = int(week_idx)
        part["target"] = base["curve"].apply(lambda arr, wi=week_idx: float(arr[wi]))
        rows.append(part)

    long_df = pd.concat(rows, ignore_index=True)
    return long_df


def assemble_pred_curve(src_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    pred_pivot = long_df.pivot_table(
        index=["external_code"], columns="week_idx", values="pred", aggfunc="mean"
    )
    pred_dict = pred_pivot.to_dict("index")

    out = src_df[["external_code", "retail"]].copy()
    true_curves = np.stack(src_df["curve"].apply(lambda x: np.asarray(x, dtype="float64")).values)
    pred_curves = true_curves.copy()

    for i, row in src_df.iterrows():
        key = row["external_code"]
        week_preds = pred_dict.get(key, {})
        for week_idx in PRED_WEEKS:
            value = week_preds.get(week_idx, None)
            if value is not None and not pd.isna(value):
                pred_curves[i, week_idx] = max(0.0, float(value))

    out["curve"] = [row.tolist() for row in true_curves]
    out["lgbm_pred_curve"] = [row.tolist() for row in pred_curves]
    for week_idx in PRED_WEEKS:
        out[f"pred_{week_idx}"] = pred_curves[:, week_idx]
    return out


def fit_predict_and_score(
    train_long: pd.DataFrame,
    test_long: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    subsample_freq: int,
    colsample_bytree: float,
    reg_lambda: float,
    device_type: str,
    max_bin: int,
    gpu_platform_id: int,
    gpu_device_id: int,
    gpu_use_dp: bool,
    start_week: int,
    hit_ratio: float,
    forecast_mode: str = "static",
    rolling_teacher_forcing: bool = False,
    hit_test_long: Optional[pd.DataFrame] = None,
    hit_test_df: Optional[pd.DataFrame] = None,
):
    X = train_long[feature_cols].copy()
    y = train_long["target"].copy()

    for c in ["retail"]:
        X[c] = X[c].astype("category")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    tr_idx, va_idx = next(gss.split(X, y, groups=train_long["external_code"]))
    X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx]

    for c in ["retail"]:
        X_va[c] = pd.Categorical(X_va[c], categories=X_tr[c].cat.categories)

    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "objective": "regression_l1",
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "subsample_freq": subsample_freq,
        "colsample_bytree": colsample_bytree,
        "reg_lambda": reg_lambda,
        "max_bin": max_bin,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if device_type != "cpu":
        model_params["device_type"] = device_type
        if device_type == "gpu":
            model_params["gpu_platform_id"] = gpu_platform_id
            model_params["gpu_device_id"] = gpu_device_id
            model_params["gpu_use_dp"] = gpu_use_dp

    model = lgb.LGBMRegressor(**model_params)

    model.fit(
        X_tr,
        y_tr,
        categorical_feature=["retail", "week_idx"],
        eval_set=[(X_va, y_va)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)],
    )

    retail_categories = X_tr["retail"].cat.categories
    pred_df = build_pred_df_from_model(
        model=model,
        long_df=test_long,
        src_df=test_df,
        feature_cols=feature_cols,
        retail_categories=retail_categories,
        start_week=start_week,
        forecast_mode=forecast_mode,
        rolling_teacher_forcing=rolling_teacher_forcing,
    )
    mae, wape = calc_mae_wape(
        gt_list=pred_df["curve"],
        pred_list=pred_df["lgbm_pred_curve"],
        start_week=start_week,
    )
    forecast_df = build_forecast_df_from_pred_df(pred_df)
    avg_pearson, avg_dtw = curve_shape_metrics_from_forecast_df(
        forecast_df,
        start_week=start_week,
    )

    hit_source_df = pred_df
    if hit_test_long is not None and hit_test_df is not None:
        hit_source_df = build_pred_df_from_model(
            model=model,
            long_df=hit_test_long,
            src_df=hit_test_df,
            feature_cols=feature_cols,
            retail_categories=retail_categories,
            start_week=start_week,
            forecast_mode=forecast_mode,
            rolling_teacher_forcing=rolling_teacher_forcing,
        )
    hit_capture_rate, top_k, n_valid = compute_hit_capture_rate(
        results_df=hit_source_df,
        hit_ratio=hit_ratio,
    )

    return model, pred_df, mae, wape, avg_pearson, avg_dtw, hit_capture_rate, top_k, n_valid


def run_hparam_analysis(
    train_long: pd.DataFrame,
    test_long: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols,
    random_state: int,
    start_week: int,
    hit_ratio: float,
    n_estimators_grid,
    learning_rate_grid,
    num_leaves_grid,
    min_child_samples_grid,
    subsample_grid,
    subsample_freq_grid,
    colsample_bytree_grid,
    reg_lambda_grid,
    device_type: str,
    max_bin: int,
    gpu_platform_id: int,
    gpu_device_id: int,
    gpu_use_dp: bool,
    forecast_mode: str = "static",
    rolling_teacher_forcing: bool = False,
    hit_test_long: Optional[pd.DataFrame] = None,
    hit_test_df: Optional[pd.DataFrame] = None,
    ref_group_flat_means: Optional[Dict[str, Optional[float]]] = None,
):
    rows = []
    grid_iter = list(
        itertools.product(
            n_estimators_grid,
            learning_rate_grid,
            num_leaves_grid,
            min_child_samples_grid,
            subsample_grid,
            subsample_freq_grid,
            colsample_bytree_grid,
            reg_lambda_grid,
        )
    )
    total = len(grid_iter)
    current = 0

    for (
        n_estimators,
        learning_rate,
        num_leaves,
        min_child_samples,
        subsample,
        subsample_freq,
        colsample_bytree,
        reg_lambda,
    ) in grid_iter:
        current += 1
        print(
            f"[{current}/{total}] training with "
            f"n_estimators={n_estimators}, learning_rate={learning_rate}, "
            f"num_leaves={num_leaves}, min_child_samples={min_child_samples}, "
            f"subsample={subsample}, subsample_freq={subsample_freq}, "
            f"colsample_bytree={colsample_bytree}, reg_lambda={reg_lambda}"
        )
        (
            model,
            pred_df,
            mae,
            wape,
            avg_pearson,
            avg_dtw,
            hit_capture_rate,
            top_k,
            n_valid,
        ) = fit_predict_and_score(
            train_long=train_long,
            test_long=test_long,
            test_df=test_df,
            feature_cols=feature_cols,
            random_state=random_state,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            device_type=device_type,
            max_bin=max_bin,
            gpu_platform_id=gpu_platform_id,
            gpu_device_id=gpu_device_id,
            gpu_use_dp=gpu_use_dp,
            start_week=start_week,
            hit_ratio=hit_ratio,
            forecast_mode=forecast_mode,
            rolling_teacher_forcing=rolling_teacher_forcing,
            hit_test_long=hit_test_long,
            hit_test_df=hit_test_df,
        )
        row = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "mae": mae,
            "wape": wape,
            "avg_pearson": avg_pearson,
            "avg_dtw": avg_dtw,
            "hit_capture_rate": hit_capture_rate,
            "hit_top_k": top_k,
            "hit_n_valid": n_valid,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "samples": len(pred_df),
        }
        if ref_group_flat_means:
            row.update(ref_group_flat_means)
        rows.append(row)
        print(
            f"    mae={mae}, wape={wape}%, avg_p={avg_pearson:.4f}, avg_dtw={avg_dtw:.4f}, "
            f"hit_capture_rate={hit_capture_rate:.4f}, "
            f"best_iteration={rows[-1]['best_iteration']}"
        )

    analysis_df = pd.DataFrame(rows).sort_values(["wape", "mae", "n_estimators", "learning_rate"])
    return analysis_df


def main():
    parser = argparse.ArgumentParser(description="从 embedding 到销量的 LGBM 预测")
    parser.add_argument("--train", default="outputs/train_item_embeddings.parquet")
    parser.add_argument("--test", default="outputs/test_item_embeddings.parquet")
    parser.add_argument("--train_nrows", type=int, default=None)
    parser.add_argument("--test_nrows", type=int, default=None)
    parser.add_argument("--output_csv", default="results/emb_lgbm_predictions.csv")
    parser.add_argument("--analyze_hparams", action="store_true", help="扫描超参数并输出 WAPE 对比表")
    parser.add_argument(
        "--n_estimators_grid",
        default="200,500,1000,2000",
        help="用于分析的 n_estimators 候选值，逗号分隔",
    )
    parser.add_argument(
        "--learning_rate_grid",
        default="0.01,0.03,0.05",
        help="用于分析的 learning_rate 候选值，逗号分隔",
    )
    parser.add_argument(
        "--num_leaves_grid",
        default="63",
        help="用于分析的 num_leaves 候选值，逗号分隔",
    )
    parser.add_argument(
        "--min_child_samples_grid",
        default="30",
        help="用于分析的 min_child_samples 候选值，逗号分隔",
    )
    parser.add_argument(
        "--subsample_grid",
        default="0.8",
        help="用于分析的 subsample 候选值，逗号分隔",
    )
    parser.add_argument(
        "--subsample_freq_grid",
        default="1",
        help="用于分析的 subsample_freq 候选值，逗号分隔",
    )
    parser.add_argument(
        "--colsample_bytree_grid",
        default="0.8",
        help="用于分析的 colsample_bytree 候选值，逗号分隔",
    )
    parser.add_argument(
        "--reg_lambda_grid",
        default="1.0",
        help="用于分析的 reg_lambda 候选值，逗号分隔",
    )
    parser.add_argument(
        "--analysis_output_csv",
        default="results/emb_lgbm_hparam_analysis.csv",
        help="超参数分析结果输出路径",
    )
    parser.add_argument(
        "--start_week",
        type=int,
        default=2,
        help="从第几周开始计算 MAE/WAPE（1-based，默认 2 表示 W2-W12）",
    )
    parser.add_argument(
        "--forecast_mode",
        type=str,
        default="static",
        choices=["static", "rolling_2w1w"],
        help="预测方式: static=原始长表预测；rolling_2w1w=2周->1周滚动递推",
    )
    parser.add_argument(
        "--rolling_teacher_forcing",
        action="store_true",
        help="仅 rolling_2w1w 有效：每一步用真实历史2周做下一周预测",
    )
    parser.add_argument(
        "--hit_ratio",
        type=float,
        default=0.1,
        help="爆款捕捉率的 top 比例，默认 0.1 表示前 10%%",
    )
    parser.add_argument(
        "--eval_ref_stats",
        action="store_true",
        help="计算相似度检索的邻居参考曲线统计（Avg_pairwise_corr/Avg_cv_*/Avg_sim）",
    )
    parser.add_argument(
        "--ref_top_k",
        type=int,
        default=20,
        help="相似度检索用于 ref stats 的 top_k（默认 20）",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--min_child_samples", type=int, default=30)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--subsample_freq", type=int, default=1)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument(
        "--device_type",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="LightGBM 训练设备：cpu/gpu(OpenCL)/cuda，默认 cpu",
    )
    parser.add_argument(
        "--max_bin",
        type=int,
        default=255,
        help="LightGBM max_bin；GPU 上可尝试 63 提速",
    )
    parser.add_argument(
        "--gpu_platform_id",
        type=int,
        default=-1,
        help="OpenCL GPU 平台 ID，仅 --device_type gpu 有效",
    )
    parser.add_argument(
        "--gpu_device_id",
        type=int,
        default=-1,
        help="OpenCL GPU 设备 ID，仅 --device_type gpu 有效",
    )
    parser.add_argument(
        "--gpu_use_dp",
        action="store_true",
        help="OpenCL GPU 使用 double precision，通常更慢，仅需要高精度时开启",
    )
    args = parser.parse_args()

    train_df = prepare_input_from_file(args.train, nrows=args.train_nrows)
    test_df = prepare_input_from_file(args.test, nrows=args.test_nrows)

    print(f"Loaded: train={len(train_df)}, test={len(test_df)}")

    train_long = build_long_from_embeddings(train_df)
    test_long = build_long_from_embeddings(test_df)

    hit_test_long = None
    hit_test_df = None
    if args.test_nrows is not None:
        hit_test_df = prepare_input_from_file(args.test, nrows=None)
        hit_test_long = build_long_from_embeddings(hit_test_df)

    # feature columns: emb_* + sales_* + week_idx + retail
    emb_cols = [c for c in train_long.columns if c.startswith("emb_")]
    meta_cols = ["sales_w0", "sales_w1", "sales_first2_sum", "sales_first2_mean", "sales_first2_delta"]
    feature_cols = emb_cols + meta_cols + ["week_idx", "retail"]

    ref_group_flat_means = None
    ref_group_summary_stats = None
    if args.eval_ref_stats:
        ref_group_flat_means, ref_group_summary_stats = compute_ref_group_flat_means(
            train_df=train_df,
            test_df=test_df,
            top_k=args.ref_top_k,
            start_week=args.start_week,
        )

    if args.analyze_hparams:
        n_estimators_grid = parse_grid_values(args.n_estimators_grid, int)
        learning_rate_grid = parse_grid_values(args.learning_rate_grid, float)
        num_leaves_grid = parse_grid_values(args.num_leaves_grid, int)
        min_child_samples_grid = parse_grid_values(args.min_child_samples_grid, int)
        subsample_grid = parse_grid_values(args.subsample_grid, float)
        subsample_freq_grid = parse_grid_values(args.subsample_freq_grid, int)
        colsample_bytree_grid = parse_grid_values(args.colsample_bytree_grid, float)
        reg_lambda_grid = parse_grid_values(args.reg_lambda_grid, float)

        if not n_estimators_grid:
            raise ValueError("n_estimators_grid 不能为空")
        if not learning_rate_grid:
            raise ValueError("learning_rate_grid 不能为空")
        if not num_leaves_grid:
            raise ValueError("num_leaves_grid 不能为空")
        if not min_child_samples_grid:
            raise ValueError("min_child_samples_grid 不能为空")
        if not subsample_grid:
            raise ValueError("subsample_grid 不能为空")
        if not subsample_freq_grid:
            raise ValueError("subsample_freq_grid 不能为空")
        if not colsample_bytree_grid:
            raise ValueError("colsample_bytree_grid 不能为空")
        if not reg_lambda_grid:
            raise ValueError("reg_lambda_grid 不能为空")

        analysis_df = run_hparam_analysis(
            train_long=train_long,
            test_long=test_long,
            test_df=test_df,
            feature_cols=feature_cols,
            random_state=args.random_state,
            start_week=args.start_week,
            hit_ratio=args.hit_ratio,
            n_estimators_grid=n_estimators_grid,
            learning_rate_grid=learning_rate_grid,
            num_leaves_grid=num_leaves_grid,
            min_child_samples_grid=min_child_samples_grid,
            subsample_grid=subsample_grid,
            subsample_freq_grid=subsample_freq_grid,
            colsample_bytree_grid=colsample_bytree_grid,
            reg_lambda_grid=reg_lambda_grid,
            device_type=args.device_type,
            max_bin=args.max_bin,
            gpu_platform_id=args.gpu_platform_id,
            gpu_device_id=args.gpu_device_id,
            gpu_use_dp=args.gpu_use_dp,
            forecast_mode=args.forecast_mode,
            rolling_teacher_forcing=args.rolling_teacher_forcing,
            hit_test_long=hit_test_long,
            hit_test_df=hit_test_df,
            ref_group_flat_means=ref_group_flat_means,
        )

        out_path = Path(args.analysis_output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(out_path, index=False)

        print("=== LGBM Hparam Analysis ===")
        print(analysis_df.head(10).to_string(index=False))
        print(f"Saved analysis to {out_path}")
        return

    (
        model,
        pred_df,
        mae,
        wape,
        avg_pearson,
        avg_dtw,
        hit_capture_rate,
        top_k,
        n_valid,
    ) = fit_predict_and_score(
        train_long=train_long,
        test_long=test_long,
        test_df=test_df,
        feature_cols=feature_cols,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        device_type=args.device_type,
        max_bin=args.max_bin,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        gpu_use_dp=args.gpu_use_dp,
        start_week=args.start_week,
        hit_ratio=args.hit_ratio,
        forecast_mode=args.forecast_mode,
        rolling_teacher_forcing=args.rolling_teacher_forcing,
        hit_test_long=hit_test_long,
        hit_test_df=hit_test_df,
    )
    print("Model n_features_in_:", model.n_features_in_)
    print("First 20 model features:", model.feature_name_[:20])

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = pred_df.copy()
    export_df["curve"] = export_df["curve"].map(json.dumps)
    export_df["lgbm_pred_curve"] = export_df["lgbm_pred_curve"].map(json.dumps)
    export_df.to_csv(out_path, index=False)

    print("=== LGBM Forecast Metrics ===")
    print(f"Samples: {len(pred_df)}")
    print(f"Global MAE (W{args.start_week}-W12): {mae}")
    print(f"Global WAPE (W{args.start_week}-W12): {wape}%")
    print(f"Avg_Pearson (W{args.start_week}-W12): {avg_pearson:.4f}")
    print(f"Avg_DTW (W{args.start_week}-W12): {avg_dtw:.4f}")
    print(
        f"Hit Capture Rate (top {args.hit_ratio:.0%}, W{args.start_week}-W12): "
        f"{hit_capture_rate:.4f} (top_k={top_k}, n_valid={n_valid})"
    )
    if ref_group_summary_stats is not None:
        print(
            "\n=== Neighbor ref-curve group stats (per test_code×test_retail, W{}+) ===".format(
                args.start_week
            )
        )
        print(ref_group_summary_stats.to_string())
        print(f"flat_means: {ref_group_flat_means}")
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
