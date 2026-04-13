"""
与 exploration 中 test_codes_df 一致：按「每个被预测 SKU × 门店」分组，
对组内多条「邻居参考曲线」算 Avg_pairwise_corr、Avg_cv_total、Avg_cv_week、Avg_neigh、Avg_sim。

正确语义：应用在 **相似度检索后的明细表**（每 test 多行邻居，含曲线 + 相似度），
例如 similarity_wape_pipeline 的 final_ref_df；而不是 train 上「一行一 embedding」的导出表。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def _pearson_pair_1d(a: np.ndarray, b: np.ndarray, min_std: float = 1e-6) -> float | None:
    if np.std(a) <= min_std or np.std(b) <= min_std:
        return None
    return float(np.corrcoef(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))[0, 1])


def ref_group_core(
    df: pd.DataFrame,
    wsl: int,
    tw: int | None,
    *,
    epsilon: float = 1e-6,
    sim_col: str | None = "final_similarity",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    df 必须含列: external_code, retail, ref_curve（每行一条邻居曲线）。
    sim_col 若存在且非 None，则用于组内 Avg_sim；否则 Avg_sim 为 nan。
    """
    if sim_col is not None and sim_col not in df.columns:
        sim_col = None

    results: list[dict[str, Any]] = []
    for _idx, group in df.groupby(["external_code", "retail"], sort=False):
        g_ref = group["ref_curve"]
        if not g_ref.notna().any():
            results.append(
                {
                    "external_code": _idx[0],
                    "retail": _idx[1],
                    "Avg_pairwise_corr": np.nan,
                    "Avg_cv_total": np.nan,
                    "Avg_cv_week": np.nan,
                    "Avg_neigh": np.nan,
                    "Avg_sim": np.nan,
                }
            )
            continue

        curves_full: list[np.ndarray] = []
        for c in g_ref:
            if c is None:
                continue
            if isinstance(c, (float, np.floating)) and np.isnan(float(c)):
                continue
            if isinstance(c, str) and not c.strip():
                continue
            arr = np.asarray(c, dtype=np.float64)
            if arr.size == 0 or not np.isfinite(arr).all():
                continue
            curves_full.append(arr)
        lens = {int(c.size) for c in curves_full}
        if len(lens) > 1:
            raise ValueError(
                f"同一 (external_code, retail)={_idx} 下参考曲线长度不一致: {sorted(lens)}"
            )
        if not curves_full:
            results.append(
                {
                    "external_code": _idx[0],
                    "retail": _idx[1],
                    "Avg_pairwise_corr": np.nan,
                    "Avg_cv_total": np.nan,
                    "Avg_cv_week": np.nan,
                    "Avg_neigh": np.nan,
                    "Avg_sim": np.nan,
                }
            )
            continue

        L = int(curves_full[0].size)
        end = L if tw is None else int(tw)
        if wsl < 0 or end > L or wsl >= end:
            raise ValueError(f"ref_group: 无效窗口 wsl={wsl}, tw={tw}, L={L}")

        curves = [c[wsl:end] for c in curves_full]
        n_curves = len(curves)

        if sim_col is not None:
            group_avg_sim = float(np.nanmean(group[sim_col].to_numpy(dtype=np.float64)))
        else:
            group_avg_sim = float("nan")

        if n_curves == 1:
            results.append(
                {
                    "external_code": _idx[0],
                    "retail": _idx[1],
                    "Avg_pairwise_corr": np.nan,
                    "Avg_cv_total": np.nan,
                    "Avg_cv_week": np.nan,
                    "Avg_neigh": float(n_curves),
                    "Avg_sim": group_avg_sim,
                }
            )
            continue

        ref_mat = np.stack(curves, axis=0)
        mu_t = np.mean(ref_mat, axis=0)
        sigma_t = np.std(ref_mat, axis=0)
        cv_total = float(np.sum(sigma_t) / (np.sum(mu_t) + epsilon))
        cv_week = float(np.mean(np.where(mu_t > 0, sigma_t / (mu_t + epsilon), 0.0)))

        pairwise_corr_lst: list[float] = []
        for i in range(n_curves):
            for j in range(i + 1, n_curves):
                pr = _pearson_pair_1d(curves[i], curves[j])
                if pr is not None:
                    pairwise_corr_lst.append(pr)
        avg_corr = float(np.mean(pairwise_corr_lst)) if pairwise_corr_lst else float("nan")

        results.append(
            {
                "external_code": _idx[0],
                "retail": _idx[1],
                "Avg_pairwise_corr": avg_corr,
                "Avg_cv_total": cv_total,
                "Avg_cv_week": cv_week,
                "Avg_neigh": float(n_curves),
                "Avg_sim": group_avg_sim,
            }
        )

    results_df = pd.DataFrame(results)
    stat_cols = ["Avg_pairwise_corr", "Avg_cv_total", "Avg_cv_week", "Avg_neigh", "Avg_sim"]
    summary_stats = results_df[stat_cols].agg(["mean", "median", "std"]).T
    return results_df, summary_stats


def ref_group_stats_from_final_ref_df(
    final_ref_df: pd.DataFrame,
    wsl: int,
    tw: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    similarity_wape_pipeline.build_similarity_refs 的输出：
    列含 test_code, test_retail, curve(邻居曲线), sim_score。
    按每个 (test_code, test_retail) 聚类，等价于笔记本里对每个新品-门店的多条 ref。
    """
    need = {"test_code", "test_retail", "curve", "sim_score"}
    if final_ref_df.empty or not need.issubset(final_ref_df.columns):
        return None
    work = final_ref_df.copy()
    work["external_code"] = work["test_code"]
    work["retail"] = work["test_retail"]
    work["ref_curve"] = work["curve"]
    work["final_similarity"] = work["sim_score"]
    slim = work[["external_code", "retail", "ref_curve", "final_similarity"]]
    return ref_group_core(slim, wsl, tw, sim_col="final_similarity")


def summary_stats_to_jsonable(summary: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for row in summary.index:
        out[str(row)] = {}
        for col in summary.columns:
            v = summary.loc[row, col]
            vf = np.asarray(v, dtype=np.float64).item()
            if not np.isfinite(vf):
                out[str(row)][str(col)] = None
            else:
                out[str(row)][str(col)] = float(vf)
    return out


def ref_group_flat_means_from_summary(
    ref_group_stats: dict[str, dict[str, float | None]] | None,
) -> dict[str, float | None]:
    mapping = (
        ("Avg_pairwise_corr", "avg_pairwise_corr"),
        ("Avg_cv_total", "avg_cv_total"),
        ("Avg_cv_week", "avg_cv_week"),
        ("Avg_sim", "avg_sim"),
        ("Avg_neigh", "avg_neigh"),
    )
    flat: dict[str, float | None] = {alias: None for _, alias in mapping}
    if not ref_group_stats:
        return flat
    for src, alias in mapping:
        block = ref_group_stats.get(src)
        if isinstance(block, dict):
            flat[alias] = block.get("mean")
    return flat


def save_ref_group_artifacts(
    out_prefix: str,
    results_df: pd.DataFrame,
    summary_stats: pd.DataFrame,
) -> None:
    """写入 per-group CSV + summary JSON。"""
    csv_path = f"{out_prefix}_ref_group_per_sku.csv"
    json_path = f"{out_prefix}_ref_group_summary.json"
    results_df.to_csv(csv_path, index=False)
    summ_json = summary_stats_to_jsonable(summary_stats)
    payload = {
        "summary_stats": summ_json,
        "flat_means": ref_group_flat_means_from_summary(summ_json),
        "n_groups": int(len(results_df)),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved ref_group: {csv_path}, {json_path}")


def train_export_ref_group_skip_reason(df: pd.DataFrame) -> str | None:
    """对「一行一 SKU」的 embedding 导出表做组统计时，列不满足则返回原因。"""
    has_std = {"external_code", "retail"}.issubset(df.columns)
    has_train = {"train_code", "train_retail"}.issubset(df.columns)
    has_code = {"code", "retail"}.issubset(df.columns)
    if not has_std and not has_train and not has_code:
        cols = list(df.columns)[:30]
        return (
            "缺分组列。需要 (external_code, retail) 或 (train_code, train_retail) 或 (code, retail)。"
            f"当前列（前30）: {cols}"
        )
    ref_ok = (
        "ref_curve" in df.columns
        or "curve" in df.columns
        or len([c for c in df.columns if str(c).endswith("_curve")]) == 1
    )
    if not ref_ok:
        return "缺曲线列。需要 ref_curve、curve，或恰好一列 *_curve。"
    return None


def ref_group_stats_from_train_export_table(
    df: pd.DataFrame,
    wsl: int,
    tw: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    在「训练用导出表」上按 SKU 分组（通常每组仅 1 条曲线）。
    与笔记本语义不完全一致；优先使用 ref_group_stats_from_final_ref_df。
    """
    if train_export_ref_group_skip_reason(df) is not None:
        return None

    df_work = df
    if not {"external_code", "retail"}.issubset(df.columns):
        df_work = df.copy()
        if {"train_code", "train_retail"}.issubset(df.columns):
            df_work["external_code"] = df_work["train_code"]
            df_work["retail"] = df_work["train_retail"]
        elif {"code", "retail"}.issubset(df.columns):
            df_work["external_code"] = df_work["code"]
        else:
            return None

    if "ref_curve" in df_work.columns:
        ref_col = "ref_curve"
    elif "curve" in df_work.columns:
        ref_col = "curve"
    else:
        curve_like = [c for c in df_work.columns if str(c).endswith("_curve")]
        if len(curve_like) != 1:
            return None
        ref_col = curve_like[0]

    sim_col: str | None
    if "final_similarity" in df_work.columns:
        sim_col = "final_similarity"
    elif "final_sim" in df_work.columns:
        sim_col = "final_sim"
    else:
        sim_col = None

    slim = pd.DataFrame(
        {
            "external_code": df_work["external_code"].values,
            "retail": df_work["retail"].values,
            "ref_curve": df_work[ref_col].values,
        }
    )
    if sim_col is not None:
        slim["final_similarity"] = df_work[sim_col].values
        return ref_group_core(slim, wsl, tw, sim_col="final_similarity")
    return ref_group_core(slim, wsl, tw, sim_col=None)
