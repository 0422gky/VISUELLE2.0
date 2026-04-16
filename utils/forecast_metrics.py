from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error

from utils.curve_eval_metrics import avg_pearson_avg_dtw


def cal_error_metrics(gt, forecasts):
    # Absolute errors
    mae = mean_absolute_error(gt, forecasts)
    wape = 100 * np.sum(np.sum(np.abs(gt - forecasts), axis=-1)) / np.sum(gt)
    return round(mae, 3), round(wape, 3)


def print_error_metrics(
    y_test,
    y_hat,
    rescaled_y_test,
    rescaled_y_hat,
    *,
    wsl: int = 0,
    tw: int | None = None,
):
    mae, wape = cal_error_metrics(y_test, y_hat)
    rescaled_mae, rescaled_wape = cal_error_metrics(rescaled_y_test, rescaled_y_hat)
    print(mae, wape, rescaled_mae, rescaled_wape)
    ap, adtw = avg_pearson_avg_dtw(rescaled_y_test, rescaled_y_hat, wsl=wsl, tw=tw)
    print(
        f"Avg_Pearson {ap:.4f}  Avg_DTW {adtw:.4f}  "
        f"(window [{wsl}:{tw if tw is not None else rescaled_y_test.shape[1]}), rescaled sales)"
    )
