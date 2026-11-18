"""Regression metrics aligned with NREL/IEC photovoltaic forecasting audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    """Blind-test evaluation metrics in physical power (kW) and normalized capacity factor."""

    r2: float
    rmse_kw: float
    mae_kw: float
    wape_pct: float
    nrmse_pu: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def compute_regression_metrics(
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
    y_true_cf: np.ndarray | None = None,
    y_pred_cf: np.ndarray | None = None,
) -> RegressionMetrics:
    """Compute R2, RMSE (kW), MAE (kW), WAPE (%), and optional CF nRMSE (p.u.)."""
    denom = float(np.sum(np.abs(y_true_kw)))
    wape_pct = (float(np.sum(np.abs(y_true_kw - y_pred_kw))) / denom * 100.0) if denom > 0 else 0.0

    nrmse_pu: float | None = None
    if y_true_cf is not None and y_pred_cf is not None:
        nrmse_pu = float(np.sqrt(mean_squared_error(y_true_cf, y_pred_cf)))

    return RegressionMetrics(
        r2=float(r2_score(y_true_kw, y_pred_kw)),
        rmse_kw=float(np.sqrt(mean_squared_error(y_true_kw, y_pred_kw))),
        mae_kw=float(mean_absolute_error(y_true_kw, y_pred_kw)),
        wape_pct=wape_pct,
        nrmse_pu=nrmse_pu,
    )
