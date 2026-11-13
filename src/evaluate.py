"""Regression metrics and cross-validation utilities."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

from .config import CONFIG, SEED


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Metrics in original dollar units (predictions must already be inverse-transformed)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions contain non-finite values.")
    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedianAE": float(median_absolute_error(y_true, y_pred)),
    }
    # RMSLE and MAPE require strictly positive targets and predictions.
    if np.all(y_true > 0) and np.all(y_pred > 0):
        out["RMSLE"] = float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))
        out["MAPE"] = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return out


def residual_stats(y_true, y_pred) -> dict[str, float]:
    res = np.asarray(y_true, float) - np.asarray(y_pred, float)
    pct = res / np.asarray(y_true, float) * 100
    return {
        "mean_residual": float(res.mean()),
        "std_residual": float(res.std(ddof=1)),
        "min_residual": float(res.min()),
        "max_residual": float(res.max()),
        "share_within_10pct": float(np.mean(np.abs(pct) <= 10) * 100),
        "share_within_20pct": float(np.mean(np.abs(pct) <= 20) * 100),
    }


def cv_splitter(n_splits: int | None = None, seed: int = SEED) -> KFold:
    return KFold(n_splits=n_splits or CONFIG["split"]["cv_folds"], shuffle=True, random_state=seed)


def cross_validate_oof(model, X: pd.DataFrame, y: pd.Series, n_splits: int | None = None,
                       seed: int = SEED, progress=None) -> dict:
    """K-fold CV returning per-fold RMSLE and out-of-fold predictions (dollars).

    Every fold clones and refits the full pipeline, so imputers/encoders/scalers
    are fitted on that fold's training rows only.
    """
    kf = cv_splitter(n_splits, seed)
    oof = np.zeros(len(y))
    fold_rmsle, fold_times = [], []
    for i, (tr, va) in enumerate(kf.split(X)):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X.iloc[tr], y.iloc[tr])
        fold_times.append(time.perf_counter() - t0)
        pred = m.predict(X.iloc[va])
        oof[va] = pred
        fold_rmsle.append(float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(y.iloc[va])) ** 2))))
        if progress:
            progress(i + 1, kf.get_n_splits())
    return {
        "cv_rmsle_mean": float(np.mean(fold_rmsle)),
        "cv_rmsle_std": float(np.std(fold_rmsle)),
        "cv_fold_rmsle": fold_rmsle,
        "cv_fit_time_mean": float(np.mean(fold_times)),
        "oof_pred": oof,
        "oof_metrics": regression_metrics(y, oof),
    }


def timed_fit_predict(model, X_train, y_train, X_eval) -> tuple[object, np.ndarray, float, float]:
    """Fit, then predict, returning (model, predictions, train_seconds, inference_ms_per_row)."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = model.predict(X_eval)
    infer_ms = (time.perf_counter() - t1) / max(len(X_eval), 1) * 1000
    return model, pred, train_s, infer_ms
