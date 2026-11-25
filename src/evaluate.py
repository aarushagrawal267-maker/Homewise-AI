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


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def bootstrap_metric_ci(y_true, y_pred, n_boot: int = 2000, level: float = 0.95,
                        seed: int = SEED) -> dict[str, dict[str, float]]:
    """Percentile-bootstrap confidence intervals for the headline metrics.

    Rows of the evaluation set are resampled with replacement, so the intervals
    describe how much a metric would move on another sample of the same size from
    the same population. They do not cover model-training randomness.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        m = regression_metrics(y_true[i], y_pred[i])
        m["within_10pct"] = float(np.mean(np.abs(y_pred[i] - y_true[i]) / y_true[i] <= 0.10) * 100)
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
    a = (1 - level) / 2 * 100
    return {k: {"lower": float(np.percentile(v, a)), "upper": float(np.percentile(v, 100 - a))}
            for k, v in draws.items()}


def paired_bootstrap_rmsle_diff(y_true, pred_a, pred_b, n_boot: int = 2000, level: float = 0.95,
                                seed: int = SEED) -> dict[str, float]:
    """RMSLE(a) − RMSLE(b) on the same rows, with a paired percentile-bootstrap interval.

    A negative difference favours model a. An interval that contains 0 means the
    evaluation set cannot tell the two models apart at this confidence level.
    """
    y_true, pred_a, pred_b = (np.asarray(v, float) for v in (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        diffs.append(_rmsle(y_true[i], pred_a[i]) - _rmsle(y_true[i], pred_b[i]))
    a = (1 - level) / 2 * 100
    return {"diff": _rmsle(y_true, pred_a) - _rmsle(y_true, pred_b),
            "lower": float(np.percentile(diffs, a)), "upper": float(np.percentile(diffs, 100 - a))}


def naive_baselines(X_train: pd.DataFrame, y_train: pd.Series, X_eval: pd.DataFrame) -> dict[str, np.ndarray]:
    """Predictions from simple reference models, fitted on the training split only.

    * the training median price for every home;
    * the training median price of the home's neighbourhood (overall median for unseen ones);
    * a two-feature log-linear model: log1p(price) ~ log(GrLivArea) + OverallQual.
    """
    from sklearn.linear_model import LinearRegression

    median = float(y_train.median())
    by_nbhd = y_train.groupby(X_train["Neighborhood"]).median()

    def design(X):
        return np.column_stack([np.log(X["GrLivArea"].astype(float).clip(lower=1)), X["OverallQual"].astype(float)])

    lr = LinearRegression().fit(design(X_train), np.log1p(y_train))
    return {
        "Training median price": np.full(len(X_eval), median),
        "Neighbourhood median price": X_eval["Neighborhood"].map(by_nbhd).fillna(median).to_numpy(float),
        "Log-linear on living area + quality": np.expm1(lr.predict(design(X_eval))),
    }
