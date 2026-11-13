"""Prediction uncertainty via split-conformal intervals on the log-price scale.

Method
------
1. Out-of-fold (OOF) predictions are produced by K-fold cross-validation on the
   training split, so every training property gets a prediction from a model
   that did not see it.
2. Absolute residuals are computed on the log scale:
   ``r_i = |log1p(y_i) − log1p(ŷ_i)|``. Working in logs makes the interval
   width proportional to price (a $500k home gets a wider dollar range).
3. For coverage level ``1 − α`` the conformal quantile is
   ``q = r_(⌈(n+1)(1−α)⌉)`` (the finite-sample-corrected empirical quantile).
4. Interval: ``[expm1(log1p(ŷ) − q), expm1(log1p(ŷ) + q)]``.

Guarantee and caveats: under exchangeability, split-conformal intervals have
marginal coverage ≥ 1 − α *on average over properties like those in the data*.
Here the residuals come from CV models rather than the final model (a CV+-style
approximation), and coverage is not guaranteed for any individual property or
for homes unlike those in Ames 2006–2010. Empirical coverage on the held-out
validation split is reported so the claim can be checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionInterval:
    estimate: float
    lower: float
    upper: float
    level: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def relative_width(self) -> float:
        return self.width / self.estimate if self.estimate else float("nan")

    def to_dict(self) -> dict:
        return {"estimate": self.estimate, "lower": self.lower, "upper": self.upper,
                "level": self.level, "width": self.width, "relative_width": self.relative_width}


def conformal_quantiles(y_true, oof_pred, levels=(0.8, 0.9, 0.95)) -> dict[float, float]:
    y_true = np.asarray(y_true, float)
    oof_pred = np.asarray(oof_pred, float)
    if np.any(oof_pred <= -1) or np.any(y_true <= -1):
        raise ValueError("Values must be > -1 for log1p residuals.")
    resid = np.sort(np.abs(np.log1p(y_true) - np.log1p(oof_pred)))
    n = len(resid)
    out = {}
    for lvl in levels:
        k = int(np.ceil((n + 1) * lvl))
        out[float(lvl)] = float(resid[min(k, n) - 1])
    return out


def interval_from_quantile(estimate: float, q: float, level: float) -> PredictionInterval:
    if not np.isfinite(estimate) or estimate <= 0:
        raise ValueError("Estimate must be a positive finite price.")
    lo = float(np.expm1(np.log1p(estimate) - q))
    hi = float(np.expm1(np.log1p(estimate) + q))
    return PredictionInterval(float(estimate), lo, hi, float(level))


def intervals(estimates, q: float):
    est = np.asarray(estimates, float)
    return np.expm1(np.log1p(est) - q), np.expm1(np.log1p(est) + q)


def empirical_coverage(y_true, estimates, quantiles: dict[float, float]) -> dict[float, dict]:
    """Share of true prices falling inside the interval, per level."""
    y_true = np.asarray(y_true, float)
    out = {}
    for lvl, q in quantiles.items():
        lo, hi = intervals(estimates, q)
        inside = (y_true >= lo) & (y_true <= hi)
        out[float(lvl)] = {
            "target": float(lvl),
            "empirical": float(inside.mean()),
            "mean_width": float(np.mean(hi - lo)),
            "median_relative_width": float(np.median((hi - lo) / np.asarray(estimates, float))),
            "log_half_width": float(q),
        }
    return out
