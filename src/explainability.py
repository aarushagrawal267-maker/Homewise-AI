"""Explainable AI: SHAP contributions, permutation importance, partial dependence.

SHAP values are computed on the model's native output, the **log price**, where
they are exactly additive::

    log1p(price) = base_value + Σ contribution_i

One-hot columns are summed back to their source characteristic (e.g. all
``Neighborhood_*`` columns → ``Neighborhood``), which preserves additivity.

Ensembles are handled exactly: a weighted average (or a linear stacking
meta-model) of member models has SHAP values equal to the same weighted
combination of the members' SHAP values.

Contributions describe how *the model* arrived at its output. They are not
causal effects of changing a property in the real market.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from .preprocessing import transformed_feature_sources

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False

LINEAR_TYPES = (LinearRegression, Ridge, Lasso)


@dataclass
class Explanation:
    base_log: float                 # expected log price over the background data
    contributions: pd.DataFrame     # rows × source features, log-price units
    feature_values: pd.DataFrame    # engineered feature values shown to users
    pred_log: np.ndarray            # model log-price output per row

    @property
    def base_price(self) -> float:
        return float(np.expm1(self.base_log))

    def additivity_error(self) -> float:
        recon = self.base_log + self.contributions.sum(axis=1).to_numpy()
        return float(np.max(np.abs(recon - self.pred_log)))

    def row(self, i: int = 0) -> pd.DataFrame:
        """Tidy explanation for one row with approximate dollar allocation."""
        phi = self.contributions.iloc[i]
        pred_price = float(np.expm1(self.pred_log[i]))
        total = float(phi.sum())
        # Allocate the dollar gap (prediction − base) proportionally to log contributions.
        factor = (pred_price - self.base_price) / total if abs(total) > 1e-9 else self.base_price
        df = pd.DataFrame({
            "feature": phi.index,
            "log_contribution": phi.to_numpy(),
            "pct_effect": (np.exp(phi.to_numpy()) - 1) * 100,
            "dollar_contribution": phi.to_numpy() * factor,
        })
        vals = self.feature_values.iloc[i]
        df["value"] = [vals.get(f, np.nan) for f in df["feature"]]
        df["abs"] = df["log_contribution"].abs()
        return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


class _PipelineExplainer:
    """SHAP for one Pipeline([clean, features, prep, model]) on the log scale."""

    def __init__(self, pipe: Pipeline, X_background: pd.DataFrame):
        self.pipe = pipe
        self.pre = Pipeline(pipe.steps[:-1])
        self.model = pipe.steps[-1][1]
        _, self.sources = transformed_feature_sources(pipe.named_steps["prep"])
        Zb = np.asarray(self.pre.transform(X_background), dtype=float)
        if isinstance(self.model, LINEAR_TYPES):
            self.kind = "linear"
            self.coef = np.ravel(self.model.coef_)
            self.mean = Zb.mean(axis=0)
            self.base = float(self.model.intercept_ + self.coef @ self.mean)
        else:
            if not SHAP_AVAILABLE:
                raise RuntimeError("SHAP is not installed.")
            try:
                self.kind = "tree"
                self.explainer = shap.TreeExplainer(self.model)
                base = self.explainer.expected_value
            except Exception:
                self.kind = "permutation"
                bg = Zb[np.random.default_rng(0).choice(len(Zb), min(100, len(Zb)), replace=False)]
                self.explainer = shap.PermutationExplainer(self.model.predict, bg)
                base = float(self.model.predict(bg).mean())
            self.base = float(np.ravel(base)[0])

    def shap_by_source(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        Z = np.asarray(self.pre.transform(X), dtype=float)
        if self.kind == "linear":
            phi = (Z - self.mean) * self.coef
        elif self.kind == "tree":
            phi = np.asarray(self.explainer.shap_values(Z, check_additivity=False))
        else:
            phi = self.explainer(Z).values
        df = pd.DataFrame(phi, columns=self.sources, index=X.index)
        return df.T.groupby(level=0, sort=False).sum().T, self.base


class HomeWiseExplainer:
    """Explains a fitted HomeWise model (single pipeline, voting or stacking)."""

    def __init__(self, model, X_background: pd.DataFrame):
        self.model = model
        reg = model.regressor_
        if isinstance(reg, Pipeline):
            self.parts = [(1.0, _PipelineExplainer(reg, X_background))]
            self.intercept = 0.0
        elif isinstance(reg, VotingRegressor):
            w = np.asarray(reg.weights if reg.weights is not None else np.ones(len(reg.estimators_)), float)
            w = w / w.sum()
            self.parts = [(wi, _PipelineExplainer(e, X_background)) for wi, e in zip(w, reg.estimators_)]
            self.intercept = 0.0
        elif isinstance(reg, StackingRegressor):
            meta = reg.final_estimator_
            self.parts = [(float(a), _PipelineExplainer(e, X_background))
                          for a, e in zip(np.ravel(meta.coef_), reg.estimators_)]
            self.intercept = float(meta.intercept_)
        else:
            raise TypeError(f"Unsupported model type for explanation: {type(reg).__name__}")
        self.feature_pipe = Pipeline(self.parts[0][1].pipe.steps[:2])

    def explain(self, X: pd.DataFrame) -> Explanation:
        total, base = None, self.intercept
        for w, part in self.parts:
            phi, b = part.shap_by_source(X)
            base += w * b
            total = phi * w if total is None else total.add(phi * w, fill_value=0.0)
        pred_log = np.log1p(self.model.predict(X))
        feats = self.feature_pipe.transform(X)
        return Explanation(float(base), total[sorted(total.columns, key=lambda c: -total[c].abs().mean())],
                           feats, pred_log)


def global_shap_importance(expl: Explanation) -> pd.DataFrame:
    imp = expl.contributions.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.to_numpy()})


def permutation_importance_raw(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5,
                               seed: int = 42) -> pd.DataFrame:
    """Permutation importance on the raw input columns, scored by RMSLE.

    Measures how much validation error increases when a column is shuffled,
    breaking its relationship with price. Correlated columns share credit, so
    individual importances can be understated.
    """
    res = permutation_importance(model, X, y, scoring="neg_root_mean_squared_log_error",
                                 n_repeats=n_repeats, random_state=seed, n_jobs=1)
    df = pd.DataFrame({"feature": X.columns, "importance_mean": res.importances_mean,
                       "importance_std": res.importances_std})
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def partial_dependence_1d(predict_fn, X: pd.DataFrame, feature: str, grid, setter=None,
                          ice_rows: int = 30) -> dict:
    """Average prediction (and ICE curves) as ``feature`` is set to each grid value.

    ``setter(df, feature, value)`` lets callers keep related fields consistent
    (e.g. floor areas when living area changes). Other features keep their
    observed values, so the curve shows the model's response, not a causal effect.
    """
    setter = setter or (lambda d, f, v: d.assign(**{f: v}))
    avg, ice = [], []
    idx = np.random.default_rng(0).choice(len(X), min(ice_rows, len(X)), replace=False)
    for v in grid:
        p = predict_fn(setter(X.copy(), feature, v))
        avg.append(float(np.mean(p)))
        ice.append(np.asarray(p)[idx].tolist())
    return {"feature": feature, "grid": [float(g) for g in grid], "average": avg,
            "ice": np.asarray(ice).T.tolist()}


def partial_dependence_2d(predict_fn, X: pd.DataFrame, f1: str, grid1, f2: str, grid2,
                          setter=None) -> dict:
    """Average model prediction over a 2-D grid (used for the 3-D price surface)."""
    setter = setter or (lambda d, f, v: d.assign(**{f: v}))
    Z = np.zeros((len(grid2), len(grid1)))
    for j, b in enumerate(grid2):
        base = setter(X.copy(), f2, b)
        frames = [setter(base.copy(), f1, a) for a in grid1]
        preds = np.asarray(predict_fn(pd.concat(frames, ignore_index=True))).reshape(len(grid1), len(X))
        Z[j, :] = preds.mean(axis=1)
    return {"f1": f1, "f2": f2, "grid1": [float(g) for g in grid1], "grid2": [float(g) for g in grid2],
            "z": Z.tolist()}
