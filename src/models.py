"""Model registry and pipeline construction.

Every model is wrapped as::

    TransformedTargetRegressor(
        regressor=Pipeline([clean, features, preprocess, model]),
        func=log1p, inverse_func=expm1)

so ``predict`` always takes raw Kaggle-format rows and returns prices in US
dollars, and exactly the same transformations run during training and serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .config import SEED
from .feature_engineering import FeatureEngineer
from .preprocessing import DomainCleaner, build_preprocessor

try:  # optional dependency
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - depends on environment
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:  # optional dependency
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False


@dataclass
class ModelSpec:
    key: str
    label: str
    kind: str  # "linear" or "tree" preprocessing
    factory: Callable[..., Any]
    defaults: dict = field(default_factory=dict)
    purpose: str = ""
    available: bool = True
    # Parameters exposed in the AI Experiment Lab: name -> (type, min, max, default)
    lab_params: dict = field(default_factory=dict)


def _xgb(**kw):
    return XGBRegressor(**kw)


def _lgbm(**kw):
    return LGBMRegressor(**kw)


MODEL_SPECS: dict[str, ModelSpec] = {
    "linear_regression": ModelSpec(
        "linear_regression", "Linear Regression", "linear", LinearRegression, {},
        "Unregularised baseline: how far a straight linear relationship gets us.",
        lab_params={}),
    "ridge": ModelSpec(
        "ridge", "Ridge Regression", "linear", Ridge, {"alpha": 10.0},
        "L2 regularisation shrinks coefficients to reduce overfitting.",
        lab_params={"alpha": ("float", 0.01, 200.0, 10.0)}),
    "lasso": ModelSpec(
        "lasso", "Lasso Regression", "linear", Lasso, {"alpha": 0.0005, "max_iter": 50000},
        "L1 regularisation drives weak coefficients to exactly zero (feature selection).",
        lab_params={"alpha": ("float", 0.00005, 0.05, 0.0005)}),
    "decision_tree": ModelSpec(
        "decision_tree", "Decision Tree", "tree", DecisionTreeRegressor,
        {"max_depth": 8, "min_samples_leaf": 5, "random_state": SEED},
        "A single tree of if/then splits: non-linear and readable, but high variance.",
        lab_params={"max_depth": ("int", 2, 30, 8), "min_samples_leaf": ("int", 1, 50, 5)}),
    "random_forest": ModelSpec(
        "random_forest", "Random Forest", "tree", RandomForestRegressor,
        {"n_estimators": 300, "max_features": 0.33, "min_samples_leaf": 1,
         "random_state": SEED, "n_jobs": -1},
        "Averages many decorrelated trees to capture interactions with lower variance.",
        lab_params={"n_estimators": ("int", 50, 800, 300), "max_depth": ("int", 3, 40, 20),
                    "min_samples_leaf": ("int", 1, 20, 1), "max_features": ("float", 0.1, 1.0, 0.33)}),
    "gradient_boosting": ModelSpec(
        "gradient_boosting", "Gradient Boosting", "tree", GradientBoostingRegressor,
        {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8,
         "max_features": "sqrt", "min_samples_leaf": 5, "loss": "huber", "random_state": SEED},
        "Sequentially fits shallow trees to the residuals of the previous ones.",
        lab_params={"n_estimators": ("int", 50, 3000, 600), "learning_rate": ("float", 0.005, 0.3, 0.05),
                    "max_depth": ("int", 2, 8, 3), "min_samples_leaf": ("int", 1, 30, 5),
                    "subsample": ("float", 0.5, 1.0, 0.8)}),
    "xgboost": ModelSpec(
        "xgboost", "XGBoost", "tree", _xgb,
        {"n_estimators": 1000, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.8,
         "colsample_bytree": 0.5, "min_child_weight": 2, "reg_lambda": 1.0, "reg_alpha": 0.0,
         "random_state": SEED, "n_jobs": 1, "tree_method": "hist", "verbosity": 0},
        "Regularised gradient boosting with second-order optimisation.",
        available=XGBOOST_AVAILABLE,
        lab_params={"n_estimators": ("int", 100, 4000, 1000), "learning_rate": ("float", 0.005, 0.3, 0.03),
                    "max_depth": ("int", 2, 10, 3), "subsample": ("float", 0.5, 1.0, 0.8),
                    "colsample_bytree": ("float", 0.2, 1.0, 0.5), "min_child_weight": ("int", 1, 20, 2)}),
    "lightgbm": ModelSpec(
        "lightgbm", "LightGBM", "tree", _lgbm,
        {"n_estimators": 1500, "learning_rate": 0.02, "num_leaves": 8, "min_child_samples": 10,
         "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.4, "reg_lambda": 1.0,
         "random_state": SEED, "n_jobs": 1, "verbose": -1},
        "Histogram-based leaf-wise boosting: fast and strong on tabular data.",
        available=LIGHTGBM_AVAILABLE,
        lab_params={"n_estimators": ("int", 100, 5000, 1500), "learning_rate": ("float", 0.005, 0.3, 0.02),
                    "num_leaves": ("int", 4, 64, 8), "min_child_samples": ("int", 3, 60, 10),
                    "colsample_bytree": ("float", 0.2, 1.0, 0.4), "subsample": ("float", 0.5, 1.0, 0.8)}),
}

ENSEMBLE_LABELS = {
    "weighted_ensemble": "Weighted Ensemble",
    "stacking_ensemble": "Stacking Ensemble",
}


def available_models() -> list[str]:
    return [k for k, s in MODEL_SPECS.items() if s.available]


def model_label(key: str) -> str:
    if key in MODEL_SPECS:
        return MODEL_SPECS[key].label
    return ENSEMBLE_LABELS.get(key, key)


def make_estimator(key: str, params: dict | None = None):
    spec = MODEL_SPECS[key]
    if not spec.available:
        raise RuntimeError(f"{spec.label} is not installed in this environment.")
    kw = {**spec.defaults, **(params or {})}
    return spec.factory(**kw)


def make_inner_pipeline(key: str, params: dict | None = None, feature_engineering: bool = True,
                        fe_groups: tuple[str, ...] | None = None) -> Pipeline:
    """Raw rows → log-price prediction (no target transform)."""
    spec = MODEL_SPECS[key]
    return Pipeline([
        ("clean", DomainCleaner()),
        ("features", FeatureEngineer(enabled=feature_engineering, groups=fe_groups)),
        ("prep", build_preprocessor(spec.kind)),
        ("model", make_estimator(key, params)),
    ])


def wrap_log_target(regressor) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p,
                                      inverse_func=np.expm1, check_inverse=False)


def build_model(key: str, params: dict | None = None, feature_engineering: bool = True,
                fe_groups: tuple[str, ...] | None = None) -> TransformedTargetRegressor:
    """Full model: raw rows in, dollar prices out."""
    return wrap_log_target(make_inner_pipeline(key, params, feature_engineering, fe_groups))


def build_weighted_ensemble(members: dict[str, dict], weights: list[float]) -> TransformedTargetRegressor:
    """Weighted average of member predictions in log-price space."""
    estimators = [(k, make_inner_pipeline(k, p)) for k, p in members.items()]
    return wrap_log_target(VotingRegressor(estimators, weights=weights))


def build_stacking_ensemble(members: dict[str, dict], cv: int = 5) -> TransformedTargetRegressor:
    """Stacked generalisation with a ridge meta-learner on out-of-fold predictions."""
    estimators = [(k, make_inner_pipeline(k, p)) for k, p in members.items()]
    meta = RidgeCV(alphas=np.logspace(-3, 2, 20))
    return wrap_log_target(StackingRegressor(estimators, final_estimator=meta, cv=cv, n_jobs=1))
