"""Controlled hyperparameter optimisation.

All searches run cross-validation *inside the training split only*; the
validation split is never seen. Budgets are sized for an ordinary laptop CPU.

* Ridge / Lasso / Decision Tree → GridSearchCV (small grids)
* Random Forest                 → RandomizedSearchCV
* Gradient Boosting / XGBoost / LightGBM → Optuna TPE (falls back to
  RandomizedSearchCV if Optuna is not installed)
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from .config import CONFIG, SEED
from .evaluate import cv_splitter
from .models import build_model

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover
    optuna = None
    OPTUNA_AVAILABLE = False

SCORING = "neg_root_mean_squared_log_error"
P = "regressor__model__"  # parameter prefix inside TransformedTargetRegressor


def _strip(params: dict) -> dict:
    return {k.replace(P, ""): (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def _grid(key: str, X, y, grid: dict, n_jobs: int) -> dict:
    search = GridSearchCV(build_model(key), {P + k: v for k, v in grid.items()},
                          scoring=SCORING, cv=cv_splitter(), n_jobs=n_jobs)
    search.fit(X, y)
    return {"best_params": _strip(search.best_params_), "best_cv_rmsle": float(-search.best_score_),
            "n_candidates": len(search.cv_results_["params"]), "method": "GridSearchCV"}


def _random(key: str, X, y, dists: dict, n_iter: int, n_jobs: int) -> dict:
    search = RandomizedSearchCV(build_model(key), {P + k: v for k, v in dists.items()}, n_iter=n_iter,
                                scoring=SCORING, cv=cv_splitter(), n_jobs=n_jobs, random_state=SEED)
    search.fit(X, y)
    return {"best_params": _strip(search.best_params_), "best_cv_rmsle": float(-search.best_score_),
            "n_candidates": n_iter, "method": "RandomizedSearchCV"}


def _optuna_space(key: str, trial) -> dict:
    if key == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
        }
    if key == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 2500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1, log=True),
        }
    if key == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 32),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 40),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.15, 0.7),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    raise KeyError(key)


_RANDOM_FALLBACK = {
    "gradient_boosting": {"n_estimators": randint(300, 1500), "learning_rate": loguniform(0.01, 0.1),
                          "max_depth": randint(2, 6), "min_samples_leaf": randint(2, 20),
                          "subsample": uniform(0.6, 0.4)},
    "xgboost": {"n_estimators": randint(400, 2500), "learning_rate": loguniform(0.01, 0.1),
                "max_depth": randint(2, 7), "subsample": uniform(0.5, 0.5),
                "colsample_bytree": uniform(0.2, 0.6)},
    "lightgbm": {"n_estimators": randint(500, 3000), "learning_rate": loguniform(0.005, 0.08),
                 "num_leaves": randint(4, 33), "min_child_samples": randint(3, 41),
                 "colsample_bytree": uniform(0.15, 0.55)},
}


def _optuna(key: str, X, y, n_trials: int, n_jobs: int) -> dict:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = cv_splitter()

    def objective(trial):
        params = _optuna_space(key, trial)
        scores = cross_val_score(build_model(key, params), X, y, scoring=SCORING, cv=cv, n_jobs=n_jobs)
        return float(-scores.mean())

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return {"best_params": dict(study.best_params), "best_cv_rmsle": float(study.best_value),
            "n_candidates": n_trials, "method": "Optuna TPE",
            "trials": [{"number": t.number, "value": t.value, "params": t.params}
                       for t in study.trials if t.value is not None]}


def tune_model(key: str, X, y, quick: bool = False, n_jobs: int = -1) -> dict:
    """Tune one model; returns best params, best CV RMSLE and search metadata."""
    cfg = CONFIG["tuning"]
    div = 3 if quick else 1
    t0 = time.perf_counter()
    if key == "linear_regression":
        res = {"best_params": {}, "best_cv_rmsle": None, "n_candidates": 0, "method": "none (no hyperparameters)"}
    elif key == "ridge":
        res = _grid(key, X, y, {"alpha": cfg["ridge_alphas"]}, n_jobs)
    elif key == "lasso":
        res = _grid(key, X, y, {"alpha": cfg["lasso_alphas"]}, n_jobs)
    elif key == "decision_tree":
        res = _grid(key, X, y, cfg["decision_tree_grid"], n_jobs)
    elif key == "random_forest":
        dists = {"n_estimators": [200, 300, 500], "max_features": [0.2, 0.33, 0.5, "sqrt"],
                 "min_samples_leaf": [1, 2, 3, 5], "max_depth": [None, 15, 25]}
        res = _random(key, X, y, dists, max(4, cfg["random_forest_iterations"] // div), n_jobs=1)
    elif key in ("gradient_boosting", "xgboost", "lightgbm"):
        trials = max(5, cfg[f"{key}_trials"] // div)
        if OPTUNA_AVAILABLE:
            res = _optuna(key, X, y, trials, n_jobs)
        else:
            res = _random(key, X, y, _RANDOM_FALLBACK[key], trials, n_jobs)
    else:
        raise KeyError(key)
    res["tuning_seconds"] = time.perf_counter() - t0
    res["model"] = key
    return res


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
