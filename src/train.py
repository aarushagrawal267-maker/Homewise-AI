"""End-to-end training orchestration.

Phases
------
1. Load and validate train.csv / test.csv.
2. EDA summary + static figures.
3. Reproducible train/validation split (validation is held out until step 8).
4. Feature-engineering ablation (CV on the training split).
5. Hyperparameter optimisation for each model (CV on the training split).
6. Cross-validated out-of-fold predictions for every tuned model.
7. Ensembles (weighted average, stacking) built from the best CV models.
8. Model selection by CV RMSLE, then a single evaluation of every model on the
   validation split (reported, never used for selection).
9. Conformal uncertainty calibration, explainability artefacts, Kaggle submission.
10. Persist the model bundle and reports.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import CONFIG, ID_COL, PATHS, SEED, TARGET
from .data_loader import dataset_status, load_test, load_train, split_features_target
from .data_validation import validate_train_test
from .eda import build_eda_summary, generate_static_figures, save_eda_summary
from .evaluate import cross_validate_oof, regression_metrics, residual_stats, timed_fit_predict
from .experiment_tracker import log_experiment, new_experiment_id
from .explainability import (
    HomeWiseExplainer,
    global_shap_importance,
    partial_dependence_1d,
    partial_dependence_2d,
    permutation_importance_raw,
)
from .feature_engineering import FEATURE_GROUPS
from .hyperparameter_tuning import to_jsonable, tune_model
from .models import (
    available_models,
    build_model,
    build_stacking_ensemble,
    build_weighted_ensemble,
    model_label,
)
from .prediction import FIELD_SPECS, set_feature
from .preprocessing import RAW_FEATURES, make_train_validation_split, remove_training_outliers
from .uncertainty import conformal_quantiles, empirical_coverage

warnings.filterwarnings("ignore", category=UserWarning)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(name: str, obj) -> None:
    with open(PATHS.reports / name, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


def feature_ablation(Xtr, ytr) -> pd.DataFrame:
    """CV RMSLE with all engineered features, none, and each group removed."""
    configs = {"no engineered features": (False, None), "all engineered features": (True, None)}
    for g in FEATURE_GROUPS:
        configs[f"without '{g}' group"] = (True, tuple(x for x in FEATURE_GROUPS if x != g))
    rows = []
    for model_key, params in (("ridge", {"alpha": 10.0}), ("lightgbm", {})):
        if model_key not in available_models():
            continue
        for name, (enabled, groups) in configs.items():
            r = cross_validate_oof(build_model(model_key, params, enabled, groups), Xtr, ytr)
            rows.append({"model": model_label(model_key), "configuration": name,
                         "cv_rmsle_mean": r["cv_rmsle_mean"], "cv_rmsle_std": r["cv_rmsle_std"]})
            log(f"  ablation {model_label(model_key):<18} {name:<28} CV RMSLE {r['cv_rmsle_mean']:.4f}")
    df = pd.DataFrame(rows)
    base = df[df.configuration == "all engineered features"].set_index("model")["cv_rmsle_mean"]
    df["delta_vs_all"] = df.apply(lambda r: r.cv_rmsle_mean - base[r.model], axis=1)
    return df


def optimise_weights(y, oof_preds: dict[str, np.ndarray]) -> np.ndarray:
    """Non-negative weights summing to 1 that minimise OOF RMSLE (log-space blend)."""
    keys = list(oof_preds)
    L = np.column_stack([np.log1p(oof_preds[k]) for k in keys])
    t = np.log1p(np.asarray(y))

    def loss(w):
        return np.sqrt(np.mean((L @ w - t) ** 2))

    n = len(keys)
    res = minimize(loss, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


TEMPLATE_KEYS = ["GrLivArea", "OverallQual", "OverallCond", "YearBuilt", "TotalBsmtSF", "GarageCars", "LotArea",
                 "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]


def build_template(X: pd.DataFrame) -> pd.Series:
    """Reference profile = the most representative *real* training sale.

    Per-column medians would produce an incoherent house (e.g. the median first-floor area
    plus the median second-floor area of 0). Instead we pick the single-family, normal-sale
    home whose key characteristics are closest (standardised distance) to the medians.
    """
    cand = X[(X["BldgType"] == "1Fam") & (X["SaleCondition"] == "Normal")]
    cand = cand if len(cand) else X
    Z = cand[TEMPLATE_KEYS].astype(float)
    d = (((Z - X[TEMPLATE_KEYS].astype(float).median()) / X[TEMPLATE_KEYS].astype(float).std()) ** 2).sum(axis=1)
    idx = d.idxmin()
    row = X.loc[idx, RAW_FEATURES].copy()
    row.name = idx
    return row


def run_training(quick: bool = False, skip_figures: bool = False) -> dict:
    t_start = time.perf_counter()
    PATHS.ensure()
    run_id = new_experiment_id("RUN")
    log(f"HomeWise AI training run {run_id} (quick={quick})")

    # 1. Load + validate ----------------------------------------------------
    train_df, test_df = load_train(), load_test()
    validation = validate_train_test(train_df, test_df)
    validation["files"] = dataset_status()
    _write_json("data_validation_report.json", validation)
    if not validation["train"]["ok"]:
        raise RuntimeError(f"Training data failed validation: {validation['train']['errors']}")
    log(f"Loaded train {train_df.shape}, test {test_df.shape}; validation ok={validation['ok']}")

    # 2. EDA ------------------------------------------------------------------
    eda = build_eda_summary(train_df, test_df)
    save_eda_summary(eda)
    if not skip_figures:
        figs = generate_static_figures(train_df)
        log(f"Saved {len(figs)} EDA figures")

    # 3. Split ---------------------------------------------------------------
    X, y = split_features_target(train_df)
    X_train, X_val, y_train, y_val = make_train_validation_split(X, y)
    n_before = len(X_train)
    removed = []
    if CONFIG["preprocessing"]["remove_training_outliers"]:
        X_train, y_train, removed = remove_training_outliers(X_train, y_train)
    split_desc = (f"stratified {1 - CONFIG['split']['validation_size']:.0%}/"
                  f"{CONFIG['split']['validation_size']:.0%} train/validation, seed {SEED}, "
                  f"{CONFIG['split']['cv_folds']}-fold CV on train")
    log(f"Train {len(X_train)} rows (removed {len(removed)} outliers of {n_before}), validation {len(X_val)}")
    pd.DataFrame({"Id": train_df.loc[X_train.index, ID_COL]}).to_csv(PATHS.processed_data / "train_ids.csv", index=False)
    pd.DataFrame({"Id": train_df.loc[X_val.index, ID_COL]}).to_csv(PATHS.processed_data / "validation_ids.csv", index=False)

    # 4. Feature engineering ablation ------------------------------------------
    log("Feature-engineering ablation (CV on training split)...")
    ablation = feature_ablation(X_train, y_train)
    ablation.to_csv(PATHS.reports / "feature_ablation.csv", index=False)

    # 5-6. Tune + CV every model -----------------------------------------------
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    tuning_log = {}
    for key in available_models():
        log(f"Tuning {model_label(key)}...")
        tune = tune_model(key, X_train, y_train, quick=quick)
        tuning_log[key] = tune
        params = tune["best_params"]
        log(f"  best params {params} | search CV RMSLE {tune['best_cv_rmsle']}")
        cv = cross_validate_oof(build_model(key, params), X_train, y_train)
        model, val_pred, train_s, infer_ms = timed_fit_predict(build_model(key, params), X_train, y_train, X_val)
        train_metrics = regression_metrics(y_train, model.predict(X_train))
        val_metrics = regression_metrics(y_val, val_pred)
        results[key] = {"key": key, "label": model_label(key), "params": params, "cv": cv,
                        "train_metrics": train_metrics, "validation_metrics": val_metrics,
                        "val_pred": val_pred, "train_seconds": train_s, "inference_ms_per_row": infer_ms,
                        "tuning_method": tune["method"], "tuning_seconds": tune["tuning_seconds"]}
        fitted[key] = model
        joblib.dump(model, PATHS.candidates_dir / f"{key}.joblib", compress=3)
        log(f"  CV RMSLE {cv['cv_rmsle_mean']:.4f} ± {cv['cv_rmsle_std']:.4f} | "
            f"train {train_s:.1f}s")

    # 7. Ensembles -------------------------------------------------------------
    ranked = sorted(results, key=lambda k: results[k]["cv"]["cv_rmsle_mean"])
    members = [k for k in ranked if k not in ("linear_regression", "decision_tree")][:4]
    log(f"Ensemble candidates (best CV): {members}")
    weights = optimise_weights(y_train, {k: results[k]["cv"]["oof_pred"] for k in members})
    log(f"  OOF-optimised weights: {dict(zip(members, np.round(weights, 3)))}")
    member_params = {k: results[k]["params"] for k in members}
    ens_defs = {
        "weighted_ensemble": (lambda: build_weighted_ensemble(member_params, list(weights)),
                              {"members": members, "weights": [float(w) for w in weights]}),
        "stacking_ensemble": (lambda: build_stacking_ensemble(member_params),
                              {"members": members, "meta_learner": "RidgeCV"}),
    }
    for key, (factory, params) in ens_defs.items():
        log(f"Evaluating {model_label(key)}...")
        cv = cross_validate_oof(factory(), X_train, y_train)
        model, val_pred, train_s, infer_ms = timed_fit_predict(factory(), X_train, y_train, X_val)
        results[key] = {"key": key, "label": model_label(key), "params": params, "cv": cv,
                        "train_metrics": regression_metrics(y_train, model.predict(X_train)),
                        "validation_metrics": regression_metrics(y_val, val_pred), "val_pred": val_pred,
                        "train_seconds": train_s, "inference_ms_per_row": infer_ms,
                        "tuning_method": "weights from OOF predictions" if key == "weighted_ensemble"
                        else "RidgeCV meta-learner", "tuning_seconds": 0.0}
        fitted[key] = model
        joblib.dump(model, PATHS.candidates_dir / f"{key}.joblib", compress=3)
        log(f"  CV RMSLE {cv['cv_rmsle_mean']:.4f} ± {cv['cv_rmsle_std']:.4f}")

    # 8. Selection by CV, then report validation ----------------------------------
    best_key = min(results, key=lambda k: results[k]["cv"]["cv_rmsle_mean"])
    best = results[best_key]
    best_model = fitted[best_key]
    log(f"Selected by CV RMSLE: {best['label']} ({best['cv']['cv_rmsle_mean']:.4f})")

    rows = []
    for k, r in results.items():
        v, t = r["validation_metrics"], r["train_metrics"]
        rows.append({"model_key": k, "Model": r["label"], "CV RMSLE": r["cv"]["cv_rmsle_mean"],
                     "CV RMSLE std": r["cv"]["cv_rmsle_std"], "Train RMSLE": t["RMSLE"],
                     "Val MAE": v["MAE"], "Val RMSE": v["RMSE"], "Val R2": v["R2"], "Val RMSLE": v["RMSLE"],
                     "Val MedianAE": v["MedianAE"], "Val MAPE %": v["MAPE"],
                     "Train time (s)": r["train_seconds"], "Inference (ms/row)": r["inference_ms_per_row"],
                     "Tuning": r["tuning_method"], "Selected": k == best_key})
        log_experiment({"experiment_id": f"{run_id}-{k}", "source": "train_all", "run_id": run_id,
                        "model": k, "model_label": r["label"], "params": r["params"], "split": split_desc,
                        "cv_rmsle_mean": r["cv"]["cv_rmsle_mean"], "cv_rmsle_std": r["cv"]["cv_rmsle_std"],
                        "cv_fold_rmsle": r["cv"]["cv_fold_rmsle"], "validation_metrics": r["validation_metrics"],
                        "train_metrics": r["train_metrics"], "train_seconds": r["train_seconds"],
                        "inference_ms_per_row": r["inference_ms_per_row"], "feature_engineering": True,
                        "selected": k == best_key, "quick_mode": quick})
    comparison = pd.DataFrame(rows).sort_values("CV RMSLE").reset_index(drop=True)
    comparison.to_csv(PATHS.reports / "model_comparison.csv", index=False)

    # Error analysis on validation (best model)
    vp = best["val_pred"]
    err = pd.DataFrame({"Id": train_df.loc[X_val.index, ID_COL].to_numpy(), "actual": y_val.to_numpy(),
                        "predicted": vp, "Neighborhood": X_val["Neighborhood"].to_numpy(),
                        "OverallQual": X_val["OverallQual"].to_numpy(), "GrLivArea": X_val["GrLivArea"].to_numpy(),
                        "SaleCondition": X_val["SaleCondition"].to_numpy()})
    err["error"] = err["predicted"] - err["actual"]
    err["abs_pct_error"] = (err["error"].abs() / err["actual"] * 100)
    err.to_csv(PATHS.reports / "validation_predictions.csv", index=False)
    err["price_band"] = pd.qcut(err["actual"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    band = err.groupby("price_band", observed=True).agg(n=("error", "size"), MAE=("error", lambda s: s.abs().mean()),
                                                        median_abs_pct=("abs_pct_error", "median"),
                                                        mean_signed_error=("error", "mean")).reset_index()
    band.to_csv(PATHS.reports / "error_by_price_band.csv", index=False)
    cond = err.groupby("SaleCondition").agg(n=("error", "size"), median_abs_pct=("abs_pct_error", "median")).reset_index()
    cond.to_csv(PATHS.reports / "error_by_sale_condition.csv", index=False)

    # 9a. Uncertainty ----------------------------------------------------------
    levels = CONFIG["uncertainty"]["levels"]
    quantiles = conformal_quantiles(y_train, best["cv"]["oof_pred"], levels)
    coverage = empirical_coverage(y_val, vp, quantiles)
    for lvl, c in coverage.items():
        log(f"  conformal {lvl:.0%}: validation coverage {c['empirical']:.1%}, "
            f"median relative width {c['median_relative_width']:.1%}")

    # 9b. Explainability --------------------------------------------------------
    log("Computing SHAP, permutation importance and partial dependence...")
    ecfg = CONFIG["explainability"]
    background = X_train.sample(min(200, len(X_train)), random_state=SEED)
    explainer = HomeWiseExplainer(best_model, background)
    shap_sample = X_val.sample(min(ecfg["shap_sample_size"], len(X_val)), random_state=SEED)
    expl = explainer.explain(shap_sample)
    log(f"  SHAP additivity max error (log scale): {expl.additivity_error():.2e}")
    shap_imp = global_shap_importance(expl)
    shap_imp.to_csv(PATHS.reports / "shap_importance.csv", index=False)
    expl.contributions.assign(Id=train_df.loc[shap_sample.index, ID_COL].to_numpy()).to_csv(
        PATHS.reports / "shap_values_validation.csv", index=False)
    feat_vals = expl.feature_values[[c for c in expl.contributions.columns if c in expl.feature_values]]
    feat_vals.assign(Id=train_df.loc[shap_sample.index, ID_COL].to_numpy()).to_csv(
        PATHS.reports / "shap_feature_values_validation.csv", index=False)

    perm = permutation_importance_raw(best_model, X_val, y_val, n_repeats=ecfg["permutation_repeats"])
    perm.to_csv(PATHS.reports / "permutation_importance.csv", index=False)

    pdp_sample = X_train.sample(min(ecfg["pdp_sample_size"], len(X_train)), random_state=SEED)
    pdps = {}
    for f in ecfg["pdp_features"]:
        lo, hi = X_train[f].quantile([0.02, 0.98])
        grid = (np.arange(int(lo), int(hi) + 1) if f in ("OverallQual", "GarageCars")
                else np.linspace(lo, hi, ecfg["pdp_grid_resolution"]).round())
        pdps[f] = partial_dependence_1d(best_model.predict, pdp_sample, f, grid, setter=set_feature)
    _write_json("partial_dependence.json", pdps)
    surf_sample = X_train.sample(150, random_state=SEED)
    surface = partial_dependence_2d(best_model.predict, surf_sample, "GrLivArea",
                                    np.linspace(700, 3200, 22).round(), "OverallQual", np.arange(3, 11),
                                    setter=set_feature)
    _write_json("price_surface.json", surface)

    # 9c. Kaggle submission (competition test set: inference only) -------------
    test_pred = best_model.predict(test_df.drop(columns=[ID_COL]))
    sub = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: test_pred})
    sub.to_csv(PATHS.root / "outputs" / "submission.csv", index=False)
    log(f"Kaggle submission written ({len(sub)} rows; no ground truth available, not scored locally)")

    # 10. Persist bundle ----------------------------------------------------------
    cat_opts = {}
    for c in RAW_FEATURES:
        if not pd.api.types.is_numeric_dtype(X[c]) or c == "MSSubClass":
            cat_opts[c] = sorted(X[c].dropna().astype(str).unique().tolist())
    ranges = {k: {"min": float(X_train[k].min()), "max": float(X_train[k].max()),
                  "p01": float(X_train[k].quantile(0.01)), "p99": float(X_train[k].quantile(0.99)),
                  "median": float(X_train[k].median())}
              for k in FIELD_SPECS if k in X_train}
    pools = X_train[X_train["PoolArea"] > 0]
    gar = X_train[X_train["GarageCars"] > 0]
    import sklearn
    bundle = {
        "model": best_model, "model_key": best_key, "model_label": best["label"], "params": best["params"],
        "cv": {k: v for k, v in best["cv"].items() if k != "oof_pred"},
        "validation_metrics": best["validation_metrics"], "train_metrics": best["train_metrics"],
        "conformal_quantiles": quantiles, "coverage": coverage,
        "default_level": CONFIG["uncertainty"]["default_level"],
        "template": build_template(X_train), "template_id": int(train_df.loc[build_template(X_train).name, ID_COL]), "category_options": cat_opts, "numeric_ranges": ranges,
        "defaults": {"pool_area": float(pools["PoolArea"].median()) if len(pools) else 500.0,
                     "pool_qc": pools["PoolQC"].mode().iloc[0] if len(pools) else "Gd",
                     "garage_area_per_car": float((gar["GarageArea"] / gar["GarageCars"]).median())},
        "background": background, "train_index": X_train.index.tolist(), "val_index": X_val.index.tolist(),
        "removed_outlier_index": removed, "run_id": run_id, "split": split_desc,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    joblib.dump(bundle, PATHS.model_bundle, compress=3)

    summary = {
        "run_id": run_id, "quick_mode": quick, "trained_at": bundle["trained_at"],
        "best_model": best_key, "best_model_label": best["label"], "best_params": best["params"],
        "best_cv": bundle["cv"], "best_validation": best["validation_metrics"],
        "best_train": best["train_metrics"], "best_residuals": residual_stats(y_val, vp),
        "n_models_trained": len(results), "split": split_desc, "n_train": len(X_train), "n_val": len(X_val),
        "removed_outliers_ids": train_df.loc[removed, ID_COL].tolist(),
        "conformal_quantiles": quantiles, "coverage": coverage,
        "ensemble_members": members, "ensemble_weights": [float(w) for w in weights],
        "tuning": {k: {kk: vv for kk, vv in v.items() if kk != "trials"} for k, v in tuning_log.items()},
        "shap_additivity_error": expl.additivity_error(),
        "total_seconds": time.perf_counter() - t_start,
    }
    _write_json("training_summary.json", summary)
    _write_json("tuning_trials.json", {k: v.get("trials", []) for k, v in tuning_log.items()})
    log(f"Done in {summary['total_seconds'] / 60:.1f} min. Best: {best['label']} | "
        f"Val RMSE ${best['validation_metrics']['RMSE']:,.0f} | R² {best['validation_metrics']['R2']:.4f}")
    return summary
