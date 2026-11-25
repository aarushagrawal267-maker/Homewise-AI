"""Re-check the reported metrics against the saved models, and add baselines and confidence intervals.

Usage:
    python scripts/audit_metrics.py              # about 1-2 min on a laptop
    python scripts/audit_metrics.py --skip-nested  # skip the ensemble-weight CV check (a few seconds)

What it does (no retraining of the saved models):
1. Rebuilds the seed-42 train/validation split and checks it matches the split stored in the bundle.
2. Re-scores the saved selected model and every saved candidate model on the validation split and
   compares the results with outputs/reports/training_summary.json and model_comparison.csv.
3. Recomputes conformal-interval coverage on the validation split.
4. Scores three simple baselines fitted on the training split.
5. Adds 95% bootstrap confidence intervals for the selected model's validation metrics, and paired
   bootstrap intervals for its RMSLE gap to the other ensemble and the best single model.
6. Measures how optimistic the weighted ensemble's CV score is: the blend weights are re-fitted inside
   each CV fold (without that fold) instead of on all out-of-fold predictions.

Writes outputs/reports/metric_audit.json. Exits with status 1 if any reported number does not match.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import CONFIG, PATHS  # noqa: E402
from src.data_loader import MANUAL_INSTRUCTIONS, datasets_available, load_train, split_features_target  # noqa: E402
from src.evaluate import (  # noqa: E402
    bootstrap_metric_ci,
    cross_validate_oof,
    cv_splitter,
    naive_baselines,
    paired_bootstrap_rmsle_diff,
    regression_metrics,
    residual_stats,
)
from src.preprocessing import make_train_validation_split, remove_training_outliers  # noqa: E402
from src.uncertainty import empirical_coverage  # noqa: E402

REL_TOL = 1e-6
COMPARISON_COLUMNS = {"MAE": "Val MAE", "RMSE": "Val RMSE", "R2": "Val R2", "RMSLE": "Val RMSLE",
                      "MedianAE": "Val MedianAE", "MAPE": "Val MAPE %"}


def close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(1.0, abs(b))


def nested_weight_check(summary: dict, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    from src.models import build_model
    from src.train import optimise_weights

    members = summary["ensemble_members"]
    oof = {k: cross_validate_oof(build_model(k, summary["tuning"][k]["best_params"]), X_train, y_train)["oof_pred"]
           for k in members}
    target = np.log1p(y_train.to_numpy(float))
    L = np.column_stack([np.log1p(oof[k]) for k in members])
    w_all = optimise_weights(y_train, oof)
    in_sample, refit = [], []
    for tr, va in cv_splitter().split(X_train):
        w = optimise_weights(y_train.iloc[tr], {k: oof[k][tr] for k in members})
        refit.append(float(np.sqrt(np.mean((L[va] @ w - target[va]) ** 2))))
        in_sample.append(float(np.sqrt(np.mean((L[va] @ w_all - target[va]) ** 2))))
    return {
        "members": members,
        "weights_recomputed": [float(w) for w in w_all],
        "weights_reported": summary["ensemble_weights"],
        "cv_rmsle_weights_fit_on_all_oof": float(np.mean(in_sample)),
        "cv_rmsle_weights_refit_per_fold": float(np.mean(refit)),
        "cv_rmsle_refit_fold_values": refit,
        "optimism": float(np.mean(refit) - np.mean(in_sample)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-nested", action="store_true", help="skip the per-fold ensemble-weight check")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples (default 2000)")
    args = ap.parse_args()
    if not datasets_available():
        print(MANUAL_INSTRUCTIONS)
        return 1
    summary = json.loads((PATHS.reports / "training_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(PATHS.reports / "model_comparison.csv").set_index("model_key")
    bundle = joblib.load(PATHS.model_bundle)
    mismatches: list[str] = []

    X, y = split_features_target(load_train())
    X_train, X_val, y_train, y_val = make_train_validation_split(X, y)
    if CONFIG["preprocessing"]["remove_training_outliers"]:
        X_train, y_train, _ = remove_training_outliers(X_train, y_train)
    if list(X_train.index) != bundle["train_index"] or list(X_val.index) != bundle["val_index"]:
        mismatches.append("train/validation split differs from the one stored in the model bundle")

    # 1. Selected model
    val_pred = bundle["model"].predict(X_val)
    val_metrics = regression_metrics(y_val, val_pred)
    for k, v in val_metrics.items():
        if not close(v, summary["best_validation"][k]):
            mismatches.append(f"selected model validation {k}: recomputed {v} vs reported {summary['best_validation'][k]}")
    resid = residual_stats(y_val, val_pred)
    for k, v in resid.items():
        if not close(v, summary["best_residuals"][k]):
            mismatches.append(f"residual {k}: recomputed {v} vs reported {summary['best_residuals'][k]}")
    quantiles = {float(k): v for k, v in summary["conformal_quantiles"].items()}
    coverage = empirical_coverage(y_val, val_pred, quantiles)
    for lvl, c in coverage.items():
        rep = summary["coverage"][str(lvl)]["empirical"]
        if not close(c["empirical"], rep):
            mismatches.append(f"coverage {lvl}: recomputed {c['empirical']} vs reported {rep}")

    # 2. Every candidate model in the comparison table
    candidate_preds = {}
    for key in comparison.index:
        path = PATHS.candidates_dir / f"{key}.joblib"
        if not path.exists():
            mismatches.append(f"missing candidate model file {path.name}")
            continue
        candidate_preds[key] = joblib.load(path).predict(X_val)
        m = regression_metrics(y_val, candidate_preds[key])
        for k, col in COMPARISON_COLUMNS.items():
            if not close(m[k], float(comparison.loc[key, col])):
                mismatches.append(f"{key} {col}: recomputed {m[k]} vs reported {comparison.loc[key, col]}")

    # 3. Baselines, confidence intervals, paired comparisons
    baselines = {name: regression_metrics(y_val, p) for name, p in naive_baselines(X_train, y_train, X_val).items()}
    ci = bootstrap_metric_ci(y_val, val_pred, n_boot=args.n_boot)
    best_key = summary["best_model"]
    singles = comparison[~comparison.index.str.contains("ensemble")].sort_values("CV RMSLE")
    others = [k for k in comparison.index if "ensemble" in k and k != best_key] + [singles.index[0]]
    paired = {k: paired_bootstrap_rmsle_diff(y_val, val_pred, candidate_preds[k], n_boot=args.n_boot)
              for k in others if k in candidate_preds}
    cov_ci = {}
    rng = np.random.default_rng(0)
    abs_log_res = np.abs(np.log1p(y_val.to_numpy(float)) - np.log1p(val_pred))
    for lvl, q in quantiles.items():
        inside = abs_log_res <= q
        draws = [inside[rng.integers(0, len(inside), len(inside))].mean() for _ in range(args.n_boot)]
        cov_ci[str(lvl)] = {"lower": float(np.percentile(draws, 2.5)), "upper": float(np.percentile(draws, 97.5)),
                            "interval_bounds_pct": [float((np.exp(-q) - 1) * 100), float((np.exp(q) - 1) * 100)]}

    nested = None
    if not args.skip_nested and best_key == "weighted_ensemble":
        nested = nested_weight_check(summary, X_train, y_train)
        if not np.allclose(nested["weights_recomputed"], nested["weights_reported"], atol=1e-4):
            mismatches.append(f"ensemble weights: recomputed {nested['weights_recomputed']} "
                              f"vs reported {nested['weights_reported']}")

    report = {
        "run_id": summary["run_id"], "n_train": len(X_train), "n_validation": len(X_val),
        "mismatches": mismatches,
        "selected_model": best_key, "validation_metrics": val_metrics,
        "validation_metrics_ci95": ci, "paired_rmsle_diff_vs": paired,
        "coverage_ci95": cov_ci, "baselines_validation": baselines, "nested_ensemble_weights": nested,
    }
    (PATHS.reports / "metric_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validation split: {len(X_val)} homes. Selected model: {summary['best_model_label']}")
    for k in ("RMSLE", "MAE", "RMSE", "R2", "MAPE", "within_10pct"):
        v = val_metrics.get(k, resid["share_within_10pct"] if k == "within_10pct" else None)
        print(f"  {k:<13} {v:>12.4f}   95% CI {ci[k]['lower']:.4f} .. {ci[k]['upper']:.4f}")
    for k, d in paired.items():
        print(f"  RMSLE minus {k}: {d['diff']:+.4f} (95% CI {d['lower']:+.4f} .. {d['upper']:+.4f})")
    for lvl, c in cov_ci.items():
        lo, hi = c["interval_bounds_pct"]
        print(f"  {float(lvl):.0%} interval: {lo:+.1f}% / {hi:+.1f}% around the estimate; "
              f"coverage {coverage[float(lvl)]['empirical']:.1%} (95% CI {c['lower']:.1%} .. {c['upper']:.1%})")
    for name, m in baselines.items():
        print(f"  Baseline {name:<38} RMSLE {m['RMSLE']:.4f}  MAE ${m['MAE']:,.0f}  R2 {m['R2']:.3f}")
    if nested:
        print(f"  Weighted ensemble CV RMSLE: {nested['cv_rmsle_weights_fit_on_all_oof']:.4f} with weights fitted on "
              f"all OOF predictions, {nested['cv_rmsle_weights_refit_per_fold']:.4f} with weights re-fitted per fold "
              f"(optimism {nested['optimism']:.4f})")
    if mismatches:
        print("\nMISMATCHES between saved models and reported metrics:")
        for m in mismatches:
            print("  - " + m)
        return 1
    print("\nAll reported validation metrics, residuals, coverage and per-model scores match the saved models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
