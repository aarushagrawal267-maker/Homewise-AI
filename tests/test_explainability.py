"""Explainability: SHAP additivity, global importance, PDP."""

import numpy as np
import pytest

from src.explainability import (
    SHAP_AVAILABLE, HomeWiseExplainer, global_shap_importance, partial_dependence_1d, partial_dependence_2d,
    permutation_importance_raw,
)
from src.models import build_model, build_weighted_ensemble
from src.prediction import set_feature


def test_linear_shap_is_exact(ridge_model, split):
    Xtr, Xva, _, _ = split
    e = HomeWiseExplainer(ridge_model, Xtr.sample(100, random_state=0)).explain(Xva.head(25))
    assert e.additivity_error() < 1e-8
    assert "Neighborhood" in e.contributions.columns  # one-hot columns aggregated back


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap not installed")
def test_tree_and_ensemble_shap_additive(split):
    Xtr, Xva, ytr, _ = split
    gb = build_model("gradient_boosting", {"n_estimators": 80}).fit(Xtr, ytr)
    e = HomeWiseExplainer(gb, Xtr.head(50)).explain(Xva.head(20))
    assert e.additivity_error() < 1e-6
    ens = build_weighted_ensemble({"ridge": {}, "gradient_boosting": {"n_estimators": 80}}, [0.6, 0.4]).fit(Xtr, ytr)
    e2 = HomeWiseExplainer(ens, Xtr.head(50)).explain(Xva.head(20))
    assert e2.additivity_error() < 1e-6


def test_individual_explanation_row(ridge_model, split):
    Xtr, Xva, _, _ = split
    e = HomeWiseExplainer(ridge_model, Xtr.head(100)).explain(Xva.head(1))
    row = e.row(0)
    pred = float(ridge_model.predict(Xva.head(1))[0])
    assert row["dollar_contribution"].sum() == pytest.approx(pred - e.base_price, rel=1e-6)
    assert {"feature", "value", "pct_effect"} <= set(row.columns)
    assert (row["log_contribution"].abs().diff().dropna() <= 1e-12).all()  # sorted by magnitude


def test_global_importance(ridge_model, split):
    Xtr, Xva, yva_tr, yva = split
    e = HomeWiseExplainer(ridge_model, Xtr.head(100)).explain(Xva.head(60))
    imp = global_shap_importance(e)
    assert imp["mean_abs_shap"].is_monotonic_decreasing and len(imp) > 10
    perm = permutation_importance_raw(ridge_model, Xva.head(120), yva.head(120), n_repeats=2)
    assert len(perm) == 79 and np.isfinite(perm["importance_mean"]).all()
    assert perm.iloc[0]["importance_mean"] > 0


def test_partial_dependence(ridge_model, split):
    Xtr, *_ = split
    pdp = partial_dependence_1d(ridge_model.predict, Xtr.head(40), "OverallQual", [4, 6, 8], setter=set_feature)
    assert len(pdp["average"]) == 3 and pdp["average"][2] > pdp["average"][0]
    surf = partial_dependence_2d(ridge_model.predict, Xtr.head(20), "GrLivArea", [1000, 2000], "OverallQual", [5, 7],
                                 setter=set_feature)
    z = np.asarray(surf["z"])
    assert z.shape == (2, 2) and np.isfinite(z).all()
