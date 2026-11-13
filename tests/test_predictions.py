"""Prediction engine: validation, missing features, inverse transform, uncertainty."""

import numpy as np
import pandas as pd
import pytest

from src.prediction import PropertyInputError, apply_inputs, set_feature, validate_inputs
from src.preprocessing import RAW_FEATURES
from src.uncertainty import conformal_quantiles, empirical_coverage, interval_from_quantile
from tests.conftest import finite_positive


def test_valid_input_prediction(engine, valid_inputs):
    r = engine.value(valid_inputs)
    assert np.isfinite(r.estimate) and 50_000 < r.estimate < 600_000
    assert r.interval.lower < r.estimate < r.interval.upper
    assert list(r.record.columns) == RAW_FEATURES


@pytest.mark.parametrize("field,value", [("OverallQual", 11), ("GrLivArea", -10), ("YearBuilt", 3000),
                                         ("Neighborhood", "Atlantis"), ("GarageCars", 2.5), ("LotArea", "big")])
def test_invalid_inputs_raise_readable_errors(engine, valid_inputs, field, value):
    bad = {**valid_inputs, field: value}
    with pytest.raises(PropertyInputError) as e:
        engine.value(bad)
    assert any(field in m for m in e.value.errors)


def test_logical_consistency_errors(engine, valid_inputs):
    with pytest.raises(PropertyInputError, match="remodeled"):
        engine.value({**valid_inputs, "YearRemodAdd": 1990, "YearBuilt": 2000})
    with pytest.raises(PropertyInputError, match="Garage area"):
        engine.value({**valid_inputs, "GarageCars": 0, "GarageArea": 400})


def test_missing_required_fields(engine):
    with pytest.raises(PropertyInputError) as e:
        engine.value({"OverallQual": 7})
    assert any("GrLivArea" in m for m in e.value.errors)


def test_missing_optional_features_filled_from_template(engine):
    r = engine.value({"Neighborhood": "NAmes", "OverallQual": 6, "GrLivArea": 1400, "YearBuilt": 1970})
    assert finite_positive([r.estimate])
    assert r.record["Neighborhood"].iloc[0] == "NAmes"


def test_inverse_transformation_matches_log_model(engine, valid_inputs):
    r = engine.value(valid_inputs)
    inner = engine.model.regressor_
    log_pred = inner.predict(r.record)[0]
    assert r.estimate == pytest.approx(np.expm1(log_pred))


def test_same_pipeline_for_training_and_prediction(engine, split):
    """Prediction engine output equals calling the fitted training pipeline directly."""
    _, Xva, _, _ = split
    direct = engine.model.predict(Xva.head(20))
    via_engine = engine.predict_frame(Xva.head(20))
    np.testing.assert_allclose(direct, via_engine)


def test_consistency_rules():
    base = pd.Series({c: np.nan for c in RAW_FEATURES}, dtype=object)
    base.update(pd.Series({"1stFlrSF": 1000, "2ndFlrSF": 500, "LowQualFinSF": 0, "GrLivArea": 1500,
                           "TotalBsmtSF": 800, "BsmtFinSF1": 400, "BsmtFinSF2": 0, "BsmtUnfSF": 400,
                           "GarageCars": 2, "GarageArea": 500, "GarageType": "Attchd", "YearBuilt": 1990,
                           "YearRemodAdd": 1990, "GarageYrBlt": 1990, "Fireplaces": 1, "FireplaceQu": "TA"}))
    d = {"pool_area": 500, "pool_qc": "Gd", "garage_area_per_car": 250}
    r = apply_inputs(base, {"GrLivArea": 3000}, d)
    assert r["1stFlrSF"] + r["2ndFlrSF"] == 3000 and r["1stFlrSF"] == 2000
    r = apply_inputs(base, {"GarageCars": 0}, d)
    assert r["GarageArea"] == 0 and pd.isna(r["GarageType"])
    r = apply_inputs(base, {"TotalBsmtSF": 0}, d)
    assert pd.isna(r["BsmtQual"]) and r["BsmtFinSF1"] == 0
    r = apply_inputs(base, {"HasFireplace": False}, d)
    assert r["Fireplaces"] == 0 and pd.isna(r["FireplaceQu"])
    r = apply_inputs(base, {"HouseStyle": "1Story"}, d)
    assert r["2ndFlrSF"] == 0 and r["1stFlrSF"] == 1500


def test_set_feature_never_creates_negative_areas(split):
    _, Xva, _, _ = split
    for v in (0, 37, 1500):
        out = set_feature(Xva, "TotalBsmtSF", v)
        assert (out["BsmtUnfSF"] >= 0).all()


def test_extrapolation_warning(engine, valid_inputs):
    r = engine.value({**valid_inputs, "GrLivArea": 7900})
    assert any("outside the training data range" in w for w in r.warnings)


def test_uncertainty_intervals():
    rng = np.random.default_rng(0)
    y = rng.uniform(1e5, 3e5, 2000)
    pred = y * np.exp(rng.normal(0, 0.1, 2000))
    q = conformal_quantiles(y[:1000], pred[:1000], (0.8, 0.9))
    assert q[0.9] > q[0.8] > 0
    cov = empirical_coverage(y[1000:], pred[1000:], q)
    assert abs(cov[0.9]["empirical"] - 0.9) < 0.04
    iv = interval_from_quantile(200000, q[0.9], 0.9)
    assert iv.lower < 200000 < iv.upper and iv.width > 0
    with pytest.raises(ValueError):
        interval_from_quantile(-5, 0.1, 0.9)
