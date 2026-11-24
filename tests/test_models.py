"""Model registry, training, persistence and reproducibility."""

import joblib
import numpy as np
import pytest

from src.config import PATHS
from src.evaluate import cross_validate_oof, regression_metrics
from src.models import MODEL_SPECS, available_models, build_model, build_stacking_ensemble, build_weighted_ensemble
from tests.conftest import finite_positive


def test_all_required_models_registered():
    for k in ["linear_regression", "ridge", "lasso", "decision_tree", "random_forest", "gradient_boosting"]:
        assert k in available_models()
    assert set(available_models()) <= set(MODEL_SPECS)


@pytest.mark.parametrize("key", ["linear_regression", "ridge", "lasso", "decision_tree", "random_forest",
                                 "gradient_boosting", "xgboost", "lightgbm"])
def test_model_initialises_trains_and_predicts(key, split):
    if key not in available_models():
        pytest.skip(f"{key} not installed")
    Xtr, Xva, ytr, yva = split
    params = {"n_estimators": 60} if key in ("random_forest", "gradient_boosting", "xgboost", "lightgbm") else {}
    m = build_model(key, params).fit(Xtr, ytr)
    p = m.predict(Xva)
    assert p.shape == (len(Xva),) and finite_positive(p)
    # predictions are in dollars (inverse log transform applied)
    assert 20_000 < np.median(p) < 500_000


def test_ensembles_train(split):
    Xtr, Xva, ytr, yva = split
    w = build_weighted_ensemble({"ridge": {}, "lasso": {}}, [0.5, 0.5]).fit(Xtr, ytr)
    s = build_stacking_ensemble({"ridge": {}, "lasso": {}}, cv=3).fit(Xtr, ytr)
    assert finite_positive(w.predict(Xva)) and finite_positive(s.predict(Xva))


def test_save_and_load_roundtrip(tmp_path, ridge_model, split):
    _, Xva, _, _ = split
    p = tmp_path / "m.joblib"
    joblib.dump(ridge_model, p)
    loaded = joblib.load(p)
    np.testing.assert_allclose(loaded.predict(Xva), ridge_model.predict(Xva))


def test_reproducible_with_seed(split):
    Xtr, Xva, ytr, yva = split
    a = build_model("random_forest", {"n_estimators": 30}).fit(Xtr, ytr).predict(Xva)
    b = build_model("random_forest", {"n_estimators": 30}).fit(Xtr, ytr).predict(Xva)
    np.testing.assert_allclose(a, b)


def test_metrics_and_cv(split):
    Xtr, Xva, ytr, yva = split
    m = build_model("ridge").fit(Xtr, ytr)
    met = regression_metrics(yva, m.predict(Xva))
    assert {"MAE", "RMSE", "R2", "RMSLE", "MedianAE", "MAPE"} <= set(met)
    assert met["R2"] > 0.8 and met["RMSE"] >= met["MAE"]
    cv = cross_validate_oof(build_model("ridge"), Xtr, ytr, n_splits=3)
    assert len(cv["cv_fold_rmsle"]) == 3 and finite_positive(cv["oof_pred"])
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1, np.inf])


def test_saved_bundle_loads_if_trained():
    if not PATHS.model_bundle.exists():
        pytest.skip("run scripts/train_all.py to create the bundle")
    from src.prediction import PredictionEngine
    eng = PredictionEngine.load()
    p = eng.predict_frame(eng.template.to_frame().T)
    assert finite_positive(p)


def test_engine_is_cached_not_retrained():
    """The app loads the persisted model once; repeated access returns the same object."""
    if not PATHS.model_bundle.exists():
        pytest.skip("run scripts/train_all.py to create the bundle")
    from src.prediction import load_engine
    assert load_engine() is load_engine()


@pytest.mark.parametrize("content, expected", [
    (b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 438000\n", "git lfs pull"),
    (b"\x80\x04not a real pickle", "Could not load"),
])
def test_unloadable_bundle_gives_clear_error(tmp_path, content, expected):
    """A Git LFS pointer or corrupt file must not surface as a cryptic unpickling traceback."""
    from src.prediction import ModelLoadError, PredictionEngine
    p = tmp_path / "homewise_model.joblib"
    p.write_bytes(content)
    with pytest.raises(ModelLoadError, match=expected):
        PredictionEngine.load(p)


def test_non_bundle_file_rejected(tmp_path):
    from src.prediction import ModelLoadError, PredictionEngine
    p = tmp_path / "homewise_model.joblib"
    joblib.dump({"model": None}, p)
    with pytest.raises(ModelLoadError, match="missing"):
        PredictionEngine.load(p)


def test_missing_bundle_raises_not_trained(tmp_path):
    from src.prediction import ModelNotTrainedError, PredictionEngine
    with pytest.raises(ModelNotTrainedError, match="train_all.py"):
        PredictionEngine.load(tmp_path / "absent.joblib")
