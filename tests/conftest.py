"""Shared pytest fixtures (real Kaggle data; small fast models)."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.data_loader import datasets_available, load_test, load_train, split_features_target  # noqa: E402
from src.evaluate import cross_validate_oof  # noqa: E402
from src.models import build_model  # noqa: E402
from src.prediction import FIELD_SPECS, PredictionEngine  # noqa: E402
from src.preprocessing import RAW_FEATURES, make_train_validation_split  # noqa: E402
from src.train import build_template  # noqa: E402
from src.uncertainty import conformal_quantiles  # noqa: E402

requires_data = pytest.mark.skipif(not datasets_available(), reason="Kaggle CSVs not present in data/raw")


@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    if not datasets_available():
        pytest.skip("Kaggle CSVs not present in data/raw")
    return load_train()


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    if not datasets_available():
        pytest.skip("Kaggle CSVs not present in data/raw")
    return load_test()


@pytest.fixture(scope="session")
def split(train_df):
    X, y = split_features_target(train_df)
    return make_train_validation_split(X, y)


@pytest.fixture(scope="session")
def ridge_model(split):
    Xtr, Xva, ytr, yva = split
    return build_model("ridge", {"alpha": 10.0}).fit(Xtr, ytr)


@pytest.fixture(scope="session")
def engine(split, ridge_model, train_df):
    """A PredictionEngine built from a quickly trained Ridge model (independent of train_all)."""
    Xtr, Xva, ytr, yva = split
    cv = cross_validate_oof(build_model("ridge", {"alpha": 10.0}), Xtr, ytr, n_splits=3)
    X, _ = split_features_target(train_df)
    cat_opts = {c: sorted(X[c].dropna().astype(str).unique().tolist())
                for c in RAW_FEATURES if not pd.api.types.is_numeric_dtype(X[c]) or c == "MSSubClass"}
    ranges = {k: {"min": float(Xtr[k].min()), "max": float(Xtr[k].max())} for k in FIELD_SPECS if k in Xtr}
    bundle = {
        "model": ridge_model, "model_key": "ridge", "model_label": "Ridge Regression",
        "template": build_template(Xtr), "category_options": cat_opts, "numeric_ranges": ranges,
        "conformal_quantiles": conformal_quantiles(ytr, cv["oof_pred"], (0.8, 0.9, 0.95)), "default_level": 0.9,
        "defaults": {"pool_area": 500.0, "pool_qc": "Gd", "garage_area_per_car": 260.0},
        "background": Xtr.sample(100, random_state=0),
    }
    return PredictionEngine(bundle)


@pytest.fixture
def valid_inputs() -> dict:
    return {"Neighborhood": "CollgCr", "BldgType": "1Fam", "HouseStyle": "2Story", "OverallQual": 7,
            "OverallCond": 5, "YearBuilt": 2003, "YearRemodAdd": 2003, "YrSold": 2008, "GrLivArea": 1710.0,
            "LotArea": 8450.0, "TotalBsmtSF": 856.0, "BedroomAbvGr": 3, "FullBath": 2, "HalfBath": 1,
            "KitchenAbvGr": 1, "TotRmsAbvGrd": 8, "GarageCars": 2, "GarageArea": 548.0, "Fireplaces": 0,
            "HasCentralAir": True, "HasPool": False, "KitchenQual": "Gd", "ExterQual": "Gd"}


def finite_positive(a) -> bool:
    a = np.asarray(a, dtype=float)
    return bool(np.all(np.isfinite(a)) and np.all(a > 0))
