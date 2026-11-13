"""Preprocessing: missing values, encoding, transforms, splitting and leakage."""

import numpy as np
import pandas as pd

from src.models import build_model
from src.preprocessing import (
    SkewLog1p, build_preprocessor, clean_raw_frame, make_train_validation_split, remove_training_outliers,
)
from src.data_loader import split_features_target


def test_absent_feature_missing_values(train_df):
    c = clean_raw_frame(train_df)
    no_bsmt = train_df["TotalBsmtSF"] == 0
    # NA basement quality means "no basement" → ordinal 0, not an imputed value
    assert (c.loc[no_bsmt, "BsmtQual"] == 0).all()
    assert (c["PoolQC"].notna()).all()
    assert (c.loc[train_df["PoolArea"] == 0, "PoolQC"] == 0).all()
    assert (c["GarageType"].loc[train_df["GarageType"].isna()] == "None").all()
    # LotFrontage is a genuinely missing measurement: left for the fitted imputer
    assert c["LotFrontage"].isna().sum() == 259


def test_not_everything_replaced_with_zero(train_df):
    c = clean_raw_frame(train_df)
    no_gar = train_df["GarageType"].isna()
    assert (c.loc[no_gar, "GarageYrBlt"] == train_df.loc[no_gar, "YearBuilt"]).all()


def test_mssubclass_is_categorical(train_df):
    c = clean_raw_frame(train_df.head(5))
    assert c["MSSubClass"].map(type).eq(str).all()


def test_categorical_encoding_handles_unseen(split):
    Xtr, Xva, ytr, yva = split
    m = build_model("ridge").fit(Xtr, ytr)
    weird = Xva.head(3).copy()
    weird["Neighborhood"] = "Atlantis"
    weird["RoofMatl"] = "Unobtainium"
    p = m.predict(weird)
    assert np.all(np.isfinite(p)) and np.all(p > 0)


def test_preprocessor_outputs_numeric_matrix(split):
    Xtr, *_ = split
    from src.feature_engineering import add_engineered_features
    Z = build_preprocessor("linear").fit_transform(add_engineered_features(clean_raw_frame(Xtr)))
    assert np.isfinite(Z).all() and Z.shape[0] == len(Xtr)


def test_skew_transform_learned_on_training_only():
    tr = np.array([[1, 10], [2, 11], [3, 12], [1000, 13]], dtype=float)
    t = SkewLog1p(0.75).fit(tr)
    assert t.mask_.tolist() == [True, False]
    out = t.transform(np.array([[-5.0, 3.0]]))  # unseen negative must not produce -inf/NaN
    assert np.isfinite(out).all()


def test_train_validation_separation(train_df):
    X, y = split_features_target(train_df)
    Xtr, Xva, ytr, yva = make_train_validation_split(X, y)
    assert set(Xtr.index).isdisjoint(Xva.index)
    assert len(Xtr) + len(Xva) == len(X)
    assert abs(len(Xva) / len(X) - 0.2) < 0.01
    Xtr2, Xva2, *_ = make_train_validation_split(X, y)
    assert Xva.index.equals(Xva2.index), "split must be reproducible"


def test_no_leakage_imputer_fitted_on_training_only(split):
    """Changing validation rows must not change the fitted preprocessing."""
    Xtr, Xva, ytr, yva = split
    m1 = build_model("ridge").fit(Xtr, ytr)
    Xva_mod = Xva.copy()
    Xva_mod["LotFrontage"] = 99999.0
    m2 = build_model("ridge").fit(Xtr, ytr)
    m2.predict(Xva_mod)
    imp1 = m1.regressor_.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_
    imp2 = m2.regressor_.named_steps["prep"].named_transformers_["num"].named_steps["impute"].statistics_
    np.testing.assert_allclose(imp1, imp2)
    lf_idx = list(m1.regressor_.named_steps["prep"].transformers_[0][2]).index("LotFrontage")
    assert imp1[lf_idx] == Xtr["LotFrontage"].median()


def test_outlier_removal_only_targets_documented_rows(train_df):
    X, y = split_features_target(train_df)
    Xr, yr, removed = remove_training_outliers(X, y)
    assert len(removed) == 2
    assert all(X.loc[removed, "GrLivArea"] > 4000) and all(y.loc[removed] < 300000)
