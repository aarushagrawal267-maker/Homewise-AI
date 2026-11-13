"""Engineered features."""

import numpy as np
import pandas as pd

from src.feature_engineering import ALL_ENGINEERED, FEATURE_GROUPS, FEATURE_RATIONALE, FeatureEngineer, add_engineered_features
from src.preprocessing import clean_raw_frame


def test_every_feature_documented():
    assert set(ALL_ENGINEERED) == set(FEATURE_RATIONALE)


def test_values_are_correct(train_df):
    raw = train_df.head(50)
    f = add_engineered_features(clean_raw_frame(raw))
    r = raw.iloc[0]
    assert f["TotalSF"].iloc[0] == r["TotalBsmtSF"] + r["1stFlrSF"] + r["2ndFlrSF"]
    assert f["TotalBathrooms"].iloc[0] == r["FullBath"] + 0.5 * r["HalfBath"] + r["BsmtFullBath"] + 0.5 * r["BsmtHalfBath"]
    assert f["HouseAge"].iloc[0] == r["YrSold"] - r["YearBuilt"]
    assert f["QualityArea"].iloc[0] == r["OverallQual"] * r["GrLivArea"]
    assert f["HasGarage"].iloc[0] == int(r["GarageCars"] > 0)
    assert f["HasCentralAir"].iloc[0] == int(r["CentralAir"] == "Y")


def test_no_infinite_or_negative_ages(train_df):
    f = add_engineered_features(clean_raw_frame(train_df))
    num = f[ALL_ENGINEERED].select_dtypes("number")
    assert np.isfinite(num.to_numpy(float)).all()
    assert (f["HouseAge"] >= 0).all() and (f["YearsSinceRemodel"] >= 0).all()
    assert (f.loc[f["HasGarage"] == 0, "AreaPerGarageCar"] == 0).all()


def test_stateless_and_toggleable(train_df):
    raw = clean_raw_frame(train_df.head(20))
    off = FeatureEngineer(enabled=False).fit(raw).transform(raw)
    assert list(off.columns) == list(raw.columns)
    only_age = FeatureEngineer(groups=("age",)).fit(raw).transform(raw)
    assert set(FEATURE_GROUPS["age"]) <= set(only_age.columns)
    assert "TotalSF" not in only_age


def test_does_not_use_target(train_df):
    with_target = add_engineered_features(clean_raw_frame(train_df.head(10)))
    assert "SalePrice" not in with_target
