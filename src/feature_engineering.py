"""Domain-driven feature engineering.

Every feature below has an explicit rationale. The transformer is *stateless*
(it learns nothing from data), so applying it inside the model pipeline cannot
leak information from validation rows into training.

Groups (can be toggled for ablation studies):

size        TotalSF, TotalBathrooms, TotalPorchSF, LivingLotRatio, AreaPerGarageCar
age         HouseAge, YearsSinceRemodel, GarageAge, WasRemodeled
quality     QualityCondition, QualityArea, QualityBand, ConditionBand
facilities  HasGarage, HasBasement, HasFireplace, HasPool, HasCentralAir, Has2ndFloor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_GROUPS: dict[str, list[str]] = {
    "size": ["TotalSF", "TotalBathrooms", "TotalPorchSF", "LivingLotRatio", "AreaPerGarageCar"],
    "age": ["HouseAge", "YearsSinceRemodel", "GarageAge", "WasRemodeled"],
    "quality": ["QualityCondition", "QualityArea", "QualityBand", "ConditionBand"],
    "facilities": ["HasGarage", "HasBasement", "HasFireplace", "HasPool",
                   "HasCentralAir", "Has2ndFloor"],
}

FEATURE_RATIONALE: dict[str, str] = {
    "TotalSF": "Basement + 1st + 2nd floor area: buyers value total usable space, which no single raw column captures.",
    "TotalBathrooms": "Full baths + 0.5 × half baths across basement and above grade, one comparable bathroom count.",
    "TotalPorchSF": "Open, enclosed, 3-season, screen porch and wood deck area combined; individually sparse.",
    "LivingLotRatio": "Living area / lot area: distinguishes dense lots from large-lot homes of similar size.",
    "AreaPerGarageCar": "Garage area per car space: a proxy for garage generosity beyond capacity.",
    "HouseAge": "Years between construction and sale (YrSold − YearBuilt), comparable across sale years.",
    "YearsSinceRemodel": "Years between last remodel/addition and sale; captures how dated the finishes are.",
    "GarageAge": "Years between garage construction and sale (house year used when no garage).",
    "WasRemodeled": "1 if the remodel year differs from the construction year.",
    "QualityCondition": "OverallQual × OverallCond interaction: high quality in poor condition differs from both alone.",
    "QualityArea": "OverallQual × GrLivArea: the premium for quality scales with how much of it there is.",
    "QualityBand": "OverallQual grouped into Low (1–4), Average (5–6), Good (7–8), Excellent (9–10).",
    "ConditionBand": "OverallCond grouped into Below average (1–4), Average (5), Above average (6–10).",
    "HasGarage": "1 if the property has a garage.",
    "HasBasement": "1 if total basement area > 0.",
    "HasFireplace": "1 if at least one fireplace.",
    "HasPool": "1 if pool area > 0 (very rare in this dataset).",
    "HasCentralAir": "1 if central air conditioning is present.",
    "Has2ndFloor": "1 if second floor area > 0.",
}

ALL_ENGINEERED: list[str] = [f for fs in FEATURE_GROUPS.values() for f in fs]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def add_engineered_features(df: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` (already domain-cleaned) with engineered features."""
    groups = list(FEATURE_GROUPS) if groups is None else groups
    out = df.copy()
    bsmt = _num(out, "TotalBsmtSF").fillna(0)
    first, second = _num(out, "1stFlrSF").fillna(0), _num(out, "2ndFlrSF").fillna(0)
    grliv, lot = _num(out, "GrLivArea"), _num(out, "LotArea")
    yr_sold, yr_built = _num(out, "YrSold"), _num(out, "YearBuilt")
    cars, garea = _num(out, "GarageCars").fillna(0), _num(out, "GarageArea").fillna(0)
    oq, oc = _num(out, "OverallQual"), _num(out, "OverallCond")

    if "size" in groups:
        out["TotalSF"] = bsmt + first + second
        out["TotalBathrooms"] = (
            _num(out, "FullBath").fillna(0) + 0.5 * _num(out, "HalfBath").fillna(0)
            + _num(out, "BsmtFullBath").fillna(0) + 0.5 * _num(out, "BsmtHalfBath").fillna(0)
        )
        porch_cols = ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]
        out["TotalPorchSF"] = sum(_num(out, c).fillna(0) for c in porch_cols)
        out["LivingLotRatio"] = grliv / lot.where(lot > 0)
        out["AreaPerGarageCar"] = np.where(cars > 0, garea / cars.where(cars > 0), 0.0)

    if "age" in groups:
        out["HouseAge"] = (yr_sold - yr_built).clip(lower=0)
        out["YearsSinceRemodel"] = (yr_sold - _num(out, "YearRemodAdd")).clip(lower=0)
        gyr = _num(out, "GarageYrBlt").fillna(yr_built)
        out["GarageAge"] = (yr_sold - gyr).clip(lower=0)
        out["WasRemodeled"] = (_num(out, "YearRemodAdd") != yr_built).astype(int)

    if "quality" in groups:
        out["QualityCondition"] = oq * oc
        out["QualityArea"] = oq * grliv
        out["QualityBand"] = pd.cut(oq, bins=[0, 4, 6, 8, 10],
                                    labels=["Low", "Average", "Good", "Excellent"]).astype(object)
        out["ConditionBand"] = pd.cut(oc, bins=[0, 4, 5, 10],
                                      labels=["BelowAverage", "Average", "AboveAverage"]).astype(object)

    if "facilities" in groups:
        gtype = out["GarageType"].astype(object) if "GarageType" in out else pd.Series("None", index=out.index)
        out["HasGarage"] = ((cars > 0) | (gtype.fillna("None") != "None")).astype(int)
        out["HasBasement"] = (bsmt > 0).astype(int)
        out["HasFireplace"] = (_num(out, "Fireplaces").fillna(0) > 0).astype(int)
        out["HasPool"] = (_num(out, "PoolArea").fillna(0) > 0).astype(int)
        ca = out["CentralAir"].astype(object) if "CentralAir" in out else pd.Series("N", index=out.index)
        out["HasCentralAir"] = (ca == "Y").astype(int)
        out["Has2ndFloor"] = (second > 0).astype(int)
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn wrapper around :func:`add_engineered_features`."""

    def __init__(self, enabled: bool = True, groups: tuple[str, ...] | None = None):
        self.enabled = enabled
        self.groups = groups

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        out = self.transform(X.head(2))
        self.feature_names_out_ = np.asarray(out.columns, dtype=object)
        return self

    def transform(self, X):
        if not self.enabled:
            return X.copy()
        return add_engineered_features(X, list(self.groups) if self.groups else None)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
