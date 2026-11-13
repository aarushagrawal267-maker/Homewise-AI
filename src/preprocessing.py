"""Leakage-safe preprocessing.

Two kinds of steps exist:

* **Stateless domain cleaning** (:class:`DomainCleaner`). Uses only the data
  dictionary, e.g. "NA in BsmtQual means no basement". Nothing is learned from
  data, so it is safe to apply anywhere.
* **Fitted transforms** (imputers, skew correction, scaling, one-hot encoding)
  live inside a scikit-learn ``ColumnTransformer`` that is part of the model
  ``Pipeline``. They are fitted only when the pipeline is fitted, i.e. only on
  the training rows of each split / CV fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG, ID_COL, SEED, TARGET
from .data_dictionary import (
    FEATURE_DESCRIPTIONS,
    NA_MEANS_ABSENT_CATEGORICAL,
    NA_MEANS_ZERO_NUMERIC,
    NUMERIC_CODED_CATEGORICAL,
)

RAW_FEATURES: list[str] = list(FEATURE_DESCRIPTIONS.keys())

QUALITY_SCALE = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
ORDINAL_MAPS: dict[str, dict[str, int]] = {
    **{c: QUALITY_SCALE for c in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                                  "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
    "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
}


def clean_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data-dictionary rules to raw Kaggle-format rows (no fitting)."""
    out = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns]).copy()
    for c in RAW_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    out = out[RAW_FEATURES]

    # Normalise string columns to plain object dtype (pandas 3 uses a string dtype).
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(object).where(out[c].notna(), np.nan)

    for c in NUMERIC_CODED_CATEGORICAL:
        out[c] = out[c].map(lambda v: np.nan if pd.isna(v) else f"C{int(float(v))}").astype(object)

    for c in NA_MEANS_ABSENT_CATEGORICAL:
        out[c] = out[c].fillna("None")
    for c in NA_MEANS_ZERO_NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    # A missing garage year only occurs when there is no garage: use the house year
    # so that age-based features stay meaningful (HasGarage flags the absence).
    out["GarageYrBlt"] = pd.to_numeric(out["GarageYrBlt"], errors="coerce").fillna(
        pd.to_numeric(out["YearBuilt"], errors="coerce"))

    for c, mapping in ORDINAL_MAPS.items():
        out[c] = out[c].map(lambda v, m=mapping: m.get(v, np.nan) if not pd.isna(v) else np.nan).astype(float)
    return out


class DomainCleaner(BaseEstimator, TransformerMixin):
    """Stateless wrapper so cleaning is part of the persisted model pipeline."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return clean_raw_frame(X)


class SkewLog1p(BaseEstimator, TransformerMixin):
    """log1p-transform numeric columns whose training skewness exceeds a threshold.

    Which columns to transform is learned during ``fit`` (training data only).
    Only non-negative columns are eligible.
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        skew = pd.DataFrame(X).skew().to_numpy()
        nonneg = np.nanmin(X, axis=0) >= 0
        self.mask_ = (np.abs(np.nan_to_num(skew)) > self.threshold) & nonneg
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        # Columns were non-negative in training; clip so unseen negatives cannot yield -inf/NaN.
        X[:, self.mask_] = np.log1p(np.clip(X[:, self.mask_], 0, None))
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(kind: str = "tree") -> ColumnTransformer:
    """Column-wise preprocessing.

    ``kind="linear"``: median impute → skew log1p → standard scale; one-hot.
    ``kind="tree"``: median impute only (trees are scale invariant); one-hot.
    """
    if kind == "linear":
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("skew", SkewLog1p(CONFIG["preprocessing"]["skew_threshold"])),
            ("scale", StandardScaler()),
        ])
    else:
        numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, make_column_selector(dtype_include="number")),
         ("cat", categorical, make_column_selector(dtype_exclude="number"))],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def transformed_feature_sources(preprocessor: ColumnTransformer) -> tuple[list[str], list[str]]:
    """Return (transformed column names, source feature for each column).

    One-hot columns map back to their original categorical feature so that
    explanations can be aggregated per property characteristic.
    """
    names: list[str] = []
    sources: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        cols = list(cols)
        if name == "num":
            names += [f"num__{c}" for c in cols]
            sources += cols
        else:
            enc: OneHotEncoder = trans.named_steps["onehot"]
            for col, cats in zip(cols, enc.categories_):
                for cat in cats:
                    names.append(f"cat__{col}_{cat}")
                    sources.append(col)
    return names, sources


def remove_training_outliers(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, list]:
    """Drop the documented partial-sale outliers (GrLivArea > 4000, price < $300k).

    The dataset author (De Cock, 2011) recommends removing houses over 4,000 sq ft.
    We remove only the two whose prices are far below the trend. Training rows only.
    """
    mask = (X["GrLivArea"] > 4000) & (y < 300_000)
    return X.loc[~mask], y.loc[~mask], list(X.index[mask])


def make_train_validation_split(X: pd.DataFrame, y: pd.Series, validation_size: float | None = None,
                                seed: int = SEED):
    """Reproducible split stratified on price deciles so both sets span the market."""
    validation_size = validation_size or CONFIG["split"]["validation_size"]
    strata = pd.qcut(y, q=10, labels=False, duplicates="drop")
    return train_test_split(X, y, test_size=validation_size, random_state=seed, stratify=strata)
