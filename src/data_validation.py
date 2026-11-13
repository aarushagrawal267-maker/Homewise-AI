"""Structural and semantic validation of the raw Kaggle CSV files."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .config import ID_COL, TARGET
from .data_dictionary import FEATURE_DESCRIPTIONS, NUMERIC_CODED_CATEGORICAL

EXPECTED_FEATURES: list[str] = list(FEATURE_DESCRIPTIONS.keys())  # 79 features


class DataValidationError(ValueError):
    """Raised when a dataset fails a blocking validation check."""


@dataclass
class ValidationReport:
    name: str
    n_rows: int
    n_columns: int
    has_target: bool
    missing_expected_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    duplicate_ids: int = 0
    missing_by_column: dict[str, int] = field(default_factory=dict)
    total_missing_cells: int = 0
    numerical_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    numeric_coded_categorical: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    target_summary: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def identify_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into numerical and categorical.

    Numerically coded categories (MSSubClass) are classed as categorical; the Id
    and target columns are excluded.
    """
    cols = [c for c in df.columns if c not in (ID_COL, TARGET)]
    numerical, categorical = [], []
    for c in cols:
        if c in NUMERIC_CODED_CATEGORICAL or not pd.api.types.is_numeric_dtype(df[c]):
            categorical.append(c)
        else:
            numerical.append(c)
    return numerical, categorical


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return pd.DataFrame({
        "missing": miss,
        "percent": (miss / len(df) * 100).round(2),
    })


def validate_target(y: pd.Series) -> tuple[dict[str, float], list[str]]:
    errors = []
    if not pd.api.types.is_numeric_dtype(y):
        errors.append(f"{TARGET} must be numeric.")
        return {}, errors
    if y.isna().any():
        errors.append(f"{TARGET} contains {int(y.isna().sum())} missing values.")
    if (y <= 0).any():
        errors.append(f"{TARGET} contains non-positive values; log transform impossible.")
    if not np.isfinite(y.dropna()).all():
        errors.append(f"{TARGET} contains non-finite values.")
    summary = {
        "count": float(y.count()), "mean": float(y.mean()), "median": float(y.median()),
        "min": float(y.min()), "max": float(y.max()), "std": float(y.std()),
        "skew": float(y.skew()),
    }
    return summary, errors


def validate_dataset(df: pd.DataFrame, name: str, expect_target: bool) -> ValidationReport:
    numerical, categorical = identify_feature_types(df)
    report = ValidationReport(
        name=name,
        n_rows=len(df),
        n_columns=df.shape[1],
        has_target=TARGET in df.columns,
        numerical_features=numerical,
        categorical_features=categorical,
        numeric_coded_categorical=[c for c in NUMERIC_CODED_CATEGORICAL if c in df.columns],
        dtypes={c: str(t) for c, t in df.dtypes.items()},
    )
    if len(df) == 0:
        report.errors.append("Dataset is empty.")
        return report

    present = set(df.columns)
    report.missing_expected_columns = [c for c in EXPECTED_FEATURES if c not in present]
    allowed = set(EXPECTED_FEATURES) | {ID_COL, TARGET}
    report.unexpected_columns = sorted(present - allowed)
    if report.missing_expected_columns:
        report.errors.append(f"Missing expected columns: {report.missing_expected_columns}")
    if report.unexpected_columns:
        report.warnings.append(f"Unexpected columns: {report.unexpected_columns}")
    if ID_COL not in present:
        report.errors.append(f"Missing identifier column '{ID_COL}'.")
    else:
        report.duplicate_ids = int(df[ID_COL].duplicated().sum())
        if report.duplicate_ids:
            report.errors.append(f"{report.duplicate_ids} duplicate {ID_COL} values.")

    feature_cols = [c for c in df.columns if c != ID_COL]
    report.duplicate_rows = int(df.duplicated(subset=feature_cols).sum())
    if report.duplicate_rows:
        report.warnings.append(f"{report.duplicate_rows} duplicate records (ignoring Id).")

    miss = df.isna().sum()
    report.missing_by_column = {c: int(v) for c, v in miss.items() if v > 0}
    report.total_missing_cells = int(miss.sum())

    if expect_target:
        if TARGET not in present:
            report.errors.append(f"Target column '{TARGET}' is missing.")
        else:
            report.target_summary, errs = validate_target(df[TARGET])
            report.errors.extend(errs)
    elif TARGET in present:
        report.warnings.append(f"Unexpected target column '{TARGET}' in unlabelled data.")
    return report


def validate_train_test(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Validate both files and check they share the same feature schema."""
    tr = validate_dataset(train, "train", expect_target=True)
    te = validate_dataset(test, "test", expect_target=False)
    train_feats = [c for c in train.columns if c != TARGET]
    schema_match = train_feats == list(test.columns)
    dtype_mismatch = []
    for c in train_feats:
        if c in test.columns:
            a, b = train[c].dtype, test[c].dtype
            if pd.api.types.is_numeric_dtype(a) != pd.api.types.is_numeric_dtype(b):
                dtype_mismatch.append(c)
    id_overlap = int(len(set(train[ID_COL]) & set(test[ID_COL]))) if ID_COL in test else None
    unseen = {}
    for c in tr.categorical_features:
        if c in test.columns:
            new = set(test[c].dropna().astype(str)) - set(train[c].dropna().astype(str))
            if new:
                unseen[c] = sorted(new)
    return {
        "train": tr.to_dict(),
        "test": te.to_dict(),
        "schema_match": schema_match,
        "dtype_kind_mismatch": dtype_mismatch,
        "id_overlap": id_overlap,
        "test_categories_unseen_in_train": unseen,
        "ok": tr.ok and te.ok and schema_match and id_overlap == 0,
    }
