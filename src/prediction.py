"""Property valuation engine.

The engine turns a small set of user-facing inputs into a complete Kaggle-format
property record, validates it, and runs it through the *persisted* model
pipeline (the exact preprocessing used during training). Predictions are
returned in US dollars together with a conformal prediction interval.

Fields the user does not provide are taken from a *reference profile*: the
median (numeric) or most frequent value (categorical) of the training data, or
the actual record when the user starts from an existing dataset property.
Related fields are kept consistent: changing living area rescales the floor
areas, removing the garage clears garage attributes, and so on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import PATHS
from .preprocessing import RAW_FEATURES
from .uncertainty import PredictionInterval, interval_from_quantile


class PropertyInputError(ValueError):
    """Raised when user input fails validation. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid property input: " + "; ".join(errors))


class ModelNotTrainedError(FileNotFoundError):
    pass


class ModelLoadError(RuntimeError):
    """Raised when the model bundle exists but cannot be loaded (corrupt, Git LFS pointer, wrong versions)."""


BUNDLE_KEYS = ("model", "model_label", "template", "category_options", "numeric_ranges",
               "conformal_quantiles", "defaults", "background")


# name -> (type, hard_min, hard_max)
FIELD_SPECS: dict[str, tuple[str, float, float]] = {
    "OverallQual": ("int", 1, 10),
    "OverallCond": ("int", 1, 10),
    "GrLivArea": ("float", 300, 8000),
    "LotArea": ("float", 1000, 250000),
    "YearBuilt": ("int", 1850, 2010),
    "YearRemodAdd": ("int", 1850, 2010),
    "YrSold": ("int", 2006, 2010),
    "BedroomAbvGr": ("int", 0, 10),
    "FullBath": ("int", 0, 5),
    "HalfBath": ("int", 0, 3),
    "KitchenAbvGr": ("int", 0, 3),
    "TotRmsAbvGrd": ("int", 1, 16),
    "TotalBsmtSF": ("float", 0, 7000),
    "GarageCars": ("int", 0, 5),
    "GarageArea": ("float", 0, 1500),
    "Fireplaces": ("int", 0, 4),
    "PoolArea": ("float", 0, 1000),
}
CATEGORICAL_INPUTS = ["Neighborhood", "BldgType", "HouseStyle", "KitchenQual", "ExterQual",
                      "CentralAir", "SaleCondition", "MSZoning"]
BOOLEAN_INPUTS = ["HasPool", "HasCentralAir", "HasFireplace"]

# Core fields a valuation form must provide.
REQUIRED_FIELDS = ["OverallQual", "GrLivArea", "YearBuilt", "Neighborhood"]


def validate_inputs(inputs: dict, category_options: dict[str, list[str]],
                    required: list[str] | None = None) -> dict:
    """Validate and coerce user inputs. Raises PropertyInputError listing all problems."""
    errors: list[str] = []
    clean: dict = {}
    for f in (REQUIRED_FIELDS if required is None else required):
        if f not in inputs or inputs[f] is None or (isinstance(inputs[f], float) and np.isnan(inputs[f])):
            errors.append(f"'{f}' is required.")
    for k, v in inputs.items():
        if v is None:
            continue
        if k in FIELD_SPECS:
            kind, lo, hi = FIELD_SPECS[k]
            try:
                num = float(v)
            except (TypeError, ValueError):
                errors.append(f"'{k}' must be a number (got {v!r}).")
                continue
            if not np.isfinite(num):
                errors.append(f"'{k}' must be finite.")
                continue
            if kind == "int" and num != int(num):
                errors.append(f"'{k}' must be a whole number.")
                continue
            if not lo <= num <= hi:
                errors.append(f"'{k}' must be between {lo:g} and {hi:g} (got {num:g}).")
                continue
            clean[k] = int(num) if kind == "int" else num
        elif k in BOOLEAN_INPUTS:
            clean[k] = bool(v)
        elif k in category_options or k in CATEGORICAL_INPUTS:
            opts = category_options.get(k)
            if opts is not None and str(v) not in opts:
                errors.append(f"'{k}' must be one of the dataset's values; '{v}' is unknown.")
                continue
            clean[k] = str(v)
        elif k in RAW_FEATURES:
            clean[k] = v
        else:
            errors.append(f"Unknown field '{k}'.")
    yb, yr, ys = clean.get("YearBuilt"), clean.get("YearRemodAdd"), clean.get("YrSold")
    if yb is not None and yr is not None and yr < yb:
        errors.append("Year remodeled cannot be earlier than year built.")
    if yb is not None and ys is not None and yb > ys:
        errors.append("Year built cannot be after the sale year.")
    if yr is not None and ys is not None and yr > ys:
        errors.append("Year remodeled cannot be after the sale year.")
    if "GarageCars" in clean and clean["GarageCars"] == 0 and clean.get("GarageArea", 0) > 0:
        errors.append("Garage area must be 0 when garage capacity is 0.")
    if "GarageCars" in clean and clean["GarageCars"] > 0 and clean.get("GarageArea", 1) == 0:
        errors.append("Garage area must be positive when the garage holds cars.")
    if errors:
        raise PropertyInputError(errors)
    return clean


def _isnull(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA


def apply_inputs(base: pd.Series, inputs: dict, defaults: dict) -> pd.Series:
    """Overlay validated inputs on a raw property record, keeping related fields consistent."""
    row = base.astype(object).copy()
    inp = dict(inputs)

    if "HasCentralAir" in inp:
        inp["CentralAir"] = "Y" if inp.pop("HasCentralAir") else "N"
    if "HasFireplace" in inp:
        has = inp.pop("HasFireplace")
        if "Fireplaces" not in inp:
            cur = 0 if _isnull(row.get("Fireplaces")) else int(row["Fireplaces"])
            inp["Fireplaces"] = max(cur, 1) if has else 0
    if "HasPool" in inp:
        has = inp.pop("HasPool")
        if "PoolArea" not in inp:
            cur = 0 if _isnull(row.get("PoolArea")) else float(row["PoolArea"])
            inp["PoolArea"] = (cur if cur > 0 else defaults["pool_area"]) if has else 0.0

    old_built = row.get("YearBuilt")
    if "HouseStyle" in inp:
        style = str(inp["HouseStyle"])
        f1 = 0 if _isnull(row.get("1stFlrSF")) else float(row["1stFlrSF"])
        f2 = 0 if _isnull(row.get("2ndFlrSF")) else float(row["2ndFlrSF"])
        total = f1 + f2
        if style in ("1Story", "SFoyer", "SLvl") and f2 > 0:
            row["1stFlrSF"], row["2ndFlrSF"] = total, 0
        elif style.startswith(("2", "1.5")) and f2 == 0 and total > 0:
            share = 0.45 if style.startswith("2") else 0.3
            row["2ndFlrSF"] = round(total * share)
            row["1stFlrSF"] = total - row["2ndFlrSF"]
    if "GrLivArea" in inp:
        new = float(inp["GrLivArea"])
        f1 = 0 if _isnull(row.get("1stFlrSF")) else float(row["1stFlrSF"])
        f2 = 0 if _isnull(row.get("2ndFlrSF")) else float(row["2ndFlrSF"])
        low = 0 if _isnull(row.get("LowQualFinSF")) else float(row["LowQualFinSF"])
        above = max(new - low, 1.0)
        if f1 + f2 > 0:
            ratio = f1 / (f1 + f2)
        else:
            ratio = 1.0
        row["1stFlrSF"] = round(above * ratio)
        row["2ndFlrSF"] = round(above - row["1stFlrSF"])
        row["GrLivArea"] = row["1stFlrSF"] + row["2ndFlrSF"] + low
        inp.pop("GrLivArea")

    if "TotalBsmtSF" in inp:
        new = float(inp.pop("TotalBsmtSF"))
        old = 0 if _isnull(row.get("TotalBsmtSF")) else float(row["TotalBsmtSF"])
        if new == 0:
            for c in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
                row[c] = np.nan
            for c in ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"):
                row[c] = 0
        elif old == 0:
            row.update(pd.Series({"BsmtQual": "TA", "BsmtCond": "TA", "BsmtExposure": "No",
                                  "BsmtFinType1": "Unf", "BsmtFinType2": "Unf",
                                  "BsmtFinSF1": 0, "BsmtFinSF2": 0, "BsmtUnfSF": new}))
        else:
            s = new / old
            for c in ("BsmtFinSF1", "BsmtFinSF2"):
                row[c] = round(float(row[c] or 0) * s)
            row["BsmtUnfSF"] = max(new - row["BsmtFinSF1"] - row["BsmtFinSF2"], 0)
        row["TotalBsmtSF"] = new

    if "GarageCars" in inp:
        cars = int(inp.pop("GarageCars"))
        had = not _isnull(row.get("GarageType"))
        if cars == 0:
            for c in ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "GarageYrBlt"):
                row[c] = np.nan
            row["GarageArea"] = 0
        else:
            if not had:
                row.update(pd.Series({"GarageType": "Attchd", "GarageFinish": "Unf", "GarageQual": "TA",
                                      "GarageCond": "TA",
                                      "GarageYrBlt": inp.get("YearBuilt", row.get("YearBuilt"))}))
            if "GarageArea" not in inp:
                old_cars = 0 if _isnull(row.get("GarageCars")) else int(row["GarageCars"])
                old_area = 0 if _isnull(row.get("GarageArea")) else float(row["GarageArea"])
                per = old_area / old_cars if old_cars > 0 and old_area > 0 else defaults["garage_area_per_car"]
                row["GarageArea"] = round(per * cars)
        row["GarageCars"] = cars

    if "Fireplaces" in inp:
        n = int(inp.pop("Fireplaces"))
        row["FireplaceQu"] = np.nan if n == 0 else (row["FireplaceQu"] if not _isnull(row.get("FireplaceQu")) else "TA")
        row["Fireplaces"] = n

    if "PoolArea" in inp:
        a = float(inp.pop("PoolArea"))
        row["PoolQC"] = np.nan if a == 0 else (row["PoolQC"] if not _isnull(row.get("PoolQC")) else defaults["pool_qc"])
        row["PoolArea"] = a

    for k, v in inp.items():
        row[k] = v

    # Keep year fields consistent.
    if "YearBuilt" in inputs:
        yb = int(row["YearBuilt"])
        if "YearRemodAdd" not in inputs and (_isnull(row.get("YearRemodAdd")) or row["YearRemodAdd"] < yb
                                             or row["YearRemodAdd"] == old_built):
            row["YearRemodAdd"] = yb
        if not _isnull(row.get("GarageYrBlt")) and (row["GarageYrBlt"] == old_built or row["GarageYrBlt"] < yb):
            row["GarageYrBlt"] = yb
    return row


def set_feature(df: pd.DataFrame, feature: str, value) -> pd.DataFrame:
    """Vectorised, consistency-preserving version of :func:`apply_inputs` for one field.

    Used by partial-dependence curves and the 3-D price surface so that, e.g.,
    changing living area also rescales the first/second floor areas.
    """
    out = df.copy()
    if feature == "GrLivArea":
        f1 = pd.to_numeric(out["1stFlrSF"], errors="coerce").fillna(0)
        f2 = pd.to_numeric(out["2ndFlrSF"], errors="coerce").fillna(0)
        low = pd.to_numeric(out["LowQualFinSF"], errors="coerce").fillna(0)
        ratio = (f1 / (f1 + f2).where(f1 + f2 > 0)).fillna(1.0)
        above = (float(value) - low).clip(lower=1)
        out["1stFlrSF"] = (above * ratio).round()
        out["2ndFlrSF"] = (above - out["1stFlrSF"]).round()
        out["GrLivArea"] = out["1stFlrSF"] + out["2ndFlrSF"] + low
    elif feature == "TotalBsmtSF":
        old = pd.to_numeric(out["TotalBsmtSF"], errors="coerce").fillna(0)
        v = float(value)
        if v == 0:
            for c in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
                out[c] = np.nan
            for c in ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"):
                out[c] = 0.0
        else:
            had = old > 0
            s = (v / old.where(had)).fillna(0)
            for c in ("BsmtFinSF1", "BsmtFinSF2"):
                out[c] = (pd.to_numeric(out[c], errors="coerce").fillna(0) * s).round()
            out["BsmtUnfSF"] = (v - out["BsmtFinSF1"] - out["BsmtFinSF2"]).clip(lower=0)
            for c, d in (("BsmtQual", "TA"), ("BsmtCond", "TA"), ("BsmtExposure", "No"),
                         ("BsmtFinType1", "Unf"), ("BsmtFinType2", "Unf")):
                out.loc[~had, c] = d
        out["TotalBsmtSF"] = v
    elif feature == "GarageCars":
        cars = int(value)
        old_c = pd.to_numeric(out["GarageCars"], errors="coerce").fillna(0)
        old_a = pd.to_numeric(out["GarageArea"], errors="coerce").fillna(0)
        per = (old_a / old_c.where(old_c > 0)).fillna(260.0)
        if cars == 0:
            for c in ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "GarageYrBlt"):
                out[c] = np.nan
            out["GarageArea"] = 0.0
        else:
            had = out["GarageType"].notna()
            for c, d in (("GarageType", "Attchd"), ("GarageFinish", "Unf"), ("GarageQual", "TA"),
                         ("GarageCond", "TA")):
                out.loc[~had, c] = d
            out.loc[~had, "GarageYrBlt"] = out.loc[~had, "YearBuilt"]
            out["GarageArea"] = (per * cars).round()
        out["GarageCars"] = cars
    elif feature == "YearBuilt":
        v = int(value)
        out["YearBuilt"] = v
        rem = pd.to_numeric(out["YearRemodAdd"], errors="coerce")
        out["YearRemodAdd"] = rem.where(rem >= v, v)
        g = pd.to_numeric(out["GarageYrBlt"], errors="coerce")
        out["GarageYrBlt"] = g.where(g.isna() | (g >= v), v)
    else:
        out[feature] = value
    return out


@dataclass
class ValuationResult:
    estimate: float
    interval: PredictionInterval
    record: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


class PredictionEngine:
    """Loads the persisted model bundle and serves valuations."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.template: pd.Series = bundle["template"]
        self.category_options: dict[str, list[str]] = bundle["category_options"]
        self.ranges: dict[str, dict] = bundle["numeric_ranges"]
        self.quantiles: dict[float, float] = bundle["conformal_quantiles"]
        self.defaults = bundle["defaults"]

    @classmethod
    def load(cls, path=None) -> "PredictionEngine":
        path = path or PATHS.model_bundle
        if not path.exists():
            raise ModelNotTrainedError(
                f"No trained model found at {path}. Run: python scripts/train_all.py")
        with open(path, "rb") as fh:
            head = fh.read(64)
        if head.startswith(b"version https://git-lfs"):
            raise ModelLoadError(
                f"{path.name} is a Git LFS pointer, not the model itself. Run 'git lfs pull' to fetch the real "
                "file, or retrain with: python scripts/train_all.py")
        try:
            bundle = joblib.load(path)
        except Exception as err:
            raise ModelLoadError(
                f"Could not load {path.name} ({type(err).__name__}: {err}). The file may be corrupt or was saved "
                "with different library versions. Install the pinned versions with "
                "'pip install -r requirements.txt', or retrain with: python scripts/train_all.py") from err
        missing = [k for k in BUNDLE_KEYS if not isinstance(bundle, dict) or k not in bundle]
        if missing:
            raise ModelLoadError(
                f"{path.name} is not a HomeWise model bundle (missing: {', '.join(missing)}). "
                "Retrain with: python scripts/train_all.py")
        return cls(bundle)

    @property
    def model_label(self) -> str:
        return self.bundle["model_label"]

    def predict_frame(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=RAW_FEATURES)
        pred = np.asarray(self.model.predict(X), dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError("Model returned a non-finite prediction.")
        return pred

    def build_record(self, inputs: dict, base: pd.Series | None = None, validate: bool = True,
                     required: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
        clean = validate_inputs(inputs, self.category_options, required) if validate else inputs
        base = (self.template if base is None else base).reindex(RAW_FEATURES)
        row = apply_inputs(base, clean, self.defaults)
        return pd.DataFrame([row.to_numpy()], columns=RAW_FEATURES), self.extrapolation_warnings(row)

    def extrapolation_warnings(self, row: pd.Series) -> list[str]:
        out = []
        for k, r in self.ranges.items():
            v = row.get(k)
            if _isnull(v):
                continue
            if v < r["min"] or v > r["max"]:
                out.append(f"{k} = {v:,.0f} is outside the training data range "
                           f"({r['min']:,.0f}–{r['max']:,.0f}); the estimate is an extrapolation.")
        return out

    def interval(self, estimate: float, level: float | None = None) -> PredictionInterval:
        level = level or self.bundle.get("default_level", 0.9)
        q = self.quantiles[min(self.quantiles, key=lambda k: abs(k - level))]
        return interval_from_quantile(estimate, q, level)

    def value(self, inputs: dict, base: pd.Series | None = None, level: float | None = None,
              required: list[str] | None = None) -> ValuationResult:
        record, warns = self.build_record(inputs, base, required=required)
        est = float(self.predict_frame(record)[0])
        return ValuationResult(est, self.interval(est, level), record, warns)


@lru_cache(maxsize=1)
def load_engine() -> PredictionEngine:
    return PredictionEngine.load()
