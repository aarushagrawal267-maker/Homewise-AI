"""Side-by-side comparison of two or more property profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_engineering import add_engineered_features
from .preprocessing import clean_raw_frame

COMPARE_FIELDS = [
    ("Estimated price", "estimate"),
    ("Recorded sale price", "SalePrice"),
    ("Neighborhood", "Neighborhood"),
    ("Building type", "BldgType"),
    ("Overall quality", "OverallQual"),
    ("Overall condition", "OverallCond"),
    ("Living area (sq ft)", "GrLivArea"),
    ("Total area incl. basement (sq ft)", "TotalSF"),
    ("Lot area (sq ft)", "LotArea"),
    ("Basement area (sq ft)", "TotalBsmtSF"),
    ("Bedrooms", "BedroomAbvGr"),
    ("Bathrooms (total)", "TotalBathrooms"),
    ("Garage capacity (cars)", "GarageCars"),
    ("Garage area (sq ft)", "GarageArea"),
    ("Year built", "YearBuilt"),
    ("Age at sale (years)", "HouseAge"),
    ("Years since remodel", "YearsSinceRemodel"),
    ("Fireplaces", "Fireplaces"),
    ("Central air", "CentralAir"),
]

RADAR_FEATURES = ["OverallQual", "GrLivArea", "TotalSF", "TotalBathrooms", "GarageCars", "LotArea", "Newness"]
RADAR_LABELS = {"OverallQual": "Quality", "GrLivArea": "Living area", "TotalSF": "Total area",
                "TotalBathrooms": "Bathrooms", "GarageCars": "Garage", "LotArea": "Lot size",
                "Newness": "Newness"}


def enrich(raw: pd.DataFrame) -> pd.DataFrame:
    """Cleaned + engineered view of raw records (keeps raw categorical labels)."""
    eng = add_engineered_features(clean_raw_frame(raw))
    eng["CentralAir"] = raw["CentralAir"].to_numpy()
    eng["Newness"] = -eng["HouseAge"]
    return eng


def comparison_table(raw_profiles: list[pd.DataFrame], names: list[str], estimates: list[float],
                     sale_prices: list[float | None] | None = None) -> pd.DataFrame:
    if not (len(raw_profiles) == len(names) == len(estimates)) or len(raw_profiles) < 2:
        raise ValueError("Provide at least two profiles with matching names and estimates.")
    raw = pd.concat(raw_profiles, ignore_index=True)
    eng = enrich(raw)
    eng["estimate"] = estimates
    eng["SalePrice"] = sale_prices if sale_prices is not None else [None] * len(names)
    table = {}
    for label, col in COMPARE_FIELDS:
        table[label] = eng[col].tolist() if col in eng else [None] * len(names)
    return pd.DataFrame(table, index=names).T


def price_differences(names: list[str], estimates: list[float]) -> pd.DataFrame:
    base_name, base = names[0], estimates[0]
    rows = []
    for n, e in zip(names[1:], estimates[1:]):
        rows.append({"property": n, "versus": base_name, "difference": e - base,
                     "percent": (e - base) / base * 100})
    return pd.DataFrame(rows)


def radar_scores(raw_profiles: list[pd.DataFrame], reference: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank (0–100) of each property within the reference dataset, per feature."""
    ref = enrich(reference)
    eng = enrich(pd.concat(raw_profiles, ignore_index=True))
    out = {}
    for f in RADAR_FEATURES:
        r = np.sort(ref[f].astype(float).dropna().to_numpy())
        out[RADAR_LABELS[f]] = [float(np.searchsorted(r, v, side="right") / len(r) * 100)
                                for v in eng[f].astype(float)]
    return pd.DataFrame(out)


def explanation_difference(contrib_a: pd.Series, contrib_b: pd.Series, top: int = 8) -> pd.DataFrame:
    """Which features account for the model's price gap between two properties (log scale)."""
    idx = contrib_a.index.union(contrib_b.index)
    d = contrib_b.reindex(idx, fill_value=0) - contrib_a.reindex(idx, fill_value=0)
    d = d.reindex(d.abs().sort_values(ascending=False).index).head(top)
    return pd.DataFrame({"feature": d.index, "log_difference": d.to_numpy(),
                         "pct_effect": (np.exp(d.to_numpy()) - 1) * 100})
