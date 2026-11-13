"""Similar-property finder: nearest historical sales in the dataset.

Properties are compared on standardised structural features plus one-hot
neighborhood and building type. Results are historical Ames sales from the
Kaggle data, not live listings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ID_COL, TARGET
from .feature_engineering import add_engineered_features
from .preprocessing import clean_raw_frame

NUMERIC_KEYS = ["GrLivArea", "TotalSF", "OverallQual", "OverallCond", "HouseAge", "TotalBathrooms",
                "BedroomAbvGr", "GarageCars", "LogLotArea", "TotRmsAbvGrd"]
CATEGORICAL_KEYS = ["Neighborhood", "BldgType"]
# Relative emphasis (applied after standardisation).
WEIGHTS = {"GrLivArea": 1.5, "TotalSF": 1.0, "OverallQual": 1.5, "OverallCond": 0.5, "HouseAge": 1.0,
           "TotalBathrooms": 0.8, "BedroomAbvGr": 0.6, "GarageCars": 0.7, "LogLotArea": 0.6,
           "TotRmsAbvGrd": 0.5, "Neighborhood": 1.0, "BldgType": 0.8}

DISPLAY_COLUMNS = [ID_COL, "Neighborhood", "BldgType", "OverallQual", "GrLivArea", "TotalBsmtSF",
                   "BedroomAbvGr", "FullBath", "GarageCars", "YearBuilt", "LotArea", TARGET]


def similarity_features(raw: pd.DataFrame) -> pd.DataFrame:
    f = add_engineered_features(clean_raw_frame(raw))
    f["LogLotArea"] = np.log1p(pd.to_numeric(f["LotArea"], errors="coerce"))
    return f[NUMERIC_KEYS + CATEGORICAL_KEYS]


@dataclass
class SimilarPropertyFinder:
    dataset: pd.DataFrame  # raw train.csv rows including Id and SalePrice

    def __post_init__(self):
        feats = similarity_features(self.dataset)
        self.pipe = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC_KEYS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_KEYS),
        ])
        Z = self.pipe.fit_transform(feats)
        self.w = self._weight_vector()
        self.Z = Z * self.w
        self.nn = NearestNeighbors(metric="euclidean").fit(self.Z)
        # Typical distance between random pairs, used to scale a 0–100 similarity score.
        rng = np.random.default_rng(0)
        a, b = rng.integers(0, len(self.Z), (2, 2000))
        self.scale = float(np.median(np.linalg.norm(self.Z[a] - self.Z[b], axis=1)))

    def _weight_vector(self) -> np.ndarray:
        w = [WEIGHTS[c] for c in NUMERIC_KEYS]
        enc = self.pipe.named_transformers_["cat"]
        for col, cats in zip(CATEGORICAL_KEYS, enc.categories_):
            # Divide by sqrt(2) so a category mismatch counts like one weighted standard deviation.
            w += [WEIGHTS[col] / np.sqrt(2)] * len(cats)
        return np.asarray(w)

    def transform(self, raw: pd.DataFrame) -> np.ndarray:
        return self.pipe.transform(similarity_features(raw)) * self.w

    def find(self, raw_property: pd.DataFrame, k: int = 8, exclude_id: int | None = None) -> pd.DataFrame:
        if len(raw_property) != 1:
            raise ValueError("Provide exactly one property.")
        k = int(max(1, min(k, len(self.dataset) - 1)))
        dist, idx = self.nn.kneighbors(self.transform(raw_property), n_neighbors=k + 1)
        rows = self.dataset.iloc[idx[0]].copy()
        rows["distance"] = dist[0]
        if exclude_id is not None:
            rows = rows[rows[ID_COL] != exclude_id]
        rows = rows.head(k)
        rows["similarity"] = 100 * np.exp(-rows["distance"] / self.scale)
        cols = [c for c in DISPLAY_COLUMNS if c in rows] + ["distance", "similarity"]
        return rows[cols].reset_index(drop=True)

    def pairwise_similarity(self, raw_a: pd.DataFrame, raw_b: pd.DataFrame) -> float:
        d = float(np.linalg.norm(self.transform(raw_a)[0] - self.transform(raw_b)[0]))
        return float(100 * np.exp(-d / self.scale))

    def distinguishing_features(self, raw_property: pd.DataFrame, neighbours: pd.DataFrame,
                                top: int = 5) -> pd.DataFrame:
        """Numeric characteristics where the property differs most from its comparables."""
        me = similarity_features(raw_property).iloc[0]
        ids = neighbours[ID_COL].tolist()
        comps = similarity_features(self.dataset[self.dataset[ID_COL].isin(ids)])
        scaler = self.pipe.named_transformers_["num"].named_steps["sc"]
        sd = pd.Series(scaler.scale_, index=NUMERIC_KEYS)
        diff = (me[NUMERIC_KEYS].astype(float) - comps[NUMERIC_KEYS].astype(float).median()) / sd
        out = pd.DataFrame({"feature": NUMERIC_KEYS, "this_property": me[NUMERIC_KEYS].astype(float).to_numpy(),
                            "comparables_median": comps[NUMERIC_KEYS].astype(float).median().to_numpy(),
                            "difference_in_sd": diff.to_numpy()})
        out = out[out.feature != "LogLotArea"]
        return out.reindex(out["difference_in_sd"].abs().sort_values(ascending=False).index).head(top)
