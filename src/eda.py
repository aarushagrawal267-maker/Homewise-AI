"""Exploratory data analysis: statistics, automatic insights and static figures.

Every number and insight is computed from the loaded data at run time; nothing
is hard-coded. Insights describe associations in historical Ames sales and do
not establish causation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from .config import ID_COL, PATHS, TARGET
from .data_validation import identify_feature_types, missing_summary


def dataset_overview(train: pd.DataFrame, test: pd.DataFrame | None = None) -> dict:
    num, cat = identify_feature_types(train)
    feats = [c for c in train.columns if c not in (ID_COL, TARGET)]
    return {
        "train_records": int(len(train)),
        "test_records": int(len(test)) if test is not None else None,
        "n_features": len(feats),
        "n_numerical": len(num),
        "n_categorical": len(cat),
        "missing_cells_train": int(train[feats].isna().sum().sum()),
        "columns_with_missing_train": int((train[feats].isna().sum() > 0).sum()),
        "duplicate_records_train": int(train.drop(columns=[ID_COL]).duplicated().sum()),
    }


def target_statistics(y: pd.Series) -> dict:
    q = y.quantile([0.25, 0.5, 0.75])
    return {
        "mean": float(y.mean()), "median": float(y.median()), "min": float(y.min()),
        "max": float(y.max()), "std": float(y.std()), "q1": float(q[0.25]), "q3": float(q[0.75]),
        "iqr": float(q[0.75] - q[0.25]), "skewness": float(y.skew()), "kurtosis": float(y.kurt()),
        "log_skewness": float(np.log1p(y).skew()),
    }


def numeric_correlations(train: pd.DataFrame, method: str = "pearson") -> pd.Series:
    num, _ = identify_feature_types(train)
    corr = train[num + [TARGET]].corr(method=method, numeric_only=True)[TARGET].drop(TARGET)
    return corr.sort_values(key=np.abs, ascending=False)


def group_price_stats(train: pd.DataFrame, col: str) -> pd.DataFrame:
    g = train.groupby(col, observed=True)[TARGET]
    out = pd.DataFrame({"count": g.size(), "median": g.median(), "mean": g.mean(),
                        "q1": g.quantile(0.25), "q3": g.quantile(0.75)})
    return out.sort_values("median", ascending=False)


def detect_outliers(train: pd.DataFrame) -> dict:
    y = train[TARGET]
    q1, q3 = y.quantile([0.25, 0.75])
    iqr = q3 - q1
    hi, lo = q3 + 1.5 * iqr, q1 - 1.5 * iqr
    log_z = np.abs(stats.zscore(np.log1p(y)))
    partial = train[(train["GrLivArea"] > 4000) & (y < 300_000)]
    return {
        "iqr_upper_fence": float(hi), "iqr_lower_fence": float(lo),
        "n_above_fence": int((y > hi).sum()), "n_below_fence": int((y < lo).sum()),
        "n_log_z_gt_3": int((log_z > 3).sum()),
        "large_low_price_ids": partial[ID_COL].tolist(),
        "large_low_price_rows": partial[[ID_COL, "GrLivArea", TARGET, "SaleCondition"]].to_dict("records"),
    }


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}"


def automatic_insights(train: pd.DataFrame) -> list[dict]:
    """Data-driven answers to the key EDA questions."""
    y = train[TARGET]
    out: list[dict] = []

    # Overall quality
    rho_q = stats.spearmanr(train["OverallQual"], y).statistic
    q = train.groupby("OverallQual")[TARGET].median()
    out.append({
        "question": "Does overall quality have a strong relationship with sale price?",
        "answer": (f"Yes. Spearman correlation is {rho_q:.2f}. Median price rises from "
                   f"{_fmt_money(q.iloc[0])} at quality {q.index[0]} to {_fmt_money(q.iloc[-1])} at "
                   f"quality {q.index[-1]} ({q.iloc[-1] / q.iloc[0]:.1f}× higher)."),
        "metric": float(rho_q)})

    # Living area
    r = stats.pearsonr(train["GrLivArea"], y).statistic
    slope = np.polyfit(train["GrLivArea"], y, 1)[0]
    out.append({
        "question": "How does living area relate to historical sale prices?",
        "answer": (f"Positively: Pearson r = {r:.2f}. A simple linear fit implies about "
                   f"{_fmt_money(slope * 100)} per extra 100 sq ft, before accounting for any other factor."),
        "metric": float(r)})

    # Newer houses
    rho_y = stats.spearmanr(train["YearBuilt"], y).statistic
    new_med = y[train["YearBuilt"] >= 2000].median()
    old_med = y[train["YearBuilt"] < 1950].median()
    out.append({
        "question": "Are newer houses generally more expensive in this dataset?",
        "answer": (f"Generally yes (Spearman {rho_y:.2f}). Homes built 2000 or later have a median of "
                   f"{_fmt_money(new_med)} versus {_fmt_money(old_med)} for homes built before 1950. "
                   "Newer homes also tend to be larger and higher quality, so age alone does not explain the gap."),
        "metric": float(rho_y)})

    # Garage
    g = train.groupby("GarageCars")[TARGET].agg(["median", "size"])
    g = g[g["size"] >= 5]
    rho_g = stats.spearmanr(train["GarageCars"], y).statistic
    parts = ", ".join(f"{int(k)} cars: {_fmt_money(v)} (n={int(n)})" for k, (v, n) in g.iterrows())
    out.append({
        "question": "How does garage capacity relate to property value?",
        "answer": f"Spearman {rho_g:.2f}. Median price by capacity — {parts}.",
        "metric": float(rho_g)})

    # Neighborhoods
    nb = group_price_stats(train, "Neighborhood")
    nb = nb[nb["count"] >= 10]
    top = ", ".join(f"{k} ({_fmt_money(v)})" for k, v in nb["median"].head(3).items())
    bottom = ", ".join(f"{k} ({_fmt_money(v)})" for k, v in nb["median"].tail(3).items())
    out.append({
        "question": "Which neighborhoods have higher median historical sale prices?",
        "answer": f"Highest medians (≥10 sales): {top}. Lowest: {bottom}.",
        "metric": float(nb["median"].iloc[0] / nb["median"].iloc[-1])})

    # Strongest correlations
    corr = numeric_correlations(train).head(5)
    out.append({
        "question": "Which property characteristics show the strongest correlations with SalePrice?",
        "answer": "Top numeric Pearson correlations: " + ", ".join(f"{k} ({v:.2f})" for k, v in corr.items()) + ".",
        "metric": float(corr.iloc[0])})

    out.append({
        "question": "What about skewness of the target?",
        "answer": (f"SalePrice is right-skewed (skewness {y.skew():.2f}); after log1p the skewness is "
                   f"{np.log1p(y).skew():.2f}, which is why models are trained on log price."),
        "metric": float(y.skew())})
    return out


def build_eda_summary(train: pd.DataFrame, test: pd.DataFrame | None = None) -> dict:
    corr = numeric_correlations(train)
    return {
        "overview": dataset_overview(train, test),
        "target": target_statistics(train[TARGET]),
        "top_correlations": {k: float(v) for k, v in corr.head(15).items()},
        "missing": missing_summary(train.drop(columns=[TARGET])).reset_index(names="column").to_dict("records"),
        "outliers": detect_outliers(train),
        "insights": automatic_insights(train),
        "neighborhood_medians": group_price_stats(train, "Neighborhood")["median"].round(0).to_dict(),
    }


def save_eda_summary(summary: dict) -> None:
    PATHS.reports.mkdir(parents=True, exist_ok=True)
    with open(PATHS.reports / "eda_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)


def generate_static_figures(train: pd.DataFrame) -> list[str]:
    """Save the core EDA charts as PNG files (for the report and README)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    y = train[TARGET]
    saved = []

    def save(fig, name):
        p = PATHS.figures / name
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(str(p.relative_to(PATHS.root)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(y, bins=50, ax=ax[0], color="#2a9d8f")
    ax[0].set_title("SalePrice distribution")
    sns.histplot(np.log1p(y), bins=50, ax=ax[1], color="#3d5a80")
    ax[1].set_title("log(1 + SalePrice) distribution")
    save(fig, "01_saleprice_distribution.png")

    num, _ = identify_feature_types(train)
    top = numeric_correlations(train).head(14).index.tolist() + [TARGET]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(train[top].corr(), cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap: top features vs SalePrice")
    save(fig, "02_correlation_heatmap.png")

    pairs = [("GrLivArea", "03_living_area_vs_price.png"), ("TotalBsmtSF", "04_basement_vs_price.png"),
             ("YearBuilt", "05_year_built_vs_price.png")]
    for col, name in pairs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(train[col], y, s=10, alpha=0.5, color="#2a9d8f")
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{col} vs SalePrice")
        save(fig, name)

    for col, name in [("OverallQual", "06_quality_vs_price.png"), ("GarageCars", "07_garage_vs_price.png"),
                      ("BedroomAbvGr", "08_bedrooms_vs_price.png"), ("FullBath", "09_bathrooms_vs_price.png")]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(x=train[col], y=y, ax=ax, color="#98c1d9")
        ax.set_title(f"{col} vs SalePrice")
        save(fig, name)

    order = train.groupby("Neighborhood")[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(x="Neighborhood", y=TARGET, data=train, order=order, ax=ax, color="#98c1d9")
    ax.tick_params(axis="x", rotation=70)
    ax.set_title("Neighborhood vs SalePrice (sorted by median)")
    save(fig, "10_neighborhood_vs_price.png")

    miss = missing_summary(train.drop(columns=[TARGET]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(miss.index[::-1], miss["percent"][::-1], color="#e76f51")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column (training data)")
    save(fig, "11_missing_values.png")

    cols = ["LotArea", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "GarageArea", "YearBuilt"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for a, c in zip(axes.ravel(), cols):
        sns.histplot(train[c], bins=40, ax=a, color="#3d5a80")
        a.set_title(f"{c} (skew {train[c].skew():.2f})")
    save(fig, "12_numeric_distributions.png")

    cats = ["MSZoning", "BldgType", "HouseStyle", "SaleCondition", "Foundation", "KitchenQual"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for a, c in zip(axes.ravel(), cats):
        vc = train[c].value_counts()
        a.bar(vc.index.astype(str), vc.values, color="#2a9d8f")
        a.set_title(c)
        a.tick_params(axis="x", rotation=45)
    save(fig, "13_categorical_frequencies.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    big = (train["GrLivArea"] > 4000) & (y < 300_000)
    ax.scatter(train.loc[~big, "GrLivArea"], y[~big], s=10, alpha=0.4, color="#3d5a80", label="other sales")
    ax.scatter(train.loc[big, "GrLivArea"], y[big], s=50, color="#e63946", label="partial-sale outliers")
    ax.legend()
    ax.set_title("Outlier analysis: GrLivArea vs SalePrice")
    save(fig, "14_outliers.png")
    return saved
