"""Housing Data Intelligence Dashboard: interactive analytics over the historical dataset."""

import numpy as np
import pandas as pd
import streamlit as st

from src.data_validation import missing_summary
from src.eda import automatic_insights, detect_outliers, numeric_correlations, target_statistics
from src.feature_engineering import add_engineered_features
from src.preprocessing import clean_raw_frame
from src.three_scenes import market_galaxy_html
from src.ui import (
    BLDG_NAMES, NEIGHBORHOOD_NAMES, footer, kpis, md, note, page_header, require_data, section, three,
)
from src.visualization import (
    box_by, correlation_heatmap, count_bar, histogram, market_3d_scatter, median_bar, missing_bar, money,
    scatter_vs_price, style, SERIES,
)

import plotly.graph_objects as go

page_header("Housing analytics", "Ames housing market intelligence",
            "Interactive analysis of 1,460 historical sales (2006–2010). These are historical patterns in one US "
            "city, not current market conditions. Correlation is not causation.")
raw = require_data()


@st.cache_data(show_spinner=False)
def enriched(df: pd.DataFrame) -> pd.DataFrame:
    eng = add_engineered_features(clean_raw_frame(df))
    out = df.copy()
    for c in ["TotalSF", "TotalBathrooms", "HouseAge"]:
        out[c] = eng[c].to_numpy()
    return out


data = enriched(raw)

# ------------------------------------------------------------------ filters (one row) -----
with st.container(border=True):
    f1, f2, f3, f4, f5 = st.columns([1.6, 1, 1.2, 1.2, 1.4])
    nbs = f1.multiselect("Neighborhood", sorted(data["Neighborhood"].unique()),
                         format_func=lambda v: NEIGHBORHOOD_NAMES.get(v, v), placeholder="All neighborhoods")
    qual = f2.slider("Overall quality", 1, 10, (1, 10))
    btypes = f3.multiselect("Property type", sorted(data["BldgType"].unique()),
                            format_func=lambda v: BLDG_NAMES.get(v, v), placeholder="All types")
    yb = f4.slider("Year built", int(data["YearBuilt"].min()), int(data["YearBuilt"].max()),
                   (int(data["YearBuilt"].min()), int(data["YearBuilt"].max())))
    pr = f5.slider("Sale price ($K)", int(data["SalePrice"].min() // 1000), int(data["SalePrice"].max() // 1000 + 1),
                   (int(data["SalePrice"].min() // 1000), int(data["SalePrice"].max() // 1000 + 1)))

m = (data["OverallQual"].between(*qual) & data["YearBuilt"].between(*yb)
     & data["SalePrice"].between(pr[0] * 1000, pr[1] * 1000))
if nbs:
    m &= data["Neighborhood"].isin(nbs)
if btypes:
    m &= data["BldgType"].isin(btypes)
df = data[m]
if len(df) < 5:
    note("Fewer than 5 sales match these filters. Widen them to see analytics.")
    st.stop()

ts = target_statistics(df["SalePrice"])
kpis([("Sales in view", f"{len(df):,}", f"of {len(data):,} total"),
      ("Median price", money(ts["median"]), f"mean {money(ts['mean'])}"),
      ("Price range", f"{money(ts['min'], True)}–{money(ts['max'], True)}", f"IQR {money(ts['iqr'], True)}"),
      ("Avg living area", f"{df['GrLivArea'].mean():,.0f} sq ft", f"lot {df['LotArea'].mean():,.0f} sq ft"),
      ("Median age at sale", f"{df['HouseAge'].median():.0f} yrs", f"skewness {ts['skewness']:.2f}")])
st.write("")

tabs = st.tabs(["Price intelligence", "Property intelligence", "Feature relationships", "3D market",
                "Data quality & EDA", "Automatic insights"])

with tabs[0]:
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(histogram(df["SalePrice"], "Sale price distribution", xprefix="$"), use_container_width=True)
    c2.plotly_chart(histogram(np.log1p(df["SalePrice"]), "log(1 + sale price) distribution", color=SERIES[1]),
                    use_container_width=True)
    nbd = df.assign(Neighborhood=df["Neighborhood"].map(lambda v: NEIGHBORHOOD_NAMES.get(v, v)))
    st.plotly_chart(median_bar(nbd, "Neighborhood", "Median sale price by neighborhood", min_count=1), use_container_width=True)
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(box_by(df, "OverallQual", "Sale price by overall quality"), use_container_width=True)
    btd = df.assign(BldgType=df["BldgType"].map(lambda v: BLDG_NAMES.get(v, v)))
    c2.plotly_chart(median_bar(btd, "BldgType", "Median price by property type", horizontal=False, height=380),
                    use_container_width=True)

with tabs[1]:
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.plotly_chart(histogram(df["HouseAge"], "Age at sale (years)", nbins=40, color=SERIES[3]), use_container_width=True)
    c2.plotly_chart(count_bar(df["BedroomAbvGr"], "Bedrooms"), use_container_width=True)
    c3.plotly_chart(count_bar(df["TotalBathrooms"], "Total bathrooms"), use_container_width=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.plotly_chart(count_bar(df["GarageCars"], "Garage capacity (cars)"), use_container_width=True)
    c2.plotly_chart(histogram(df["GrLivArea"], "Living area (sq ft)", nbins=40, color=SERIES[0]), use_container_width=True)
    c3.plotly_chart(histogram(df["LotArea"].clip(upper=df["LotArea"].quantile(.99)), "Lot area (sq ft, 99th pct capped)",
                              nbins=40, color=SERIES[1]), use_container_width=True)

with tabs[2]:
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(scatter_vs_price(df, "GrLivArea"), use_container_width=True)
    c2.plotly_chart(scatter_vs_price(df, "YearBuilt"), use_container_width=True)
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(scatter_vs_price(df, "GarageArea"), use_container_width=True)
    c2.plotly_chart(scatter_vs_price(df, "TotalBsmtSF"), use_container_width=True)
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(box_by(df, "GarageCars", "Garage capacity vs sale price"), use_container_width=True)
    c2.plotly_chart(box_by(df, "FullBath", "Full bathrooms vs sale price"), use_container_width=True)
    st.plotly_chart(box_by(df, "BedroomAbvGr", "Bedrooms vs sale price"), use_container_width=True)
    note("Trend lines are simple least-squares fits for orientation. They ignore every other characteristic and do "
         "not imply that changing one feature causes a price change.", "info")

with tabs[3]:
    three(market_galaxy_html(df), height=560)
    md('<p style="color:var(--ink-3);font-size:12.5px">WebGL point cloud of the filtered sales: living area (x), '
       'year built (depth), sale price (height). Below: the same sales by living area × quality × price.</p>')
    st.plotly_chart(market_3d_scatter(df), use_container_width=True)

with tabs[4]:
    miss = missing_summary(raw.drop(columns=["SalePrice"]))
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        st.plotly_chart(missing_bar(miss, "Missing values (full training file)"), use_container_width=True)
    with c2:
        md("<div class='hw-insight'><div class='q'>How missing values are treated</div><div class='a'>For 15 "
           "columns (pool, alley, fence, fireplace, garage and basement descriptors, masonry type) the data "
           "dictionary says NA means the feature is absent, so they become a “None” category and related areas "
           "become 0. LotFrontage and the remaining gaps are imputed inside the model pipeline (median / most "
           "frequent), fitted on training rows only.</div></div>")
        out = detect_outliers(raw)
        md(f"<div class='hw-insight'><div class='q'>Outliers</div><div class='a'>{out['n_above_fence']} sales lie "
           f"above the 1.5×IQR fence ({money(out['iqr_upper_fence'])}). Homes "
           f"{', '.join('#' + str(i) for i in out['large_low_price_ids'])} have over 4,000 sq ft yet sold for under "
           f"$300K (partial sales), so they are removed from the training split only.</div></div>")
        md(f"<div class='hw-insight'><div class='q'>Duplicates</div><div class='a'>"
           f"{int(raw.drop(columns=['Id']).duplicated().sum())} duplicate records (ignoring Id).</div></div>")
    corr_cols = numeric_correlations(df).head(12).index.tolist() + ["SalePrice"]
    st.plotly_chart(correlation_heatmap(df[corr_cols].corr(), "Correlation heatmap (top 12 numeric features)"),
                    use_container_width=True)
    c1, c2, c3 = st.columns(3)
    for c, col in zip((c1, c2, c3), ["MSZoning", "Foundation", "SaleCondition"]):
        c.plotly_chart(count_bar(df[col], col), use_container_width=True)

with tabs[5]:
    if len(df) < 30:
        note("Select at least 30 sales for statistically meaningful automatic insights.")
    else:
        for ins in automatic_insights(df):
            md(f'<div class="hw-insight hw-anim"><div class="q">{ins["question"]}</div><div class="a">{ins["answer"]}</div></div>')
        note("Generated from the filtered data at run time. They describe associations in historical Ames sales, "
             "not causes, and not today's market.", "info")
footer()
