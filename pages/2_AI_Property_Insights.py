"""AI Property Insights: how the valued property sits in the market, generated from data + model."""

import numpy as np
import pandas as pd
import streamlit as st

from src.data_dictionary import friendly
from src.three_scenes import market_galaxy_html
from src.ui import (
    BLDG_NAMES, NEIGHBORHOOD_NAMES, current_property, footer, kpis, load_explainer_cached, load_finder_cached, md, note,
    page_header, report_json, require_data, require_engine, section, three,
)
from src.visualization import SERIES, money, price_surface_3d, scatter_vs_price, style

import plotly.graph_objects as go

page_header("AI property insights", "Where does this home sit in the market?",
            "Automatically generated insights that combine the model's estimate with the historical sales data. "
            "Every statement is computed live from the dataset and the trained model.")
engine = require_engine()
df = require_data()
inputs, base, label = current_property(engine)
if "hw_inputs" not in st.session_state:
    note("No property valued yet in this session, so insights are shown for the reference profile "
         "(median/most-common values of the training data). Value your own home on the Property Valuation page.", "info")

res = engine.value(inputs, base=base)
rec = res.record.iloc[0]
est = res.estimate
nb = rec["Neighborhood"]
nb_df = df[df["Neighborhood"] == nb]
q_df = df[df["OverallQual"] == rec["OverallQual"]]
pct_all = float((df["SalePrice"] < est).mean() * 100)
pct_nb = float((nb_df["SalePrice"] < est).mean() * 100) if len(nb_df) else np.nan
ppsf = est / float(rec["GrLivArea"])
nb_ppsf = float((nb_df["SalePrice"] / nb_df["GrLivArea"]).median()) if len(nb_df) else np.nan

kpis([
    ("Model estimate", money(est), f"{res.interval.level:.0%} range {money(res.interval.lower, True)}–{money(res.interval.upper, True)}"),
    ("Market percentile", f"{pct_all:.0f}th", "of all 1,460 recorded sale prices"),
    (f"In {NEIGHBORHOOD_NAMES.get(nb, nb)}", f"{pct_nb:.0f}th" if np.isfinite(pct_nb) else "—",
     f"neighborhood median {money(nb_df['SalePrice'].median())} (n={len(nb_df)})"),
    ("Price per sq ft", money(ppsf), f"neighborhood median {money(nb_ppsf)}" if np.isfinite(nb_ppsf) else ""),
])

# ------------------------------------------------------------------ generated narrative ---
explainer = load_explainer_cached()
row = explainer.explain(res.record).row(0)
finder = load_finder_cached()
sims = finder.find(res.record, k=8, exclude_id=st.session_state.get("hw_base_id"))
dist = finder.distinguishing_features(res.record, sims)

top_up = row[row.dollar_contribution > 0].head(3)
top_dn = row[row.dollar_contribution < 0].head(2)
comp_med = float(sims["SalePrice"].median())
gap = (est - comp_med) / comp_med * 100
insights = [
    ("Headline", f"The model values this {BLDG_NAMES.get(rec['BldgType'], rec['BldgType']).lower()} home in "
                 f"{NEIGHBORHOOD_NAMES.get(nb, nb)} at <b>{money(est)}</b>, above {pct_all:.0f}% of historical Ames sales."),
    ("Strongest value drivers", "Raising the estimate most: " + ", ".join(
        f"{friendly(r.feature)} ({'+' if r.dollar_contribution > 0 else ''}{money(r.dollar_contribution, True)})"
        for r in top_up.itertuples()) + "."),
    ("Holding it back", ("Pulling the estimate down: " + ", ".join(
        f"{friendly(r.feature)} ({money(r.dollar_contribution, True)})" for r in top_dn.itertuples()) + ".")
     if len(top_dn) else "No characteristic pulls the estimate materially below the average home."),
    ("Versus comparable sales", f"The 8 most similar historical sales have a median recorded price of "
                                f"{money(comp_med)}; the estimate is {abs(gap):.1f}% {'above' if gap >= 0 else 'below'} that."),
    ("What sets it apart", "Compared with those comparables, the biggest differences are " + ", ".join(
        f"{friendly(r.feature)} ({r.this_property:,.1f} vs {r.comparables_median:,.1f})" for r in dist.head(3).itertuples()) + "."),
    ("Uncertainty", f"The {res.interval.level:.0%} interval spans {money(res.interval.width)} "
                    f"(±{res.interval.relative_width / 2:.0%}). Wider ranges are normal for unusual or high-value homes."),
]
section("Generated insights", "Template-generated from live model outputs and dataset statistics (no language model involved).")
for q, a in insights:
    md(f'<div class="hw-insight hw-anim"><div class="q">{q}</div><div class="a">{a}</div></div>')

# ------------------------------------------------------------------ 3D market position -----
section("Your property in the 3D market", "Orange marker = this property's model estimate among all real sales.")
three(market_galaxy_html(df, highlight={"GrLivArea": float(rec["GrLivArea"]), "YearBuilt": float(rec["YearBuilt"]),
                                        "price": est}), height=520)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.plotly_chart(scatter_vs_price(df, "GrLivArea", "Living area vs price — all sales",
                                     highlight={"x": float(rec["GrLivArea"]), "y": est, "label": "This property"}),
                    width="stretch")
with c2:
    surface = report_json("price_surface.json")
    if surface:
        st.plotly_chart(price_surface_3d(surface, point={"x": float(rec["GrLivArea"]), "y": int(rec["OverallQual"]), "z": est},
                                         height=440), width="stretch")
        md('<p style="color:var(--ink-3);font-size:13px">3D price surface: average model prediction across '
           'training homes as living area and quality vary (partial dependence). Drag to rotate.</p>')

section("Neighborhood context")
g = df.groupby("Neighborhood")["SalePrice"].median().sort_values()
fig = go.Figure(go.Bar(x=[NEIGHBORHOOD_NAMES.get(i, i) for i in g.index], y=g.values,
                       marker=dict(color=[SERIES[2] if i == nb else "#2A3A55" for i in g.index], cornerradius=4),
                       hovertemplate="%{x}<br>Median $%{y:,.0f}<extra></extra>"))
fig.add_hline(y=est, line=dict(color=SERIES[0], dash="dot"), annotation_text="This estimate",
              annotation_font_color="#EDEDED")
fig.update_yaxes(tickprefix="$", tickformat=",.0f")
st.plotly_chart(style(fig, "Median recorded sale price by neighborhood", legend=False, height=420), width="stretch")
footer()
