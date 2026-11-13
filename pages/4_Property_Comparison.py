"""Compare Properties: side-by-side 3D, table, radar, similarity and explanation differences."""

import pandas as pd
import streamlit as st

from src.config import ID_COL, TARGET
from src.data_dictionary import friendly
from src.prediction import PropertyInputError
from src.property_comparison import comparison_table, explanation_difference, price_differences, radar_scores
from src.three_scenes import house_scene_html, house_spec
from src.ui import (
    NEIGHBORHOOD_NAMES, current_property, dataset_record, footer, house_sub, kpis, load_explainer_cached,
    load_finder_cached, md, note, page_header, property_picker, require_data, require_engine, section,
    show_input_errors, three,
)
from src.visualization import SERIES, money, radar, style

import plotly.graph_objects as go

page_header("Compare properties", "Put homes head to head",
            "Compare real homes from the dataset with your own profiles. Prices are model estimates; recorded "
            "sale prices are shown for dataset homes.")
engine = require_engine()
df = require_data()
explainer = load_explainer_cached()
finder = load_finder_cached()

n = st.segmented_control("Number of properties", [2, 3], default=2, key="cmp_n") or 2
default_ids = df.sort_values("SalePrice")["Id"].iloc[[len(df) // 4, len(df) * 3 // 4, len(df) // 2]].tolist()
profiles, names, actuals = [], [], []
cols = st.columns(n, gap="large")
for i, c in enumerate(cols):
    with c:
        md(f'<div class="hw-eyebrow" style="margin-bottom:8px">Property {chr(65 + i)}</div>')
        opts = ["Dataset home", "My valued property", "Quick custom profile"]
        kind = st.radio("Source", opts, index=0 if i else (1 if "hw_inputs" in st.session_state else 0),
                        key=f"cmp_kind_{i}", horizontal=False, label_visibility="collapsed")
        try:
            if kind == "Dataset home":
                ids = df[ID_COL].tolist()
                st.session_state.setdefault(f"cmp_pick_{i}", int(default_ids[i]))
                pid = property_picker(df, f"cmp_pick_{i}", "Home")
                rec = dataset_record(df, pid)
                profiles.append(rec.drop(labels=[ID_COL, TARGET]).to_frame().T.reset_index(drop=True))
                names.append(f"#{pid}")
                actuals.append(float(rec[TARGET]))
            elif kind == "My valued property":
                inputs, base, label = current_property(engine)
                profiles.append(engine.value(inputs, base=base).record)
                names.append("Mine" if "hw_inputs" in st.session_state else "Reference")
                actuals.append(None)
            else:
                t = engine.template
                nb = st.selectbox("Neighborhood", engine.category_options["Neighborhood"], key=f"cmp_nb_{i}",
                                  index=engine.category_options["Neighborhood"].index("CollgCr"),
                                  format_func=lambda v: NEIGHBORHOOD_NAMES.get(v, v))
                q = st.slider("Overall quality", 1, 10, 7, key=f"cmp_q_{i}")
                a = st.number_input("Living area (sq ft)", 400, 6000, 1800, 50, key=f"cmp_a_{i}")
                yb = st.number_input("Year built", 1872, 2010, 2000, key=f"cmp_y_{i}")
                g = st.number_input("Garage (cars)", 0, 4, 2, key=f"cmp_g_{i}")
                b = st.number_input("Basement (sq ft)", 0, 4000, 1000, 50, key=f"cmp_b_{i}")
                r = engine.value({"Neighborhood": nb, "OverallQual": q, "GrLivArea": a, "YearBuilt": yb,
                                  "YearRemodAdd": yb, "GarageCars": g, "TotalBsmtSF": b, "YrSold": 2010})
                profiles.append(r.record)
                names.append(f"Custom {chr(65 + i)}")
                actuals.append(None)
        except PropertyInputError as err:
            show_input_errors(err)
            st.stop()

estimates = [float(engine.predict_frame(p)[0]) for p in profiles]

section("Side by side in 3D", "Each home is rebuilt from its real attributes. Toggle x-ray to see basements.")
houses = [house_spec(p.iloc[0], names[i], money(estimates[i]), house_sub(p.iloc[0])) for i, p in enumerate(profiles)]
three(house_scene_html(houses), height=520)

kpi_items = []
for i, (nm, e) in enumerate(zip(names, estimates)):
    sub = f"recorded {money(actuals[i])}" if actuals[i] else "model estimate"
    if i:
        sub = f"{(e - estimates[0]) / estimates[0]:+.1%} vs {names[0]} · " + sub
    kpi_items.append((nm, money(e), sub))
sim_ab = finder.pairwise_similarity(profiles[0], profiles[1])
kpi_items.append(("Similarity A↔B", f"{sim_ab:.0f}/100", "weighted feature distance"))
kpis(kpi_items)

section("Characteristics")
tbl = comparison_table(profiles, names, estimates, actuals)


def fmt(v, label):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if "price" in label.lower():
        return money(float(v))
    if isinstance(v, (int, float)):
        return f"{v:,.1f}" if float(v) % 1 else f"{v:,.0f}"
    return str(v)


rows = "".join(f"<tr><td>{lab}</td>" + "".join(f'<td class="num">{fmt(v, lab)}</td>' for v in tbl.loc[lab]) + "</tr>"
               for lab in tbl.index)
md(f'<div class="hw-card" style="padding:8px 4px"><table class="hw-table"><thead><tr><th>Characteristic</th>'
   + "".join(f'<th style="text-align:right">{nm}</th>' for nm in names) + f"</tr></thead><tbody>{rows}</tbody></table></div>")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.plotly_chart(radar(radar_scores(profiles, df), names), use_container_width=True)
with c2:
    diffs = price_differences(names, estimates)
    fig = go.Figure(go.Bar(x=names, y=estimates, marker=dict(color=SERIES[:len(names)], cornerradius=4),
                           text=[money(e, True) for e in estimates], textposition="outside",
                           hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(style(fig, "Model estimates", legend=False, height=440), use_container_width=True)

section("Why the model prices them differently", f"Explanation difference: {names[1]} relative to {names[0]}.")
ea = explainer.explain(profiles[0]).contributions.iloc[0]
eb = explainer.explain(profiles[1]).contributions.iloc[0]
d = explanation_difference(ea, eb, top=10).iloc[::-1]
fig = go.Figure(go.Bar(x=d["pct_effect"], y=[friendly(f) for f in d["feature"]], orientation="h",
                       marker=dict(color=[SERIES[1] if v >= 0 else SERIES[2] for v in d["pct_effect"]], cornerradius=4),
                       hovertemplate="%{y}: %{x:+.1f}% effect on estimate<extra></extra>"))
fig.add_vline(x=0, line=dict(color="rgba(255,255,255,.25)"))
fig.update_xaxes(title=f"Effect on {names[1]}'s estimate relative to {names[0]} (%)", ticksuffix="%")
st.plotly_chart(style(fig, legend=False, height=420), use_container_width=True)
note("Differences in SHAP contributions explain how the model distinguishes the two homes. They are associations "
     "learned from historical sales, not causal effects.", "info")
footer()
