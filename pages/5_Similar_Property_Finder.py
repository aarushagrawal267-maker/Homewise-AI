"""Find Similar Homes: nearest historical sales to a property."""

import streamlit as st

from src.config import ID_COL, TARGET
from src.data_dictionary import friendly
from src.three_scenes import house_scene_html, house_spec
from src.ui import (
    NEIGHBORHOOD_NAMES, current_property, dataset_record, footer, kpis, load_finder_cached, md, note, page_header,
    property_picker, require_data, require_engine, section, three,
)
from src.visualization import SERIES, money, style

import plotly.graph_objects as go

page_header("Find similar homes", "Which real sales look most like this property?",
            "Nearest-neighbour search over standardised size, quality, age, bathrooms, garage and lot features plus "
            "neighborhood and building type. Results are historical Ames sales, not properties currently for sale.")
engine = require_engine()
df = require_data()
finder = load_finder_cached()

c1, c2, c3 = st.columns([1.4, 1, 1])
with c1:
    src_choice = st.radio("Search around", ["My latest valued property", "A real home from the dataset"], horizontal=True)
with c2:
    k = st.slider("Number of matches", 3, 20, 8)
with c3:
    same_nb = st.toggle("Same neighborhood only", False)

if src_choice.startswith("My"):
    inputs, base, label = current_property(engine)
    record = engine.value(inputs, base=base).record
    exclude = st.session_state.get("hw_base_id")
else:
    pid = property_picker(df, "sim_pick")
    rec = dataset_record(df, pid)
    record = rec.drop(labels=[ID_COL, TARGET]).to_frame().T
    label, exclude = f"Home #{pid}", pid

est = float(engine.predict_frame(record)[0])
sims = finder.find(record, k=k if not same_nb else min(60, len(df) - 1), exclude_id=exclude)
if same_nb:
    sims = sims[sims["Neighborhood"] == record.iloc[0]["Neighborhood"]].head(k)
if sims.empty:
    note("No matches with these filters. Turn off ‘Same neighborhood only’.")
    st.stop()

kpis([("Selected property", label, NEIGHBORHOOD_NAMES.get(record.iloc[0]["Neighborhood"], "")),
      ("Model estimate", money(est), engine.model_label),
      ("Comparables median", money(sims[TARGET].median()), f"{len(sims)} historical sales"),
      ("Comparable range", f"{money(sims[TARGET].min(), True)}–{money(sims[TARGET].max(), True)}", "recorded prices")])

section("Your property and its three closest matches in 3D")
top3 = sims.head(3)
houses = [house_spec(record.iloc[0], label, money(est), "model estimate")]
for r in top3.itertuples():
    rr = dataset_record(df, r.Id)
    houses.append(house_spec(rr, f"#{r.Id} · {r.similarity:.0f}% match", money(r.SalePrice), "recorded sale price"))
three(house_scene_html(houses), height=500)

section("Matches")
show = sims.copy()
show["Neighborhood"] = show["Neighborhood"].map(lambda v: NEIGHBORHOOD_NAMES.get(v, v))
st.dataframe(show.rename(columns={TARGET: "Recorded sale price", "similarity": "Similarity (0–100)",
                                  "distance": "Distance", **{c: friendly(c) for c in show.columns if c not in (TARGET,)}}),
             width="stretch", hide_index=True,
             column_config={"Recorded sale price": st.column_config.NumberColumn(format="$%d"),
                            "Similarity (0–100)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                            "Distance": st.column_config.NumberColumn(format="%.2f")})

c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sims["GrLivArea"], y=sims[TARGET], mode="markers+text", text=[f"#{i}" for i in sims[ID_COL]],
                             textposition="top center", textfont=dict(color="#A9B6CA", size=10),
                             marker=dict(size=sims["similarity"] / 4 + 6, color=SERIES[1], opacity=.8),
                             name="Comparable sales", hovertemplate="%{text}<br>%{x:,} sq ft<br>$%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[record.iloc[0]["GrLivArea"]], y=[est], mode="markers", name="This property (estimate)",
                             marker=dict(size=18, color=SERIES[2], symbol="diamond", line=dict(color="white", width=2))))
    fig.update_xaxes(title="Living area (sq ft)")
    fig.update_yaxes(title="Price ($)", tickprefix="$", tickformat=",.0f")
    st.plotly_chart(style(fig, "Comparables: size vs recorded price", height=420), width="stretch")
with c2:
    section("What makes this home different")
    dist = finder.distinguishing_features(record, sims)
    for r in dist.itertuples():
        arrow = "higher" if r.difference_in_sd > 0 else "lower"
        md(f'<div class="hw-insight"><div class="q">{friendly(r.feature)}</div><div class="a">{r.this_property:,.1f} vs '
           f'comparables median {r.comparables_median:,.1f} — {abs(r.difference_in_sd):.1f} standard deviations {arrow}.</div></div>')
footer()
