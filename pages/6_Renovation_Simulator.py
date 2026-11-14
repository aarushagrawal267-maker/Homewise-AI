"""Property Value Simulator: hypothetical model-based renovation scenarios with 3D before/after."""

import numpy as np
import streamlit as st

from src.config import ID_COL, TARGET
from src.data_dictionary import friendly
from src.prediction import PropertyInputError
from src.renovation_simulator import (
    SIMULATOR_DISCLAIMER, log_contribution_shift, scenarios_frame, sensitivity_curve, simulate_many,
)
from src.three_scenes import house_scene_html, house_spec
from src.ui import (
    QUAL_NAMES, current_property, dataset_record, footer, inputs_from_record, kpis, load_explainer_cached, md, note,
    page_header, property_picker, require_data, require_engine, section, show_input_errors, three,
)
from src.visualization import SERIES, money, style

import plotly.graph_objects as go

page_header("Property value simulator", "What if this home were different?",
            "Explore hypothetical changes and see how the model's estimate responds — before and after, side by side "
            "in 3D. A model-based scenario tool, not a renovation investment calculator.")
note(SIMULATOR_DISCLAIMER)
engine = require_engine()
df = require_data()
explainer = load_explainer_cached()

src_choice = st.radio("Base property", ["My latest valued property", "A real home from the dataset"], horizontal=True)
if src_choice.startswith("My"):
    base_inputs, base_rec, label = current_property(engine)
else:
    pid = property_picker(df, "sim_base")
    rec = dataset_record(df, pid)
    base_rec = rec.drop(labels=[ID_COL, TARGET])
    base_inputs = inputs_from_record(rec)
    label = f"Home #{pid}"
b = base_inputs
q_opts = [q for q in ["Po", "Fa", "TA", "Gd", "Ex"] if q in engine.category_options["KitchenQual"]]


def clip(v, lo, hi) -> int:
    return int(min(max(float(v), lo), hi))


def scenario_controls(i: int, preset: dict) -> dict:
    """Widgets for one scenario; returns only the inputs that differ from the base."""
    k = f"sc{i}_{label}"
    c1, c2, c3 = st.columns(3)
    oq = c1.slider("Overall quality", 1, 10, int(preset.get("OverallQual", b["OverallQual"])), key=k + "oq")
    gla = c2.slider("Living area (sq ft)", 300, 6000, clip(preset.get("GrLivArea", b["GrLivArea"]), 300, 6000), 10, key=k + "gla")
    bs = c3.slider("Basement area (sq ft)", 0, 6200, clip(preset.get("TotalBsmtSF", b["TotalBsmtSF"]), 0, 6200), 10, key=k + "bs")
    c1, c2, c3, c4 = st.columns(4)
    gc = c1.slider("Garage (cars)", 0, 5, clip(preset.get("GarageCars", b["GarageCars"]), 0, 5), key=k + "gc")
    fb = c2.slider("Full baths", 0, 5, clip(preset.get("FullBath", b["FullBath"]), 0, 5), key=k + "fb")
    hb = c3.slider("Half baths", 0, 3, clip(preset.get("HalfBath", b["HalfBath"]), 0, 3), key=k + "hb")
    kq_default = preset.get("KitchenQual", b["KitchenQual"])
    kq = c4.select_slider("Kitchen quality", q_opts, value=kq_default if kq_default in q_opts else "TA",
                          format_func=lambda v: QUAL_NAMES.get(v, v), key=k + "kq")
    c1, c2, c3, c4 = st.columns(4)
    yr = c1.slider("Year remodeled", int(b["YearBuilt"]), int(b["YrSold"]),
                   int(min(max(preset.get("YearRemodAdd", b["YearRemodAdd"]), b["YearBuilt"]), b["YrSold"])), key=k + "yr")
    fp = c2.slider("Fireplaces", 0, 4, clip(preset.get("Fireplaces", b["Fireplaces"]), 0, 4), key=k + "fp")
    ca = c3.toggle("Central air", bool(preset.get("HasCentralAir", b["HasCentralAir"])), key=k + "ca")
    pool = c4.toggle("Pool", bool(preset.get("HasPool", b["HasPool"])), key=k + "pool")
    new = {"OverallQual": oq, "GrLivArea": float(gla), "TotalBsmtSF": float(bs), "GarageCars": gc, "FullBath": fb,
           "HalfBath": hb, "KitchenQual": kq, "YearRemodAdd": yr, "Fireplaces": fp, "HasCentralAir": ca, "HasPool": pool}
    changes = {}
    for key, v in new.items():
        old = b.get(key)
        if isinstance(v, float) or isinstance(old, float):
            if abs(float(v) - float(old)) > 0.5:
                changes[key] = v
        elif v != old:
            changes[key] = v
    if "GarageCars" in changes:
        changes["GarageArea"] = 0.0 if gc == 0 else float(round(engine.defaults["garage_area_per_car"] * gc))
    return changes


presets = [
    ("Quality refresh", {"OverallQual": min(10, b["OverallQual"] + 1), "KitchenQual": "Gd" if b["KitchenQual"] in ("TA", "Fa", "Po") else "Ex",
                         "YearRemodAdd": b["YrSold"]}),
    ("Add living space", {"GrLivArea": b["GrLivArea"] + 300, "FullBath": min(4, b["FullBath"] + 1)}),
    ("Garage & amenities", {"GarageCars": min(4, b["GarageCars"] + 1), "Fireplaces": max(1, b["Fireplaces"]),
                            "HasCentralAir": True}),
]
section("Design your scenarios", f"Base: {label}. Each tab is one hypothetical version of the home.")
tabs = st.tabs([p[0] for p in presets])
scenarios = {}
for i, (tab, (name, preset)) in enumerate(zip(tabs, presets)):
    with tab:
        scenarios[name] = scenario_controls(i, preset)

try:
    results = simulate_many(engine, b, scenarios, base_record=base_rec)
except PropertyInputError as err:
    show_input_errors(err)
    st.stop()
if not results:
    note("Every scenario equals the base property. Move a slider to create a hypothetical change.", "info")
    st.stop()

orig = results[0].original_price
section("Before and after in 3D", "Original on the left; each scenario rebuilt from its modified attributes.")
houses = [house_spec(results[0].original_record.iloc[0], "Original", money(orig), "model estimate")]
for r in results:
    houses.append(house_spec(r.modified_record.iloc[0], r.name, money(r.modified_price),
                             f"{'+' if r.difference >= 0 else '−'}{money(abs(r.difference))} · {r.percent:+.1f}%"))
three(house_scene_html(houses), height=520)

kpis([("Original estimate", money(orig), label)] +
     [(r.name, money(r.modified_price), f"{'+' if r.difference >= 0 else '−'}{money(abs(r.difference))} ({r.percent:+.1f}%)")
      for r in results])

c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    names = ["Original"] + [r.name for r in results]
    vals = [orig] + [r.modified_price for r in results]
    fig = go.Figure(go.Bar(x=names, y=vals, marker=dict(color=["#2A3A55"] + SERIES[:len(results)], cornerradius=4),
                           text=[money(v, True) for v in vals], textposition="outside",
                           error_y=dict(type="data", symmetric=False,
                                        array=[0] + [r.upper - r.modified_price for r in results],
                                        arrayminus=[0] + [r.modified_price - r.lower for r in results],
                                        color="#7D8BA1", thickness=1)))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(style(fig, "Model estimates (whiskers: 90% interval of each scenario)", legend=False, height=420),
                    width="stretch")
with c2:
    st.dataframe(scenarios_frame(results), width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="$%.0f") for c in
                                ["Original estimate", "Modified estimate", "Difference"]} |
                               {"Difference %": st.column_config.NumberColumn(format="%+.2f%%")})
    note("Overlapping intervals mean the model cannot clearly distinguish the scenarios from the original given "
         "its typical error.", "info")

section("How the model responded")
pick = st.selectbox("Scenario", [r.name for r in results])
r = next(x for x in results if x.name == pick)
c1, c2 = st.columns([1, 1.2], gap="large")
with c1:
    md("<div class='hw-insight'><div class='q'>Fields changed in the model input</div><div class='a'>" +
       "<br>".join(f"{friendly(f.field)}: {f.before} → {f.after}" for f in r.changed_fields.itertuples()) + "</div></div>")
with c2:
    ea = explainer.explain(r.original_record).contributions.iloc[0]
    eb = explainer.explain(r.modified_record).contributions.iloc[0]
    shift = log_contribution_shift(ea, eb).iloc[::-1]
    fig = go.Figure(go.Bar(x=shift["pct"], y=[friendly(f) for f in shift["feature"]], orientation="h",
                           marker=dict(color=[SERIES[1] if v >= 0 else SERIES[2] for v in shift["pct"]], cornerradius=4),
                           hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
    fig.update_xaxes(title="Change in SHAP effect on the estimate (%)", ticksuffix="%")
    st.plotly_chart(style(fig, "Shift in feature contributions", legend=False, height=360), width="stretch")

section("Sensitivity", "Model estimate as one characteristic sweeps across a range, everything else fixed.")
feat = st.selectbox("Characteristic", ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath"], format_func=friendly)
grids = {"OverallQual": list(range(1, 11)), "GrLivArea": list(np.linspace(600, 4000, 18).round()),
         "TotalBsmtSF": list(np.linspace(0, 2500, 11).round()), "GarageCars": [0, 1, 2, 3, 4], "FullBath": [0, 1, 2, 3]}
curve = sensitivity_curve(engine, b, feat, grids[feat], base_record=base_rec)
fig = go.Figure(go.Scatter(x=curve[feat], y=curve["estimate"], mode="lines+markers", line=dict(color=SERIES[0], width=3),
                           fill="tozeroy", fillcolor="rgba(31,165,146,.08)", hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"))
fig.add_vline(x=float(b[feat]), line=dict(color=SERIES[2], dash="dot"), annotation_text="current",
              annotation_font_color="#E6EDF7")
fig.update_xaxes(title=friendly(feat))
fig.update_yaxes(tickprefix="$", tickformat=",.0f", rangemode="normal")
st.plotly_chart(style(fig, legend=False, height=380), width="stretch")
footer()
