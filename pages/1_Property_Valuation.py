"""Property Valuation: form → live model estimate, interval, 3D twin and explanation."""

import streamlit as st

from src.config import ID_COL, TARGET
from src.prediction import PropertyInputError
from src.three_scenes import house_scene_html, house_spec
from src.ui import (
    contribution_chips, dataset_record, footer, house_sub, inputs_from_record, load_explainer_cached,
    load_finder_cached, md, note, page_header, price_card, property_form, property_picker, require_data,
    require_engine, section, show_input_errors, three,
)
from src.visualization import contribution_bar, money

page_header("Property valuation", "What is this home worth?",
            "Describe a property, or start from a real home in the dataset. The estimate comes from the trained "
            "ensemble using exactly the preprocessing applied during training.")
engine = require_engine()
df = require_data()

start = st.radio("Start from", ["Reference profile (typical Ames home)", "A real home from the dataset"],
                 horizontal=True, key="val_start")
base, base_id = None, None
if start.startswith("A real"):
    base_id = property_picker(df, "val_pick")
    rec = dataset_record(df, base_id)
    base = rec.drop(labels=[ID_COL, TARGET])
    defaults = inputs_from_record(rec)
    md(f'<p style="color:var(--ink-3);font-size:13px">Unlisted characteristics (roof, exterior, porches…) are taken '
       f'from home #{base_id}. Its recorded sale price was <b style="color:var(--ink)">{money(rec[TARGET])}</b>.</p>')
else:
    defaults = st.session_state.get("hw_inputs") if st.session_state.get("hw_base_id") is None and \
        st.session_state.get("hw_inputs") else inputs_from_record(engine.template)

form_col, out_col = st.columns([1.05, 1], gap="large")
with form_col:
    submitted = property_form(engine, defaults, key=f"valuation_form_{base_id}")

if submitted is not None:
    try:
        result = engine.value(submitted, base=base)
        st.session_state.update(hw_inputs=submitted, hw_base_id=base_id, hw_result=result,
                                hw_label=f"Home #{base_id} (edited)" if base_id else "Your property")
    except PropertyInputError as err:
        with out_col:
            show_input_errors(err)
        st.stop()

result = st.session_state.get("hw_result")
with out_col:
    if result is None:
        md('<div class="hw-empty"><h3>Your estimate will appear here</h3><p>Fill in the form and press '
           '<b>Estimate value</b>. You will get a price, a calibrated range, a 3D model of the home and the '
           'factors behind the number.</p></div>')
    else:
        rec = result.record.iloc[0]
        price_card(result.estimate, result.interval.lower, result.interval.upper, result.interval.level,
                   caption=f"Model: {engine.model_label}. The range is a split-conformal interval calibrated on "
                           "cross-validation errors; it is not guaranteed to contain the true price.")
        for w in result.warnings:
            note(w)
        three(house_scene_html([house_spec(rec, st.session_state.get("hw_label", "Your property"),
                                           money(result.estimate), house_sub(rec))]), height=430)

if result is not None:
    explainer = load_explainer_cached()
    expl = explainer.explain(result.record)
    row = expl.row(0)
    section("Why this estimate", "Exact SHAP contributions of the model, converted to approximate dollars.")
    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        md(contribution_chips(row, 4))
        md(f'<p style="color:var(--ink-3);font-size:12.5px;margin-top:14px">The model starts from the average '
           f'training-home log price ({money(expl.base_price)}) and each characteristic moves it up or down. '
           'Contributions describe the model, not causal market effects.</p>')
        st.page_link("pages/3_Explainable_AI.py", label="Full explanation & waterfall", icon=":material/psychology:")
    with c2:
        st.plotly_chart(contribution_bar(row, top=10), use_container_width=True)

    finder = load_finder_cached()
    section("Closest historical sales", "Real Ames sales most similar to this property (not live listings).")
    sim = finder.find(result.record, k=4, exclude_id=st.session_state.get("hw_base_id"))
    cols = st.columns(4)
    for c, r in zip(cols, sim.itertuples()):
        c.markdown(f'<div class="hw-kpi"><div class="k">#{r.Id} · {r.Neighborhood}</div>'
                   f'<div class="v">{money(r.SalePrice)}</div><div class="s">Q{r.OverallQual} · '
                   f'{r.GrLivArea:,} sq ft · built {r.YearBuilt}<br>similarity {r.similarity:.0f}/100</div></div>',
                   unsafe_allow_html=True)
    st.write("")
    b1, b2, b3 = st.columns(3)
    b1.page_link("pages/2_AI_Property_Insights.py", label="AI insights for this home", icon=":material/auto_awesome:")
    b2.page_link("pages/6_Renovation_Simulator.py", label="Simulate renovations", icon=":material/construction:")
    b3.page_link("pages/7_Affordability_Explorer.py", label="Check affordability", icon=":material/account_balance_wallet:")
footer()
