"""Explainable AI: individual SHAP explanations, global importance, partial dependence."""

import pandas as pd
import streamlit as st

from src.config import ID_COL, TARGET
from src.data_dictionary import friendly
from src.ui import (
    current_property, dataset_record, footer, kpis, load_explainer_cached, md, note, page_header, property_picker,
    report_csv, report_json, require_data, require_engine, section,
)
from src.visualization import (
    contribution_bar, importance_bar, money, pdp_chart, price_surface_3d, shap_beeswarm, waterfall,
)

page_header("Explainable AI", "Why is this property valued this way?",
            "SHAP decomposes each prediction exactly into contributions from every characteristic. Global views "
            "show what the model relies on overall. None of these establish cause and effect in the real market.")
engine = require_engine()
df = require_data()
explainer = load_explainer_cached()

tab1, tab2, tab3, tab4 = st.tabs(["Individual explanation", "Global importance", "Partial dependence", "3D price surface"])

with tab1:
    src_choice = st.radio("Explain", ["My latest valued property", "A real home from the dataset"], horizontal=True)
    if src_choice.startswith("My"):
        inputs, base, label = current_property(engine)
        res = engine.value(inputs, base=base)
        record, actual = res.record, None
    else:
        pid = property_picker(df, "xai_pick")
        rec = dataset_record(df, pid)
        record = rec.drop(labels=[ID_COL, TARGET]).to_frame().T
        actual, label = float(rec[TARGET]), f"Home #{pid}"
    expl = explainer.explain(record)
    row = expl.row(0)
    est = float(engine.predict_frame(record)[0])
    kpis([("Property", label, ""), ("Model estimate", money(est), engine.model_label),
          ("Baseline (average home)", money(expl.base_price), "expected model output"),
          ("Recorded sale price", money(actual) if actual else "—",
           f"error {money(est - actual)} ({(est - actual) / actual:+.1%})" if actual else "not a dataset sale")])
    st.write("")
    st.plotly_chart(waterfall(row, expl.base_price, est, top=10), width="stretch")
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.plotly_chart(contribution_bar(row, top=14, title="Feature contributions (≈ $)"), width="stretch")
    with c2:
        section("How to read this")
        md(f"""<div class="hw-insight"><div class="a">The model predicts the <b>logarithm</b> of price. On that scale SHAP
        values add up exactly: baseline + contributions = prediction (max reconstruction error here
        {expl.additivity_error():.1e}). Dollar values allocate the gap between the baseline
        ({money(expl.base_price)}) and the estimate ({money(est)}) in proportion to each log contribution, so they
        sum to the estimate. The “% effect” column is the exact multiplicative effect on the price.</div></div>
        <div class="hw-insight"><div class="a">For the ensemble, each member model is explained exactly
        (TreeSHAP for boosted trees, closed-form for linear models) and contributions are combined with the
        ensemble weights. One-hot columns are summed back to their original characteristic.</div></div>""")
        note("A positive contribution means the model associates this value with higher prices in 2006–2010 Ames "
             "sales. It is not evidence that changing the feature would cause that price change.")
    tbl = row.head(25).copy()
    tbl["feature"] = tbl["feature"].map(friendly)
    tbl["value"] = tbl["value"].map(lambda v: f"{v:,.4g}" if isinstance(v, (int, float)) else str(v))
    st.dataframe(tbl.rename(columns={"feature": "Characteristic", "value": "Value", "dollar_contribution": "≈ $ contribution",
                                     "pct_effect": "% effect", "log_contribution": "SHAP (log price)"}),
                 width="stretch", hide_index=True,
                 column_config={"≈ $ contribution": st.column_config.NumberColumn(format="$%.0f"),
                                "% effect": st.column_config.NumberColumn(format="%+.2f%%"),
                                "SHAP (log price)": st.column_config.NumberColumn(format="%+.4f")})

with tab2:
    shap_vals = report_csv("shap_values_validation.csv")
    shap_feat = report_csv("shap_feature_values_validation.csv")
    shap_imp = report_csv("shap_importance.csv")
    perm = report_csv("permutation_importance.csv")
    if shap_vals is None:
        note("Global explanation files are missing. Re-run python scripts/train_all.py.")
    else:
        st.plotly_chart(shap_beeswarm(shap_vals.drop(columns=["Id"]), shap_feat.drop(columns=["Id"]), top=16),
                        width="stretch")
        md('<p style="color:var(--ink-3);font-size:13px">Each dot is one validation property. Position = how much '
           'that characteristic moved its predicted log price; colour = whether its value was high (blue) or low '
           '(orange) relative to other homes.</p>')
        c1, c2 = st.columns(2, gap="large")
        c1.plotly_chart(importance_bar(shap_imp, "mean_abs_shap", "SHAP importance (mean |SHAP|)", top=15,
                                       xtitle="Mean |SHAP| (log price)"), width="stretch")
        c2.plotly_chart(importance_bar(perm, "importance_mean", "Permutation importance (validation)", top=15,
                                       err_col="importance_std", xtitle="Increase in RMSLE when shuffled"),
                        width="stretch")
        note("SHAP importance is computed on engineered features; permutation importance shuffles the raw input "
             "columns on the validation set. Correlated features (e.g. GrLivArea, TotalSF, QualityArea) share credit, "
             "so a single feature can look less important than the underlying concept really is.", "info")

with tab3:
    pdps = report_json("partial_dependence.json")
    if not pdps:
        note("Partial-dependence results are missing. Re-run python scripts/train_all.py.")
    else:
        feat = st.selectbox("Characteristic", list(pdps), format_func=friendly)
        inputs, base, _ = current_property(engine)
        cur = engine.value(inputs, base=base).record.iloc[0][feat]
        st.plotly_chart(pdp_chart(pdps[feat], current=float(cur)), width="stretch")
        md('<p style="color:var(--ink-3);font-size:13px">Thick line: average model prediction over 300 training '
           'homes when the characteristic is set to each value (related fields such as floor areas are kept '
           'consistent). Thin lines: individual homes (ICE). Dotted line: your latest property.</p>')
        note("Partial dependence shows how the model's output responds while other characteristics keep their "
             "observed values. It is not proof that changing the feature would change a real sale price by that amount.")

with tab4:
    surface = report_json("price_surface.json")
    if surface:
        inputs, base, _ = current_property(engine)
        r = engine.value(inputs, base=base)
        rr = r.record.iloc[0]
        st.plotly_chart(price_surface_3d(surface, point={"x": float(rr["GrLivArea"]), "y": int(rr["OverallQual"]),
                                                         "z": r.estimate}, height=640), width="stretch")
        md('<p style="color:var(--ink-3);font-size:13px">Two-way partial dependence: average model prediction over '
           '150 training homes for each living-area × quality combination. Drag to rotate; scroll to zoom.</p>')
footer()
