"""Home: landing page and live dashboard."""

import streamlit as st

from src.three_scenes import house_scene_html, house_spec, market_galaxy_html
from src.ui import (
    feature_card, footer, kpis, load_engine_cached, load_train_df, md, note, report_csv, report_json, section, three,
)
from src.visualization import importance_bar, money

df = load_train_df()
engine = load_engine_cached()
summary = report_json("training_summary.json")

# ---------------------------------------------------------------- hero -----------------
left, right = st.columns([1.0, 1.25], gap="large")
with left:
    md("""<div class="hw-anim" style="padding-top:32px"><span class="hw-eyebrow">Explainable property intelligence</span>
    <div class="hw-hero-title">Understand what a home is really worth, and why</div>
    <p class="hw-hero-sub">HomeWise AI values a property with a cross-validated machine-learning ensemble, shows the
    range around the estimate, and breaks down exactly which characteristics drove it. Then lets you rotate, compare
    and re-imagine the home in 3D.</p></div>""")
    b1, b2 = st.columns(2)
    b1.page_link("pages/1_Property_Valuation.py", label="Value a property", icon=":material/arrow_forward:",
                 width="stretch")
    b2.page_link("pages/6_Renovation_Simulator.py", label="Open the 3D simulator", icon=":material/view_in_ar:",
                 width="stretch")
    md('<p class="hw-hero-meta">Trained on 1,460 real Ames, Iowa sales · '
       'Kaggle House Prices dataset · educational prototype</p>')

with right:
    if engine is not None:
        tmpl = engine.template
        est = float(engine.predict_frame(tmpl.to_frame().T)[0])
        tid = engine.bundle.get("template_id")
        house = house_spec(tmpl, f"Representative Ames home{f' #{tid}' if tid else ''} · live estimate", money(est),
                           f"{tmpl['GrLivArea']:,.0f} sq ft · Quality {int(tmpl['OverallQual'])} · "
                           f"{int(tmpl['GarageCars'])}-car garage")
        three(house_scene_html([house]), height=520)
    else:
        note("Train the models (python scripts/train_all.py) to activate the live 3D valuation preview.")

# ---------------------------------------------------------------- live metrics ------------
section("Live model dashboard", "Every number below is read from the latest training run. Nothing is hard-coded.")
if summary and df is not None:
    v, cv = summary["best_validation"], summary["best_cv"]
    kpis([
        ("Training records", f"{len(df):,}", f"{summary['n_train']} train · {summary['n_val']} validation"),
        ("Best model", summary["best_model_label"], "selected by 5-fold cross-validation"),
        ("Validation R²", f"{v['R2']:.3f}", "share of price variance explained"),
        ("Validation MAE", money(v["MAE"]), f"median error {money(v['MedianAE'])}"),
        ("CV RMSLE", f"{cv['cv_rmsle_mean']:.4f}", f"± {cv['cv_rmsle_std']:.4f} across folds"),
        ("Models trained", f"{summary['n_models_trained']}", "incl. 2 ensembles"),
    ], cols=3)
else:
    note("Model metrics will appear here after the first training run.", "info")

# ---------------------------------------------------------------- features ---------------
section("One platform, the whole valuation story")
cards = [
    ("home", "Property valuation", "A live estimate from the trained ensemble with a calibrated prediction interval."),
    ("brain", "Explainable AI", "Exact SHAP breakdowns show which characteristics pushed the price up or down."),
    ("cube", "3D property twin", "Every valued home is rebuilt in 3D from its real attributes. Drag to rotate, x-ray the walls."),
    ("wand", "Renovation simulator", "Change quality, space or amenities and watch the model's estimate respond, side by side in 3D."),
    ("scale", "Compare properties", "Put real dataset homes and your own profiles head to head with radar and explanation diffs."),
    ("search", "Find similar homes", "Nearest-neighbour search over 1,460 real historical sales with similarity scores."),
    ("wallet", "Affordability explorer", "Illustrative amortising-loan maths: payments, interest and payment-to-income."),
    ("chart", "Housing analytics", "Interactive market intelligence with filters, 3D market galaxy and data-driven insights."),
    ("flask", "AI experiment lab", "Configure and run your own models, tracked in an append-only experiment log."),
]
# The two core tools lead; the rest read as a ruled index rather than a grid of equal tiles.
lead_a, lead_b = st.columns([1.4, 1], gap="medium")
lead_a.markdown(feature_card(*cards[0], lead=True), unsafe_allow_html=True)
lead_b.markdown(feature_card(*cards[1], lead=True), unsafe_allow_html=True)
md('<div class="hw-index">' + "".join(
    f'<div class="row"><span class="t">{t}</span><span class="d">{d}</span></div>' for _, t, d in cards[2:]
) + '</div>')

# ---------------------------------------------------------------- galaxy + importance -----
if df is not None:
    section("The Ames market in three dimensions",
            "1,460 real sales: living area × year built × sale price. Hover any point to inspect the sale.")
    g1, g2 = st.columns([1.5, 1], gap="large")
    with g1:
        three(market_galaxy_html(df), height=480)
    with g2:
        shap_imp = report_csv("shap_importance.csv")
        if shap_imp is not None:
            st.plotly_chart(importance_bar(shap_imp, "mean_abs_shap", "What the model relies on most", top=10,
                                           xtitle="Mean |SHAP| (log-price)"), width="stretch")
            md('<p style="color:var(--ink-3);font-size:13px">Average absolute SHAP contribution on validation '
               'homes. These show what the model relies on, not proof that changing a feature causes price changes.</p>')

# ---------------------------------------------------------------- workflow ----------------
section("How the AI works")
md("""<div class="hw-steps">
<div class="hw-step"><div class="n">01</div><h5>Validate & clean</h5><p>Schema, target and duplicate checks. Missing values handled using the data dictionary (“NA” basement = no basement).</p></div>
<div class="hw-step"><div class="n">02</div><h5>Engineer features</h5><p>Total area, bathrooms, age at sale, quality interactions and amenity flags — each with an ablation test.</p></div>
<div class="hw-step"><div class="n">03</div><h5>Tune & compare</h5><p>Eight algorithms tuned with grid search, random search and Optuna inside 5-fold cross-validation.</p></div>
<div class="hw-step"><div class="n">04</div><h5>Ensemble & select</h5><p>Weighted blending and stacking; the winner is chosen by CV only, then scored once on held-out validation.</p></div>
<div class="hw-step"><div class="n">05</div><h5>Explain & quantify</h5><p>Exact SHAP contributions, permutation importance, partial dependence and conformal intervals.</p></div>
</div>""")

section("Data & limitations")
c1, c2 = st.columns(2, gap="large")
c1.markdown(feature_card("chart", "Kaggle House Prices: Advanced Regression Techniques",
                         "1,460 labelled sales (train.csv) and 1,459 unlabelled homes (test.csv) from Ames, Iowa, "
                         "sold 2006–2010, with 79 property characteristics. The competition test set is used only "
                         "for submission-style inference, never for scoring or model selection."), unsafe_allow_html=True)
c2.markdown(feature_card("shield", "Use responsibly",
                         "A research and educational prototype. It does not value homes in other cities or "
                         "countries, does not reflect today's market, and is not an appraisal. Feature contributions "
                         "and simulator differences describe the model, not causal effects."), unsafe_allow_html=True)
footer()
