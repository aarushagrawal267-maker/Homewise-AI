"""About HomeWise AI."""

import streamlit as st

from src.data_loader import dataset_status
from src.ui import feature_card, footer, md, page_header, report_json, section

page_header("About HomeWise AI", "Understand property value. Discover what drives it.",
            "An explainable property valuation and housing analytics platform built as a research and educational "
            "prototype.")

section("The problem")
md("""<div class="hw-insight"><div class="a">Two houses on the same street can sell for very different prices because
of size, quality, condition, age, layout and amenities. A single predicted number does not explain those differences,
and it hides how uncertain the estimate is. HomeWise AI pairs a validated machine-learning valuation with exact
explanations, calibrated uncertainty, comparable sales and what-if simulation, so the estimate can be understood and
questioned rather than taken on faith.</div></div>""")

section("Dataset")
s = dataset_status()
md(f"""<div class="hw-insight"><div class="a"><b>Kaggle — House Prices: Advanced Regression Techniques</b>
(<a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data" target="_blank">competition page</a>),
derived from Dean De Cock's Ames, Iowa housing data (Journal of Statistics Education, 2011). train.csv: 1,460 sales
with SalePrice; test.csv: 1,459 homes without prices; 79 characteristics; sales from 2006 to 2010.<br><br>
Local files: train.csv {'✓' if s['train.csv']['exists'] else '✗'}, test.csv {'✓' if s['test.csv']['exists'] else '✗'},
data_description.txt {'✓' if s['data_description.txt']['exists'] else '— (built-in feature dictionary used)'}.
Checksums are verified on download.</div></div>""")

section("Limitations")
c1, c2 = st.columns(2, gap="large")
c1.markdown(feature_card("alert", "Geography and time",
                         "Only Ames, Iowa, 2006–2010. The model cannot value homes in Bengaluru, Mumbai, London or any "
                         "other market, and knows nothing about prices after 2010."), unsafe_allow_html=True)
c2.markdown(feature_card("alert", "Correlation, not causation",
                         "SHAP values, partial dependence and simulator differences describe the model's learned "
                         "associations. They do not prove that renovating would change a sale price."), unsafe_allow_html=True)
st.write("")
c1, c2 = st.columns(2, gap="large")
c1.markdown(feature_card("alert", "Small data",
                         "1,460 sales limit precision, especially for rare homes (very large, very old, unusual "
                         "neighborhoods). Intervals are wider and less reliable there."), unsafe_allow_html=True)
c2.markdown(feature_card("alert", "Not advice",
                         "Not an appraisal, mortgage decision or financial advice. The affordability tool uses "
                         "illustrative arithmetic only."), unsafe_allow_html=True)

section("Technology")
md("""<div class="hw-steps">
<div class="hw-step"><div class="n">ML</div><h5>scikit-learn · XGBoost · LightGBM</h5><p>Pipelines, ColumnTransformers, eight algorithms, voting & stacking ensembles.</p></div>
<div class="hw-step"><div class="n">TUNE</div><h5>Optuna · GridSearchCV · RandomizedSearchCV</h5><p>Cross-validated searches sized for a laptop CPU.</p></div>
<div class="hw-step"><div class="n">XAI</div><h5>SHAP · permutation · PDP/ICE</h5><p>Exact TreeSHAP and linear SHAP, combined across ensemble members.</p></div>
<div class="hw-step"><div class="n">UI</div><h5>Streamlit · Plotly · Three.js</h5><p>Custom design system, WebGL 3D property twins and market galaxy.</p></div>
</div>""")

summary = report_json("training_summary.json")
if summary:
    section("Current model")
    md(f"""<div class="hw-insight"><div class="a">{summary['best_model_label']} · trained {summary['trained_at']} ·
    run {summary['run_id']} · {summary['n_models_trained']} models compared. See Model Performance for every metric.</div></div>""")
footer()
