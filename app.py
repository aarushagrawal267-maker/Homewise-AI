"""HomeWise AI — entry point.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import inject_css  # noqa: E402

st.set_page_config(
    page_title="HomeWise AI · Property Intelligence",
    page_icon=str(ROOT / "assets" / "icons" / "mark.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
st.logo(str(ROOT / "assets" / "images" / "logo.svg"), icon_image=str(ROOT / "assets" / "icons" / "mark.svg"),
        size="large")

P = ROOT / "pages"
nav = st.navigation({
    "Overview": [
        st.Page(P / "0_Home.py", title="Home", icon=":material/home:", default=True),
    ],
    "Valuation": [
        st.Page(P / "1_Property_Valuation.py", title="Property Valuation", icon=":material/price_check:"),
        st.Page(P / "2_AI_Property_Insights.py", title="AI Property Insights", icon=":material/query_stats:"),
        st.Page(P / "3_Explainable_AI.py", title="Explainable AI", icon=":material/psychology:"),
    ],
    "Explore": [
        st.Page(P / "4_Property_Comparison.py", title="Property Comparison", icon=":material/compare:"),
        st.Page(P / "5_Similar_Property_Finder.py", title="Similar Property Finder", icon=":material/travel_explore:"),
        st.Page(P / "6_Renovation_Simulator.py", title="Renovation Simulator", icon=":material/construction:"),
        st.Page(P / "7_Affordability_Explorer.py", title="Affordability Explorer", icon=":material/account_balance_wallet:"),
    ],
    "Intelligence": [
        st.Page(P / "8_Housing_Analytics.py", title="Housing Analytics", icon=":material/monitoring:"),
        st.Page(P / "9_AI_Experiment_Lab.py", title="AI Experiment Lab", icon=":material/science:"),
        st.Page(P / "10_Model_Performance.py", title="Model Performance", icon=":material/speed:"),
    ],
    "Project": [
        st.Page(P / "11_About.py", title="About HomeWise AI", icon=":material/info:"),
    ],
})
nav.run()
