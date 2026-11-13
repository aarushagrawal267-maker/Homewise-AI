"""Affordability Explorer: illustrative amortising-loan calculations (independent of the ML model)."""

import numpy as np
import streamlit as st

from src.affordability import (
    AFFORDABILITY_DISCLAIMER, AffordabilityInputError, amortization_schedule, calculate, rate_sensitivity,
)
from src.ui import footer, kpis, md, note, page_header, section
from src.visualization import SERIES, amortization_chart, balance_chart, money, style

import plotly.graph_objects as go

page_header("Affordability explorer", "What would the monthly payment look like?",
            "Standard amortising-loan arithmetic under your own assumptions. This module is mathematically separate "
            "from the valuation model.")
note(AFFORDABILITY_DISCLAIMER)

result = st.session_state.get("hw_result")
default_price = int(round(result.estimate, -3)) if result else 180000
if result:
    md(f'<p style="color:var(--ink-3);font-size:13px">Price pre-filled with your latest model estimate '
       f'({money(result.estimate)}). Change it freely.</p>')

c1, c2 = st.columns([1, 1.6], gap="large")
with c1:
    with st.container(border=True):
        price = st.number_input("Property price ($)", 10000, 5_000_000, default_price, step=5000)
        dp_mode = st.radio("Down payment as", ["Percent", "Amount"], horizontal=True)
        if dp_mode == "Percent":
            dp_pct = st.slider("Down payment (%)", 0.0, 100.0, 20.0, 0.5)
            down = price * dp_pct / 100
        else:
            down = st.number_input("Down payment ($)", 0, int(price), int(price * 0.2), step=1000)
        custom_loan = st.toggle("Set loan amount manually", False,
                                help="By default the loan is price minus down payment.")
        loan_amt = st.number_input("Loan amount ($)", 0, 5_000_000, int(price - down), step=1000) if custom_loan else None
        rate = st.slider("Annual interest rate (%)", 0.0, 15.0, 6.5, 0.05)
        years = st.select_slider("Loan duration (years)", [5, 10, 15, 20, 25, 30, 35, 40], value=30)
        income = st.number_input("Monthly gross income ($, optional)", 0, 1_000_000, 7000, step=250)

try:
    r = calculate(price, down, rate, years, monthly_income=income or None, loan_amount=loan_amt)
except AffordabilityInputError as e:
    with c2:
        note(str(e))
    st.stop()

with c2:
    kpis([("Monthly instalment", money(r.monthly_payment), f"{r.years} years at {r.annual_rate_pct:.2f}%"),
          ("Loan amount", money(r.loan_amount), f"down payment {r.down_payment_pct:.1f}% ({money(r.down_payment)})")], cols=2)
    st.write("")
    pti = r.payment_to_income_pct
    kpis([("Payment-to-income", f"{pti:.1f}%" if pti is not None else "—",
           "share of gross monthly income" if pti is not None else "enter an income"),
          ("Total interest", money(r.total_interest), f"total repaid {money(r.total_repayment)}")], cols=2)
    if pti is not None:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=pti, number=dict(suffix="%", font=dict(color="#E6EDF7", family="Space Grotesk")),
            gauge=dict(axis=dict(range=[0, 60], tickcolor="#7D8BA1"), bar=dict(color=SERIES[0], thickness=.3),
                       bgcolor="rgba(255,255,255,.03)", borderwidth=0,
                       steps=[dict(range=[0, 28], color="rgba(31,165,146,.12)"),
                              dict(range=[28, 36], color="rgba(57,135,229,.12)"),
                              dict(range=[36, 60], color="rgba(208,111,31,.14)")])))
        st.plotly_chart(style(fig, "Payment as % of income", legend=False, height=260), use_container_width=True)
        md('<p style="color:var(--ink-3);font-size:12.5px">Bands at 28% and 36% are common rule-of-thumb reference '
           'points in US lending discussions, shown for orientation only; they are not an affordability verdict.</p>')

if r.loan_amount > 0:
    sched = amortization_schedule(r.loan_amount, r.annual_rate_pct, r.years)
    section("Repayment over time")
    a, bcol = st.columns(2, gap="large")
    a.plotly_chart(amortization_chart(sched), use_container_width=True)
    bcol.plotly_chart(balance_chart(sched), use_container_width=True)
    section("Interest-rate sensitivity", "Monthly instalment for the same loan at different rates.")
    sens = rate_sensitivity(r.loan_amount, r.years, np.round(np.arange(max(0, rate - 3), rate + 3.01, 0.5), 2))
    fig = go.Figure(go.Bar(x=[f"{v:.1f}%" for v in sens["annual_rate_pct"]], y=sens["monthly_payment"],
                           marker=dict(color=[SERIES[2] if abs(v - rate) < 1e-9 else SERIES[1] for v in sens["annual_rate_pct"]],
                                       cornerradius=4),
                           hovertemplate="%{x}: %{y:$,.0f}/month<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", title="Monthly instalment")
    st.plotly_chart(style(fig, legend=False, height=340), use_container_width=True)
    with st.expander("Formula"):
        st.latex(r"M = P\,\frac{r}{1-(1+r)^{-n}},\quad r=\frac{\text{annual rate}}{12},\; n = 12 \times \text{years}")
        st.caption("If the rate is 0%, M = P / n. Taxes, insurance, fees and mortgage insurance are excluded.")
footer()
