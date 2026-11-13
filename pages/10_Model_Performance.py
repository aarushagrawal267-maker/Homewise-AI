"""Model Performance: comparison, cross-validation, validation, errors and uncertainty calibration."""

import json

import pandas as pd
import streamlit as st

from src.config import PATHS
from src.experiment_tracker import load_experiments
from src.ui import footer, kpis, md, note, page_header, report_csv, report_json, require_engine, section
from src.visualization import (
    SERIES, actual_vs_predicted, cv_folds_chart, model_comparison_bar, money, residual_chart, style,
)

import plotly.graph_objects as go

page_header("Model performance", "How good is the model, honestly?",
            "Three kinds of numbers, never mixed: training fit (optimistic), 5-fold cross-validation on the training "
            "split (used to choose the model) and the held-out validation split (scored once, for reporting).")
engine = require_engine()
summary = report_json("training_summary.json")
comp = report_csv("model_comparison.csv")
if summary is None or comp is None:
    note("Training reports not found. Run python scripts/train_all.py.")
    st.stop()

v, cv = summary["best_validation"], summary["best_cv"]
kpis([("Selected model", summary["best_model_label"], "lowest CV RMSLE"),
      ("CV RMSLE", f"{cv['cv_rmsle_mean']:.4f}", f"± {cv['cv_rmsle_std']:.4f} (5 folds)"),
      ("Validation RMSE", money(v["RMSE"]), f"MAE {money(v['MAE'])} · median {money(v['MedianAE'])}"),
      ("Validation R²", f"{v['R2']:.4f}", f"RMSLE {v['RMSLE']:.4f} · MAPE {v['MAPE']:.1f}%"),
      ("Within ±10%", f"{summary['best_residuals']['share_within_10pct']:.0f}%",
       f"within ±20%: {summary['best_residuals']['share_within_20pct']:.0f}% of validation homes")], cols=3)
md(f'<p style="color:var(--ink-3);font-size:12.5px;margin-top:10px">Run {summary["run_id"]} · trained '
   f'{summary["trained_at"]} · {summary["split"]} · {summary["n_train"]} training rows after removing outlier Ids '
   f'{summary["removed_outliers_ids"]} · {summary["n_val"]} validation rows.</p>')

tabs = st.tabs(["Model comparison", "Validation diagnostics", "Uncertainty calibration", "Feature engineering",
                "Hyperparameter tuning"])

with tabs[0]:
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(model_comparison_bar(comp, "CV RMSLE", "Cross-validation RMSLE (selection metric, lower = better)"),
                    use_container_width=True)
    c2.plotly_chart(model_comparison_bar(comp, "Val RMSE", "Validation RMSE in dollars (lower = better)", fmt=",.0f"),
                    use_container_width=True)
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(model_comparison_bar(comp, "Val R2", "Validation R² (higher = better)", lower_is_better=False),
                    use_container_width=True)
    c2.plotly_chart(model_comparison_bar(comp, "Train time (s)", "Training time on the training split (s)", fmt=".2f"),
                    use_container_width=True)
    st.dataframe(comp.drop(columns=["model_key"]), use_container_width=True, hide_index=True,
                 column_config={"Val MAE": st.column_config.NumberColumn(format="$%.0f"),
                                "Val RMSE": st.column_config.NumberColumn(format="$%.0f"),
                                "Val MedianAE": st.column_config.NumberColumn(format="$%.0f"),
                                **{c: st.column_config.NumberColumn(format="%.4f") for c in
                                   ["CV RMSLE", "CV RMSLE std", "Train RMSLE", "Val R2", "Val RMSLE"]},
                                "Val MAPE %": st.column_config.NumberColumn(format="%.2f"),
                                "Train time (s)": st.column_config.NumberColumn(format="%.2f"),
                                "Inference (ms/row)": st.column_config.NumberColumn(format="%.3f")})
    runs = [e for e in load_experiments() if e.get("run_id") == summary["run_id"]]
    if runs:
        st.plotly_chart(cv_folds_chart({e["model_label"]: e["cv_fold_rmsle"] for e in
                                        sorted(runs, key=lambda e: e["cv_rmsle_mean"])}), use_container_width=True)
    note("Train RMSLE is measured on the data the model was fitted on and is therefore optimistic. The gap between "
         "train and CV RMSLE indicates overfitting (largest for tree models). Selection used CV only; the validation "
         "split was not used to choose models or hyperparameters.", "info")

with tabs[1]:
    vp = report_csv("validation_predictions.csv")
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(actual_vs_predicted(vp["actual"], vp["predicted"]), use_container_width=True)
    c2.plotly_chart(residual_chart(vp["actual"], vp["predicted"]), use_container_width=True)
    band = report_csv("error_by_price_band.csv")
    cond = report_csv("error_by_sale_condition.csv")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure(go.Bar(x=band["price_band"], y=band["median_abs_pct"], marker=dict(color=SERIES[1], cornerradius=4),
                               customdata=band[["n", "MAE"]].to_numpy(),
                               hovertemplate="%{x}<br>median |error| %{y:.1f}%<br>n=%{customdata[0]} · MAE $%{customdata[1]:,.0f}<extra></extra>"))
        fig.update_yaxes(title="Median absolute % error", ticksuffix="%")
        st.plotly_chart(style(fig, "Error by actual price quartile", legend=False, height=340), use_container_width=True)
    with c2:
        fig = go.Figure(go.Bar(x=cond["SaleCondition"], y=cond["median_abs_pct"], marker=dict(color=SERIES[3], cornerradius=4),
                               customdata=cond["n"], hovertemplate="%{x}<br>median |error| %{y:.1f}%<br>n=%{customdata}<extra></extra>"))
        fig.update_yaxes(title="Median absolute % error", ticksuffix="%")
        st.plotly_chart(style(fig, "Error by sale condition", legend=False, height=340), use_container_width=True)
    section("Largest validation errors")
    worst = vp.reindex(vp["abs_pct_error"].sort_values(ascending=False).index).head(10)
    st.dataframe(worst, use_container_width=True, hide_index=True,
                 column_config={"actual": st.column_config.NumberColumn(format="$%.0f"),
                                "predicted": st.column_config.NumberColumn(format="$%.0f"),
                                "error": st.column_config.NumberColumn(format="$%.0f"),
                                "abs_pct_error": st.column_config.NumberColumn("|error| %", format="%.1f")})

with tabs[2]:
    cov = summary["coverage"]
    rows = [{"Target coverage": f"{float(k):.0%}", "Validation coverage": f"{c['empirical']:.1%}",
             "Median relative width": f"±{c['median_relative_width'] / 2:.1%}", "Mean width": money(c["mean_width"]),
             "Log half-width q": f"{c['log_half_width']:.4f}"} for k, c in cov.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    levels = [float(k) for k in cov]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{l:.0%}" for l in levels], y=[l * 100 for l in levels], name="Target", marker=dict(color="#2A3A55")))
    fig.add_trace(go.Bar(x=[f"{l:.0%}" for l in levels], y=[cov[k]["empirical"] * 100 for k in cov], name="Achieved on validation",
                         marker=dict(color=SERIES[0])))
    fig.update_yaxes(title="Coverage (%)", range=[0, 100], ticksuffix="%")
    st.plotly_chart(style(fig, "Conformal interval coverage: target vs validation", height=340), use_container_width=True)
    md("<div class='hw-insight'><div class='q'>Method</div><div class='a'>Split-conformal prediction on the log-price "
       "scale. Absolute residuals of out-of-fold predictions (each training home predicted by a model that did not see "
       "it) give the finite-sample-corrected quantile q; intervals are exp(log ŷ ± q). Coverage holds on average over "
       "homes similar to the training data and is checked here on the untouched validation split. It is not "
       "guaranteed for any individual home.</div></div>")

with tabs[3]:
    abl = report_csv("feature_ablation.csv")
    fig = go.Figure()
    for i, (mdl, g) in enumerate(abl.groupby("model")):
        fig.add_trace(go.Bar(x=g["configuration"], y=g["cv_rmsle_mean"], name=mdl, marker=dict(color=SERIES[i], cornerradius=4),
                             error_y=dict(type="data", array=g["cv_rmsle_std"], color="#7D8BA1", thickness=1)))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="CV RMSLE (lower = better)", range=[abl["cv_rmsle_mean"].min() * 0.9, abl["cv_rmsle_mean"].max() * 1.04])
    st.plotly_chart(style(fig, "Feature-engineering ablation (5-fold CV on the training split)", height=420),
                    use_container_width=True)
    st.dataframe(abl, use_container_width=True, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["cv_rmsle_mean", "cv_rmsle_std"]} |
                               {"delta_vs_all": st.column_config.NumberColumn("Δ vs all features", format="%+.4f")})
    note("A positive Δ when a group is removed means that group was helping. Fold-to-fold standard deviation is "
         "larger than most differences, so small effects are not conclusive.", "info")

with tabs[4]:
    tun = summary.get("tuning", {})
    rows = [{"Model": k, "Method": t.get("method"), "Candidates": t.get("n_candidates"),
             "Best search CV RMSLE": t.get("best_cv_rmsle"), "Search time (s)": t.get("tuning_seconds"),
             "Best parameters": json.dumps(t.get("best_params"))} for k, t in tun.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                 column_config={"Best search CV RMSLE": st.column_config.NumberColumn(format="%.4f"),
                                "Search time (s)": st.column_config.NumberColumn(format="%.1f")})
    trials = report_json("tuning_trials.json") or {}
    opt = {k: v for k, v in trials.items() if v}
    if opt:
        pick = st.selectbox("Optuna optimisation history", list(opt))
        tr = pd.DataFrame(opt[pick])
        tr["best_so_far"] = tr["value"].cummin()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr["number"], y=tr["value"], mode="markers", name="Trial", marker=dict(color=SERIES[1], size=8)))
        fig.add_trace(go.Scatter(x=tr["number"], y=tr["best_so_far"], mode="lines", name="Best so far",
                                 line=dict(color=SERIES[0], width=3, shape="hv")))
        fig.update_xaxes(title="Trial")
        fig.update_yaxes(title="CV RMSLE")
        st.plotly_chart(style(fig, f"Optuna TPE search: {pick}", height=360), use_container_width=True)
    note(f"Ensemble members (best CV): {', '.join(summary['ensemble_members'])}; weights optimised on out-of-fold "
         f"predictions: {', '.join(f'{w:.3f}' for w in summary['ensemble_weights'])}. Because weights are fitted on the "
         "same OOF predictions used to score the weighted ensemble's CV, its CV score is slightly optimistic; the "
         "validation split provides the unbiased check.", "info")
footer()
