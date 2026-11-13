"""AI Experiment Lab: configure, run and track your own model experiments."""

import time

import joblib
import pandas as pd
import streamlit as st

from src.config import CONFIG, PATHS
from src.data_loader import split_features_target
from src.evaluate import cross_validate_oof, regression_metrics, timed_fit_predict
from src.experiment_tracker import experiments_frame, log_experiment, new_experiment_id
from src.explainability import permutation_importance_raw
from src.models import MODEL_SPECS, available_models, build_model, model_label
from src.preprocessing import make_train_validation_split, remove_training_outliers
from src.ui import footer, kpis, md, note, page_header, require_data, section
from src.visualization import SERIES, importance_bar, style

import plotly.graph_objects as go

page_header("AI experiment lab", "Run your own model experiments",
            "Pick an algorithm, set its hyperparameters and run cross-validation. Every run is appended to the "
            "experiment log; earlier records are never overwritten.")
raw = require_data()

c1, c2 = st.columns([1, 1.4], gap="large")
with c1:
    with st.container(border=True):
        key = st.selectbox("Model", available_models(), format_func=model_label)
        spec = MODEL_SPECS[key]
        md(f'<p style="color:var(--ink-3);font-size:13px">{spec.purpose}</p>')
        params = {}
        for name, (kind, lo, hi, default) in spec.lab_params.items():
            if kind == "int":
                params[name] = st.slider(name, int(lo), int(hi), int(default), key=f"lab_{key}_{name}")
            else:
                params[name] = st.number_input(name, float(lo), float(hi), float(default), format="%.5f",
                                               key=f"lab_{key}_{name}")
        st.divider()
        a, b = st.columns(2)
        folds = a.slider("CV folds", 3, 10, int(CONFIG["split"]["cv_folds"]))
        seed = b.number_input("Random seed", 0, 99999, int(CONFIG["split"]["random_seed"]))
        fe = st.toggle("Feature engineering", True)
        outl = st.toggle("Remove training outliers", bool(CONFIG["preprocessing"]["remove_training_outliers"]))
        save_model = st.toggle("Save the trained model file", False)
        run = st.button("Run experiment", type="primary", use_container_width=True)

with c2:
    if run:
        X, y = split_features_target(raw)
        Xtr, Xva, ytr, yva = make_train_validation_split(X, y, seed=int(seed))
        if outl:
            Xtr, ytr, _ = remove_training_outliers(Xtr, ytr)
        if "random_state" in spec.defaults:
            params["random_state"] = int(seed)
        model = build_model(key, params, feature_engineering=fe)
        bar = st.progress(0.0, text="Cross-validating…")
        t0 = time.perf_counter()
        cv = cross_validate_oof(model, Xtr, ytr, n_splits=folds, seed=int(seed),
                                progress=lambda i, n: bar.progress(i / (n + 1), text=f"Fold {i}/{n} complete"))
        bar.progress(folds / (folds + 1), text="Training on the full training split…")
        fitted, val_pred, train_s, infer_ms = timed_fit_predict(build_model(key, params, feature_engineering=fe), Xtr, ytr, Xva)
        val = regression_metrics(yva, val_pred)
        bar.progress(1.0, text=f"Done in {time.perf_counter() - t0:.1f}s")
        exp_id = new_experiment_id("LAB")
        model_path = None
        if save_model:
            PATHS.lab_models_dir.mkdir(parents=True, exist_ok=True)
            model_path = PATHS.lab_models_dir / f"{exp_id}.joblib"
            joblib.dump(fitted, model_path, compress=3)
        rec = log_experiment({
            "experiment_id": exp_id, "source": "experiment_lab", "model": key, "model_label": model_label(key),
            "params": params, "split": f"stratified {1 - CONFIG['split']['validation_size']:.0%}/"
                                       f"{CONFIG['split']['validation_size']:.0%}, seed {seed}, {folds}-fold CV",
            "cv_rmsle_mean": cv["cv_rmsle_mean"], "cv_rmsle_std": cv["cv_rmsle_std"], "cv_fold_rmsle": cv["cv_fold_rmsle"],
            "validation_metrics": val, "train_seconds": train_s, "inference_ms_per_row": infer_ms,
            "feature_engineering": fe, "outliers_removed": outl,
            "model_file": str(model_path.relative_to(PATHS.root)) if model_path else None})
        st.session_state["lab_last"] = {"rec": rec, "fitted": fitted, "Xva": Xva, "yva": yva, "folds": cv["cv_fold_rmsle"]}

    last = st.session_state.get("lab_last")
    if last is None:
        md('<div class="hw-empty"><h3>No experiment run yet</h3><p>Configure a model on the left and press '
           '<b>Run experiment</b>. A Ridge or LightGBM run takes a few seconds on a laptop.</p></div>')
    else:
        r = last["rec"]
        v = r["validation_metrics"]
        md(f'<p style="color:var(--ink-3);font-size:13px">Experiment <code>{r["experiment_id"]}</code> · '
           f'{r["model_label"]} · logged to outputs/experiments/experiments.jsonl</p>')
        kpis([("CV RMSLE", f"{r['cv_rmsle_mean']:.4f}", f"± {r['cv_rmsle_std']:.4f}"),
              ("Validation RMSE", f"${v['RMSE']:,.0f}", f"MAE ${v['MAE']:,.0f}"),
              ("Validation R²", f"{v['R2']:.4f}", f"RMSLE {v['RMSLE']:.4f}"),
              ("Training time", f"{r['train_seconds']:.2f}s", f"{r['inference_ms_per_row']:.3f} ms/row")], cols=2)
        fig = go.Figure(go.Bar(x=[f"Fold {i + 1}" for i in range(len(last["folds"]))], y=last["folds"],
                               marker=dict(color=SERIES[0], cornerradius=4)))
        fig.add_hline(y=r["cv_rmsle_mean"], line=dict(color=SERIES[2], dash="dot"), annotation_text="mean",
                      annotation_font_color="#E6EDF7")
        st.plotly_chart(style(fig, "RMSLE per cross-validation fold", legend=False, height=280), use_container_width=True)
        if st.button("Compute permutation importance (validation, ~10–60 s)"):
            with st.spinner("Shuffling each column and re-scoring…"):
                pim = permutation_importance_raw(last["fitted"], last["Xva"], last["yva"], n_repeats=3)
            st.plotly_chart(importance_bar(pim, "importance_mean", "Permutation importance", top=12,
                                           err_col="importance_std", xtitle="Increase in RMSLE"), use_container_width=True)
        note("Validation metrics are shown for learning purposes. Choosing settings by repeatedly checking the "
             "validation score overfits it; the production model was selected by cross-validation only.", "info")

section("Experiment tracking", "All runs from training scripts and this lab, newest first.")
exp = experiments_frame()
if exp.empty:
    note("No experiments recorded yet.", "info")
else:
    exp = exp.iloc[::-1].reset_index(drop=True)
    src_filter = st.multiselect("Source", sorted(exp["Source"].dropna().unique()), placeholder="All sources")
    view = exp[exp["Source"].isin(src_filter)] if src_filter else exp
    st.dataframe(view, use_container_width=True, hide_index=True,
                 column_config={"Val MAE": st.column_config.NumberColumn(format="$%.0f"),
                                "Val RMSE": st.column_config.NumberColumn(format="$%.0f"),
                                "Val R²": st.column_config.NumberColumn(format="%.4f"),
                                "CV RMSLE": st.column_config.NumberColumn(format="%.4f"),
                                "CV RMSLE std": st.column_config.NumberColumn(format="%.4f"),
                                "Val RMSLE": st.column_config.NumberColumn(format="%.4f"),
                                "Train time (s)": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download experiment log (CSV)", view.to_csv(index=False).encode(), "experiments.csv",
                       "text/csv")
    pts = view.dropna(subset=["CV RMSLE", "Val RMSLE"])
    if len(pts):
        fig = go.Figure(go.Scatter(x=pts["CV RMSLE"], y=pts["Val RMSLE"], mode="markers", text=pts["Model"],
                                   marker=dict(size=10, color=SERIES[1], opacity=.8, line=dict(color="#0B1220", width=1)),
                                   hovertemplate="%{text}<br>CV %{x:.4f} · Val %{y:.4f}<extra></extra>"))
        lo, hi = min(pts["CV RMSLE"].min(), pts["Val RMSLE"].min()), max(pts["CV RMSLE"].max(), pts["Val RMSLE"].max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="rgba(255,255,255,.25)", dash="dash"),
                                 hoverinfo="skip"))
        fig.update_xaxes(title="Cross-validation RMSLE")
        fig.update_yaxes(title="Validation RMSLE")
        st.plotly_chart(style(fig, "Cross-validation vs validation error per experiment", legend=False, height=380),
                        use_container_width=True)
footer()
