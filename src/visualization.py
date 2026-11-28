"""Plotly chart builders with the HomeWise visual language.

Palette validated (dark surface #0B1220) for lightness band, chroma, colour-
vision-deficiency separation and contrast. Categorical hues are assigned in a
fixed order; "raises / lowers the estimate" uses the blue/orange diverging pair;
price magnitude uses a single-hue teal ramp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data_dictionary import friendly

SURFACE = "#0B1220"
INK = "#E6EDF7"
INK_2 = "#A9B6CA"
INK_3 = "#7D8BA1"
GRID = "rgba(255,255,255,0.06)"
SERIES = ["#1FA592", "#3987E5", "#D06F1F", "#9085E9", "#D55181"]
UP, DOWN = "#3987E5", "#D06F1F"
TEAL_RAMP = [[0.0, "#0D2B33"], [0.25, "#0F4C4F"], [0.5, "#14806F"], [0.75, "#1FA592"], [1.0, "#7EE8D2"]]
FONT = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"


def money(v: float, short: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if short:
        if abs(v) >= 1e6:
            return f"${v / 1e6:.2f}M"
        if abs(v) >= 1e3:
            return f"${v / 1e3:.0f}K"
    return f"${v:,.0f}"


def style(fig: go.Figure, title: str | None = None, legend: bool = True, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=dict(text=title, font=dict(size=15, color=INK, family=FONT), x=0, xanchor="left") if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK_2, size=12),
        margin=dict(l=10, r=10, t=48 if title else 16, b=10),
        colorway=SERIES,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#232325", bordercolor="#3a3a3d", font=dict(family=FONT, color=INK)),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="rgba(255,255,255,0.1)",
                     tickfont=dict(color=INK_3))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="rgba(255,255,255,0.1)",
                     tickfont=dict(color=INK_3))
    return fig


# --- Distributions & relationships ------------------------------------------------------

def histogram(series: pd.Series, title: str, nbins: int = 50, color: str = SERIES[0],
              xprefix: str = "") -> go.Figure:
    fig = go.Figure(go.Histogram(x=series, nbinsx=nbins, marker=dict(color=color, line=dict(color=SURFACE, width=1)),
                                 hovertemplate=f"{xprefix}%{{x}}<br>%{{y}} homes<extra></extra>"))
    fig.update_yaxes(title="Homes")
    return style(fig, title=title, legend=False)


def scatter_vs_price(df: pd.DataFrame, x: str, title: str | None = None, highlight: dict | None = None,
                     trend: bool = True) -> go.Figure:
    fig = go.Figure(go.Scattergl(
        x=df[x], y=df["SalePrice"], mode="markers",
        marker=dict(size=7, color=df["SalePrice"], colorscale=TEAL_RAMP, opacity=0.75,
                    line=dict(color=SURFACE, width=0.5)),
        customdata=np.stack([df["Id"], df["Neighborhood"], df["OverallQual"]], axis=-1) if "Id" in df else None,
        hovertemplate=(f"{friendly(x)}: %{{x:,}}<br>Sale price: $%{{y:,.0f}}"
                       + ("<br>Id %{customdata[0]} · %{customdata[1]} · Q%{customdata[2]}" if "Id" in df else "")
                       + "<extra></extra>"),
        name="Historical sales"))
    if trend and len(df) > 2 and pd.api.types.is_numeric_dtype(df[x]):
        xs = df[x].astype(float)
        m = np.isfinite(xs)
        if m.sum() > 2:
            k = np.polyfit(xs[m], df["SalePrice"][m], 1)
            gx = np.linspace(xs[m].min(), xs[m].max(), 50)
            fig.add_trace(go.Scatter(x=gx, y=np.polyval(k, gx), mode="lines", line=dict(color=SERIES[1], width=2),
                                     name="Linear trend", hoverinfo="skip"))
    if highlight:
        fig.add_trace(go.Scatter(x=[highlight["x"]], y=[highlight["y"]], mode="markers+text",
                                 marker=dict(size=16, color=SERIES[2], line=dict(color="white", width=2)),
                                 text=[highlight.get("label", "Your property")], textposition="top center",
                                 textfont=dict(color=INK), name="Your property"))
    fig.update_xaxes(title=friendly(x))
    fig.update_yaxes(title="Sale price ($)", tickprefix="$", tickformat=",.0f")
    return style(fig, title=title or f"{friendly(x)} vs sale price")


def box_by(df: pd.DataFrame, col: str, title: str | None = None, order: list | None = None,
           height: int = 380) -> go.Figure:
    order = order or sorted(df[col].dropna().unique().tolist())
    fig = go.Figure()
    for v in order:
        s = df.loc[df[col] == v, "SalePrice"]
        fig.add_trace(go.Box(y=s, name=str(v), marker=dict(color=SERIES[0], size=3), line=dict(color=SERIES[0], width=1.5),
                             fillcolor="rgba(31,165,146,0.18)", boxpoints="outliers",
                             hovertemplate=f"{friendly(col)} {v}<br>$%{{y:,.0f}}<extra></extra>"))
    fig.update_yaxes(title="Sale price ($)", tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(title=friendly(col), type="category")
    return style(fig, title=title or f"{friendly(col)} vs sale price", legend=False, height=height)


def median_bar(df: pd.DataFrame, col: str, title: str, min_count: int = 1, horizontal: bool = True,
               height: int = 520) -> go.Figure:
    g = df.groupby(col, observed=True)["SalePrice"].agg(["median", "size"])
    g = g[g["size"] >= min_count].sort_values("median")
    if horizontal:
        fig = go.Figure(go.Bar(x=g["median"], y=g.index.astype(str), orientation="h",
                               marker=dict(color=g["median"], colorscale=TEAL_RAMP, cornerradius=4),
                               customdata=g["size"], hovertemplate="%{y}<br>Median $%{x:,.0f}<br>%{customdata} sales<extra></extra>"))
        fig.update_xaxes(title="Median sale price ($)", tickprefix="$", tickformat=",.0f")
    else:
        fig = go.Figure(go.Bar(x=g.index.astype(str), y=g["median"],
                               marker=dict(color=g["median"], colorscale=TEAL_RAMP, cornerradius=4),
                               customdata=g["size"], hovertemplate="%{x}<br>Median $%{y:,.0f}<br>%{customdata} sales<extra></extra>"))
        fig.update_yaxes(title="Median sale price ($)", tickprefix="$", tickformat=",.0f")
    return style(fig, title=title, legend=False, height=height)


def count_bar(series: pd.Series, title: str, height: int = 300) -> go.Figure:
    vc = series.value_counts().sort_index()
    fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker=dict(color=SERIES[1], cornerradius=4),
                           hovertemplate="%{x}: %{y} homes<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title="Homes")
    return style(fig, title=title, legend=False, height=height)


def correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation matrix") -> go.Figure:
    labels = [friendly(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1,
                               colorscale=[[0, DOWN], [0.5, "#1E2636"], [1, UP]],
                               hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
                               colorbar=dict(thickness=10, outlinewidth=0)))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return style(fig, title=title, legend=False, height=620)


def missing_bar(miss: pd.DataFrame, title: str = "Missing values by column") -> go.Figure:
    m = miss.sort_values("percent")
    fig = go.Figure(go.Bar(x=m["percent"], y=m.index, orientation="h", marker=dict(color=SERIES[2], cornerradius=4),
                           customdata=m["missing"], hovertemplate="%{y}: %{x:.1f}% (%{customdata} rows)<extra></extra>"))
    fig.update_xaxes(title="% of rows missing", ticksuffix="%")
    return style(fig, title=title, legend=False, height=max(300, 22 * len(m)))


# --- Model results -------------------------------------------------------------------------

def contribution_bar(expl_row: pd.DataFrame, top: int = 12, title: str | None = None) -> go.Figure:
    d = expl_row.head(top).iloc[::-1]
    colors = [UP if v >= 0 else DOWN for v in d["dollar_contribution"]]
    labels = [f"{friendly(f)} = {_fmt_val(v)}" for f, v in zip(d["feature"], d["value"])]
    fig = go.Figure(go.Bar(
        x=d["dollar_contribution"], y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        customdata=d["pct_effect"],
        hovertemplate="%{y}<br>≈ %{x:$,.0f} (%{customdata:+.1f}% multiplicative)<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
    fig.update_xaxes(title="Approximate contribution to the estimate ($)", tickprefix="$", tickformat=",.0f")
    return style(fig, title=title, legend=False, height=max(320, 34 * len(d) + 80))


def _fmt_val(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "—"
        return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.2f}"
    return str(v)


def waterfall(expl_row: pd.DataFrame, base_price: float, estimate: float, top: int = 10,
              title: str | None = "How the model built this estimate") -> go.Figure:
    d = expl_row.head(top)
    rest = expl_row["dollar_contribution"].iloc[top:].sum()
    xs = ["Average home"] + [f"{friendly(f)}" for f in d["feature"]] + ["All other features", "Estimate"]
    ys = [base_price] + d["dollar_contribution"].tolist() + [rest, estimate]
    measure = ["absolute"] + ["relative"] * (len(d) + 1) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=xs, y=ys, measure=measure, connector=dict(line=dict(color="rgba(255,255,255,0.15)", width=1)),
        increasing=dict(marker=dict(color=UP)), decreasing=dict(marker=dict(color=DOWN)),
        totals=dict(marker=dict(color=SERIES[0])),
        hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(tickangle=-35)
    return style(fig, title=title, legend=False, height=440)


def importance_bar(df: pd.DataFrame, value_col: str, title: str, top: int = 15, err_col: str | None = None,
                   xtitle: str = "") -> go.Figure:
    d = df.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d[value_col], y=[friendly(f) for f in d["feature"]], orientation="h",
        marker=dict(color=SERIES[0], cornerradius=4),
        error_x=dict(type="data", array=d[err_col], color=INK_3, thickness=1) if err_col else None,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title=xtitle)
    return style(fig, title=title, legend=False, height=max(320, 28 * len(d) + 80))


def shap_beeswarm(contrib: pd.DataFrame, values: pd.DataFrame, top: int = 15,
                  title: str = "SHAP summary (validation properties)") -> go.Figure:
    feats = contrib.abs().mean().sort_values(ascending=False).head(top).index[::-1]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for i, f in enumerate(feats):
        phi = contrib[f].to_numpy()
        v = pd.to_numeric(values[f], errors="coerce") if f in values else pd.Series(np.nan, index=contrib.index)
        vv = v.to_numpy(dtype=float)
        if np.all(np.isnan(vv)):
            norm = np.full(len(vv), 0.5)
        else:
            lo, hi = np.nanpercentile(vv, 5), np.nanpercentile(vv, 95)
            norm = np.clip((vv - lo) / (hi - lo + 1e-12), 0, 1)
            norm = np.where(np.isnan(norm), 0.5, norm)
        fig.add_trace(go.Scattergl(
            x=phi, y=i + rng.uniform(-0.28, 0.28, len(phi)), mode="markers",
            marker=dict(size=5, color=norm, colorscale=[[0, DOWN], [0.5, "#5B6475"], [1, UP]], cmin=0, cmax=1,
                        showscale=(i == len(feats) - 1),
                        colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["Low", "High"], thickness=10,
                                      outlinewidth=0)),
            customdata=values[f].astype(str).to_numpy() if f in values else None,
            hovertemplate=f"{friendly(f)} = %{{customdata}}<br>SHAP %{{x:+.3f}} log-price<extra></extra>",
            showlegend=False))
    fig.update_yaxes(tickvals=list(range(len(feats))), ticktext=[friendly(f) for f in feats])
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.25)", width=1))
    fig.update_xaxes(title="SHAP value (impact on log price)")
    return style(fig, title=title, legend=False, height=max(420, 30 * len(feats) + 90))


def pdp_chart(pdp: dict, title: str | None = None, current: float | None = None) -> go.Figure:
    fig = go.Figure()
    for curve in pdp.get("ice", [])[:30]:
        fig.add_trace(go.Scatter(x=pdp["grid"], y=curve, mode="lines", line=dict(color="rgba(57,135,229,0.18)", width=1),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=pdp["grid"], y=pdp["average"], mode="lines+markers", line=dict(color=SERIES[0], width=3),
                             marker=dict(size=6), name="Average model prediction",
                             hovertemplate=f"{friendly(pdp['feature'])} %{{x:,}}<br>%{{y:$,.0f}}<extra></extra>"))
    if current is not None:
        fig.add_vline(x=current, line=dict(color=SERIES[2], dash="dot"))
    fig.update_xaxes(title=friendly(pdp["feature"]))
    fig.update_yaxes(title="Predicted price ($)", tickprefix="$", tickformat=",.0f")
    return style(fig, title=title or f"Partial dependence: {friendly(pdp['feature'])}", height=360)


def price_surface_3d(surface: dict, point: dict | None = None, height: int = 560) -> go.Figure:
    z = np.asarray(surface["z"])
    fig = go.Figure(go.Surface(
        x=surface["grid1"], y=surface["grid2"], z=z, colorscale=TEAL_RAMP, opacity=0.95,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        colorbar=dict(title="Price", tickprefix="$", thickness=10, outlinewidth=0),
        hovertemplate="Living area %{x:,.0f} sq ft<br>Quality %{y}<br>Model price %{z:$,.0f}<extra></extra>"))
    if point:
        fig.add_trace(go.Scatter3d(x=[point["x"]], y=[point["y"]], z=[point["z"]], mode="markers+text",
                                   marker=dict(size=7, color=SERIES[2], line=dict(color="white", width=2)),
                                   text=["Your property"], textfont=dict(color=INK), name="Your property"))
    axis = dict(gridcolor="rgba(255,255,255,0.08)", backgroundcolor="rgba(0,0,0,0)", showbackground=True,
                zerolinecolor=GRID, color=INK_3)
    fig.update_layout(scene=dict(
        xaxis=dict(title="Living area (sq ft)", **axis), yaxis=dict(title="Overall quality", **axis),
        zaxis=dict(title="Price ($)", tickprefix="$", **axis),
        camera=dict(eye=dict(x=1.6, y=-1.5, z=0.9)), aspectmode="manual", aspectratio=dict(x=1.3, y=1, z=0.75)))
    return style(fig, height=height, legend=False)


def market_3d_scatter(df: pd.DataFrame, x: str = "GrLivArea", y: str = "OverallQual", height: int = 600,
                      point: dict | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter3d(
        x=df[x], y=df[y] + np.random.default_rng(1).uniform(-0.25, 0.25, len(df)), z=df["SalePrice"],
        mode="markers",
        marker=dict(size=3, color=df["SalePrice"], colorscale=TEAL_RAMP, opacity=0.85),
        customdata=np.stack([df["Neighborhood"], df["YearBuilt"]], axis=-1),
        hovertemplate=(f"{friendly(x)} %{{x:,}}<br>{friendly(y)} %{{y:.0f}}<br>$%{{z:,.0f}}"
                       "<br>%{customdata[0]} · built %{customdata[1]}<extra></extra>"), name="Sales"))
    if point:
        fig.add_trace(go.Scatter3d(x=[point["x"]], y=[point["y"]], z=[point["z"]], mode="markers",
                                   marker=dict(size=9, color=SERIES[2], line=dict(color="white", width=2)),
                                   name="Your property"))
    axis = dict(gridcolor="rgba(255,255,255,0.08)", backgroundcolor="rgba(0,0,0,0)", color=INK_3)
    fig.update_layout(scene=dict(xaxis=dict(title=friendly(x), **axis), yaxis=dict(title=friendly(y), **axis),
                                 zaxis=dict(title="Sale price", tickprefix="$", **axis),
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))))
    return style(fig, height=height, legend=False)


def model_comparison_bar(comp: pd.DataFrame, metric: str, title: str, fmt: str = ".4f",
                         lower_is_better: bool = True) -> go.Figure:
    d = comp.sort_values(metric, ascending=not lower_is_better)
    colors = [SERIES[0] if s else "#2A3A55" for s in d.get("Selected", [False] * len(d))]
    fig = go.Figure(go.Bar(x=d[metric], y=d["Model"], orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[format(v, fmt) for v in d[metric]], textposition="outside",
                           textfont=dict(color=INK_2), hovertemplate="%{y}: %{x:" + fmt + "}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, float(d[metric].max()) * 1.18])
    return style(fig, title=title, legend=False, height=60 + 36 * len(d))


def cv_folds_chart(comp_cv: dict[str, list[float]], title: str = "Cross-validation RMSLE by fold") -> go.Figure:
    fig = go.Figure()
    for i, (name, folds) in enumerate(comp_cv.items()):
        fig.add_trace(go.Box(y=folds, name=name, boxpoints="all", jitter=0.3, pointpos=0,
                             marker=dict(color=SERIES[i % len(SERIES)] if i < len(SERIES) else "#5B6475", size=5),
                             line=dict(width=1.5)))
    fig.update_yaxes(title="RMSLE (lower is better)")
    return style(fig, title=title, legend=False, height=420)


def actual_vs_predicted(actual, predicted, title: str = "Validation: actual vs predicted") -> go.Figure:
    a, p = np.asarray(actual), np.asarray(predicted)
    lim = [min(a.min(), p.min()) * 0.95, max(a.max(), p.max()) * 1.05]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
                             name="Perfect prediction", hoverinfo="skip"))
    fig.add_trace(go.Scattergl(x=a, y=p, mode="markers", name="Validation homes",
                               marker=dict(size=7, color=SERIES[0], opacity=0.75, line=dict(color=SURFACE, width=0.5)),
                               hovertemplate="Actual $%{x:,.0f}<br>Predicted $%{y:,.0f}<extra></extra>"))
    fig.update_xaxes(title="Actual sale price ($)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Predicted ($)", tickprefix="$", tickformat=",.0f")
    return style(fig, title=title, height=420)


def residual_chart(actual, predicted, title: str = "Residuals (predicted − actual)") -> go.Figure:
    a, p = np.asarray(actual), np.asarray(predicted)
    r = p - a
    fig = go.Figure(go.Scattergl(x=p, y=r, mode="markers", marker=dict(size=6, color=[UP if v >= 0 else DOWN for v in r], opacity=0.75),
                                 hovertemplate="Predicted $%{x:,.0f}<br>Residual %{y:$,.0f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)"))
    fig.update_xaxes(title="Predicted price ($)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Residual ($)", tickprefix="$", tickformat=",.0f")
    return style(fig, title=title, legend=False, height=380)


def radar(scores: pd.DataFrame, names: list[str], title: str = "Percentile profile vs all Ames sales") -> go.Figure:
    cats = list(scores.columns)
    fig = go.Figure()
    for i, n in enumerate(names):
        vals = scores.iloc[i].tolist()
        c = SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=n, fill="toself",
                                      line=dict(color=c, width=2), opacity=0.8,
                                      hovertemplate="%{theta}: %{r:.0f}th percentile<extra>" + n + "</extra>"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 100], gridcolor=GRID, color=INK_3, ticksuffix=""),
                                 angularaxis=dict(gridcolor=GRID, color=INK_2)))
    return style(fig, title=title, height=440)


def interval_chart(estimate: float, lower: float, upper: float, level: float, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lower, upper], y=[0, 0], mode="lines", line=dict(color=SERIES[0], width=14),
                             opacity=0.35, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[lower, estimate, upper], y=[0, 0, 0], mode="markers+text",
                             marker=dict(size=[10, 18, 10], color=[SERIES[0], "white", SERIES[0]]),
                             text=[money(lower, True), money(estimate, True), money(upper, True)],
                             textposition="top center", textfont=dict(color=INK, size=13), showlegend=False,
                             hovertemplate="%{x:$,.0f}<extra></extra>"))
    fig.update_yaxes(visible=False, range=[-1, 1.5])
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return style(fig, title=title or f"{level:.0%} prediction interval", legend=False, height=170)


def amortization_chart(sched: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sched["year"], y=sched["principal"], name="Principal", marker=dict(color=SERIES[0])))
    fig.add_trace(go.Bar(x=sched["year"], y=sched["interest"], name="Interest", marker=dict(color=SERIES[2])))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Paid per year ($)", tickprefix="$", tickformat=",.0f")
    return style(fig, title="Yearly payments: principal vs interest", height=360)


def balance_chart(sched: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=sched["year"], y=sched["balance"], mode="lines", fill="tozeroy",
                               line=dict(color=SERIES[1], width=2), fillcolor="rgba(57,135,229,0.15)",
                               hovertemplate="Year %{x}<br>Balance %{y:$,.0f}<extra></extra>"))
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Remaining balance ($)", tickprefix="$", tickformat=",.0f")
    return style(fig, title="Remaining loan balance", legend=False, height=360)
