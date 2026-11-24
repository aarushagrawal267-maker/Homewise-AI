"""Shared Streamlit UI: design system injection, cached resources and components."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from .config import ID_COL, PATHS, TARGET
from .data_dictionary import friendly
from .visualization import money

CSS_FILE = PATHS.root / "assets" / "styles" / "main.css"

NEIGHBORHOOD_NAMES = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale", "BrkSide": "Brookside",
    "ClearCr": "Clear Creek", "CollgCr": "College Creek", "Crawfor": "Crawford", "Edwards": "Edwards",
    "Gilbert": "Gilbert", "IDOTRR": "Iowa DOT & Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa", "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames", "OldTown": "Old Town", "SWISU": "South & West of Iowa State Univ.",
    "Sawyer": "Sawyer", "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook",
    "Timber": "Timberland", "Veenker": "Veenker",
}
BLDG_NAMES = {"1Fam": "Single-family detached", "2fmCon": "Two-family conversion", "Duplex": "Duplex",
              "TwnhsE": "Townhouse (end unit)", "Twnhs": "Townhouse (inside unit)"}
QUAL_NAMES = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical / average", "Fa": "Fair", "Po": "Poor"}
STYLE_NAMES = {"1Story": "One story", "1.5Fin": "1½ story (finished)", "1.5Unf": "1½ story (unfinished)",
               "2Story": "Two story", "2.5Fin": "2½ story (finished)", "2.5Unf": "2½ story (unfinished)",
               "SFoyer": "Split foyer", "SLvl": "Split level"}

ICONS = {
    "home": '<path d="M3 11.5 12 4l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
    "spark": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8"/>',
    "brain": '<path d="M9 4a3 3 0 0 0-3 3v.5A3 3 0 0 0 4 10v1a3 3 0 0 0 1 2.2V15a3 3 0 0 0 3 3h1V4z"/><path d="M15 4a3 3 0 0 1 3 3v.5a3 3 0 0 1 2 2.5v1a3 3 0 0 1-1 2.2V15a3 3 0 0 1-3 3h-1V4z"/>',
    "scale": '<path d="M12 3v18M5 7h14M5 7l-3 7h6zM19 7l-3 7h6z"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    "wand": '<path d="m4 20 11-11M14 4l1 2 2 1-2 1-1 2-1-2-2-1 2-1zM19 10l.6 1.4L21 12l-1.4.6L19 14l-.6-1.4L17 12l1.4-.6z"/>',
    "wallet": '<rect x="3" y="6" width="18" height="13" rx="2"/><path d="M16 12h2M3 9h18"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "flask": '<path d="M9 3h6M10 3v6L4 19a1 1 0 0 0 .9 1.5h14.2A1 1 0 0 0 20 19l-6-10V3"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.5 18a9 9 0 1 1 17 0"/>',
    "shield": '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "alert": '<path d="M12 4 2 20h20zM12 10v4M12 17v.5"/>',
    "cube": '<path d="M12 2 3 7v10l9 5 9-5V7z"/><path d="M3 7l9 5 9-5M12 12v10"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS.get(name, ICONS["spark"])}</svg>')


# ---------------------------------------------------------------- design system ---------

@st.cache_data(show_spinner=False)
def _css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def inject_css() -> None:
    st.markdown(f"<style>{_css()}</style>", unsafe_allow_html=True)


def md(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    md(f"""<div class="hw-page-head hw-anim"><span class="hw-eyebrow"><span class="hw-dot"></span>{eyebrow}</span>
    <div class="hw-page-title">{title}</div><p class="hw-page-sub">{subtitle}</p></div>""")


def section(title: str, subtitle: str = "") -> None:
    md(f'<div class="hw-section"><div><h3>{title}</h3>{f"<p>{subtitle}</p>" if subtitle else ""}</div></div>')


def kpis(items: list[tuple[str, str, str]], cols: int | None = None) -> None:
    """Stat tiles; wraps onto several rows when there are more items than ``cols``."""
    per_row = cols or len(items)
    for start in range(0, len(items), per_row):
        columns = st.columns(per_row)
        for c, (k, v, s) in zip(columns, items[start:start + per_row]):
            c.markdown(f'<div class="hw-kpi hw-anim"><div class="k">{k}</div><div class="v" title="{v}">{v}</div>'
                       f'<div class="s">{s}</div></div>', unsafe_allow_html=True)
        if start + per_row < len(items):
            st.write("")


def feature_card(icon_name: str, title: str, text: str) -> str:
    return (f'<div class="hw-card hw-anim"><div class="hw-icon">{icon(icon_name)}</div>'
            f'<h4>{title}</h4><p>{text}</p></div>')


def note(text: str, kind: str = "warn") -> None:
    ic = icon("alert" if kind == "warn" else "info", 18)
    md(f'<div class="hw-note {"info" if kind != "warn" else ""}">{ic}<div>{text}</div></div>')


def empty_state(title: str, text: str, icon_name: str = "cube") -> None:
    md(f'<div class="hw-empty"><div class="hw-icon" style="margin:0 auto">{icon(icon_name)}</div>'
       f'<h3>{title}</h3><p>{text}</p></div>')


def price_card(estimate: float, lower: float, upper: float, level: float, label: str = "Estimated value",
               caption: str = "") -> None:
    lo_axis, hi_axis = lower * 0.92, upper * 1.08
    span = hi_axis - lo_axis
    left = (lower - lo_axis) / span * 100
    width = (upper - lower) / span * 100
    pin = (estimate - lo_axis) / span * 100
    md(f"""<div class="hw-price hw-anim"><div class="lbl">{label}</div><div class="val">{money(estimate)}</div>
    <div class="rng">{level:.0%} prediction interval · {money(lower)} – {money(upper)}</div>
    <div class="hw-bar"><div class="fill" style="left:{left:.1f}%;width:{width:.1f}%"></div>
    <div class="pin" style="left:calc({pin:.1f}% - 2px)"></div></div>
    <div class="hw-bar-labels"><span>{money(lower, True)}</span><span>±{(upper - lower) / 2 / estimate:.0%}</span>
    <span>{money(upper, True)}</span></div>
    {f'<div class="rng" style="margin-top:12px;font-size:12.5px">{caption}</div>' if caption else ''}</div>""")


def contribution_chips(expl_row: pd.DataFrame, n: int = 4) -> str:
    pos = expl_row[expl_row["dollar_contribution"] > 0].head(n)
    neg = expl_row[expl_row["dollar_contribution"] < 0].head(n)

    def chip(r, cls):
        sign = "+" if r.dollar_contribution > 0 else "−"
        return (f'<span class="hw-chip {cls}">{html.escape(friendly(r.feature))}'
                f'<b>{sign}{money(abs(r.dollar_contribution), True)}</b></span>')
    up = "".join(chip(r, "up") for r in pos.itertuples())
    down = "".join(chip(r, "down") for r in neg.itertuples())
    return (f'<div style="margin:6px 0 2px;font-size:12px;color:var(--ink-3);letter-spacing:.08em;text-transform:uppercase">'
            f'Raises the estimate</div>{up or "<span class=hw-chip>None</span>"}'
            f'<div style="margin:12px 0 2px;font-size:12px;color:var(--ink-3);letter-spacing:.08em;text-transform:uppercase">'
            f'Lowers the estimate</div>{down or "<span class=hw-chip>None</span>"}')


def three(html_doc: str, height: int = 520) -> None:
    """Embed a self-contained Three.js scene.

    ``st.iframe`` replaces the deprecated ``st.components.v1.html`` in newer Streamlit releases; the old API is kept
    as a fallback so the app still runs on versions that predate ``st.iframe``. The HTML is generated by
    ``src/three_scenes.py`` from model outputs, never from raw user text.
    """
    if hasattr(st, "iframe"):
        st.iframe(html_doc, height=height)
    else:  # pragma: no cover - older Streamlit
        components.html(html_doc, height=height, scrolling=False)


def footer() -> None:
    md('<div class="hw-footer">HomeWise AI is a research and educational prototype trained on historical residential '
       'sales from Ames, Iowa (2006–2010, Kaggle “House Prices: Advanced Regression Techniques”). Estimates are '
       'model outputs, not appraisals, and do not reflect current market conditions or other locations.</div>')


# ------------------------------------------------------------------- cached resources ----

# Failures (missing dataset, failed download, unloadable model) are never cached: the cached
# helpers raise instead of returning a sentinel, so the next rerun tries again.

@st.cache_data(show_spinner=False)
def _read_train() -> pd.DataFrame:
    from .data_loader import load_train
    return load_train()


@st.cache_data(show_spinner=False)
def _read_test() -> pd.DataFrame:
    from .data_loader import load_test
    return load_test()


def load_train_df() -> pd.DataFrame | None:
    return _read_train() if ensure_dataset() else None


def load_test_df() -> pd.DataFrame | None:
    return _read_test() if ensure_dataset() else None


@st.cache_resource(show_spinner="Downloading the Kaggle House Prices dataset (first run only)…")
def _download_dataset_once() -> bool:
    from .data_loader import acquire_dataset
    acquire_dataset(verbose=False)  # raises on failure, so a failed attempt is not cached
    return True


def ensure_dataset() -> bool:
    """On a fresh server (e.g. a cloud deployment) fetch the checksum-verified CSVs once."""
    from .data_loader import datasets_available
    if datasets_available():
        return True
    try:
        _download_dataset_once()
    except Exception:
        return False
    return datasets_available()


@st.cache_resource(show_spinner="Loading the trained model…")
def _load_engine_resource():
    from .prediction import PredictionEngine
    return PredictionEngine.load()  # raises ModelLoadError on a bad file (not cached)


def engine_status():
    """Return ``(engine, problem)``: the engine, or None plus a readable reason it is unavailable."""
    from .prediction import ModelLoadError
    if not PATHS.model_bundle.exists():
        return None, None
    try:
        return _load_engine_resource(), None
    except ModelLoadError as err:
        return None, str(err)


def load_engine_cached():
    return engine_status()[0]


@st.cache_resource(show_spinner="Preparing the explainer…")
def _explainer_resource():
    from .explainability import HomeWiseExplainer
    eng = _load_engine_resource()
    return HomeWiseExplainer(eng.model, eng.bundle["background"])


def load_explainer_cached():
    return None if load_engine_cached() is None else _explainer_resource()


@st.cache_resource(show_spinner=False)
def _finder_resource():
    from .similar_properties import SimilarPropertyFinder
    return SimilarPropertyFinder(_read_train())


def load_finder_cached():
    return _finder_resource() if ensure_dataset() else None


@st.cache_data(show_spinner=False)
def report_json(name: str):
    p = PATHS.reports / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def report_csv(name: str) -> pd.DataFrame | None:
    p = PATHS.reports / name
    return pd.read_csv(p) if p.exists() else None


def require_data() -> pd.DataFrame:
    df = load_train_df()
    if df is None:
        empty_state("Dataset not found",
                    "Run <code>python scripts/download_data.py</code> or download train.csv and test.csv from the "
                    "Kaggle House Prices competition into <code>data/raw/</code>.")
        st.stop()
    return df


def require_engine():
    eng, problem = engine_status()
    if problem:
        empty_state("The saved model could not be loaded", html.escape(problem), "flask")
        st.stop()
    if eng is None:
        empty_state("No trained model yet",
                    "Train the models once with <code>python scripts/train_all.py</code>. The app loads the saved "
                    "model afterwards and never retrains on refresh.", "flask")
        st.stop()
    return eng


# --------------------------------------------------------------- property input ----------

DEFAULT_KEYS = ["Neighborhood", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "KitchenQual", "ExterQual",
                "YearBuilt", "YearRemodAdd", "YrSold", "GrLivArea", "LotArea", "TotalBsmtSF", "BedroomAbvGr",
                "FullBath", "HalfBath", "KitchenAbvGr", "TotRmsAbvGrd", "GarageCars", "GarageArea", "Fireplaces",
                "HasCentralAir", "HasPool"]


def inputs_from_record(rec: pd.Series) -> dict:
    def num(k, d=0):
        v = rec.get(k)
        return d if v is None or (isinstance(v, float) and np.isnan(v)) else v
    return {
        "Neighborhood": str(num("Neighborhood", "NAmes")), "BldgType": str(num("BldgType", "1Fam")),
        "HouseStyle": str(num("HouseStyle", "1Story")),
        "OverallQual": int(num("OverallQual", 6)), "OverallCond": int(num("OverallCond", 5)),
        "KitchenQual": str(num("KitchenQual", "TA")), "ExterQual": str(num("ExterQual", "TA")),
        "YearBuilt": int(num("YearBuilt", 1975)), "YearRemodAdd": int(num("YearRemodAdd", num("YearBuilt", 1975))),
        "YrSold": int(num("YrSold", 2010)), "GrLivArea": float(num("GrLivArea", 1500)),
        "LotArea": float(num("LotArea", 9500)), "TotalBsmtSF": float(num("TotalBsmtSF", 0)),
        "BedroomAbvGr": int(num("BedroomAbvGr", 3)), "FullBath": int(num("FullBath", 2)),
        "HalfBath": int(num("HalfBath", 0)), "KitchenAbvGr": int(num("KitchenAbvGr", 1)),
        "TotRmsAbvGrd": int(num("TotRmsAbvGrd", 6)), "GarageCars": int(num("GarageCars", 0)),
        "GarageArea": float(num("GarageArea", 0)), "Fireplaces": int(num("Fireplaces", 0)),
        "HasCentralAir": str(num("CentralAir", "Y")) == "Y", "HasPool": float(num("PoolArea", 0)) > 0,
    }


def dataset_record(df: pd.DataFrame, pid: int) -> pd.Series:
    return df.loc[df[ID_COL] == pid].iloc[0]


def current_property(engine) -> tuple[dict, pd.Series | None, str]:
    """The property most recently valued in this session (or the reference profile)."""
    df = load_train_df()
    inputs = st.session_state.get("hw_inputs")
    base_id = st.session_state.get("hw_base_id")
    base = None
    if base_id is not None and df is not None:
        base = dataset_record(df, base_id).drop(labels=[ID_COL, TARGET])
    if inputs is None:
        tmpl = engine.template
        inputs = inputs_from_record(tmpl)
        tid = engine.bundle.get("template_id")
        label = f"Representative Ames home #{tid}" if tid else "Reference profile"
    else:
        label = st.session_state.get("hw_label", "Your property")
    return inputs, base, label


def property_form(engine, defaults: dict, key: str, submit_label: str = "Estimate value",
                  compact: bool = False) -> dict | None:
    """Render the property form; returns the input dict when submitted."""
    opts = engine.category_options
    d = defaults
    with st.form(key, border=True):
        st.markdown("**Location & type**")
        c1, c2, c3 = st.columns(3)
        nb_opts = opts["Neighborhood"]
        nb = c1.selectbox("Neighborhood", nb_opts, index=nb_opts.index(d["Neighborhood"]) if d["Neighborhood"] in nb_opts else 0,
                          format_func=lambda v: f"{NEIGHBORHOOD_NAMES.get(v, v)}")
        bt_opts = opts["BldgType"]
        bt = c2.selectbox("Building type", bt_opts, index=bt_opts.index(d["BldgType"]) if d["BldgType"] in bt_opts else 0,
                          format_func=lambda v: BLDG_NAMES.get(v, v))
        hs_opts = opts["HouseStyle"]
        hs = c3.selectbox("House style", hs_opts, index=hs_opts.index(d["HouseStyle"]) if d["HouseStyle"] in hs_opts else 0,
                          format_func=lambda v: STYLE_NAMES.get(v, v))

        st.markdown("**Quality & age**")
        c1, c2, c3, c4 = st.columns(4)
        oq = c1.slider("Overall quality", 1, 10, int(d["OverallQual"]), help="Material and finish quality, 1 = very poor, 10 = very excellent")
        oc = c2.slider("Overall condition", 1, 10, int(d["OverallCond"]))
        q_opts = [q for q in ["Ex", "Gd", "TA", "Fa", "Po"] if q in opts["KitchenQual"]]
        kq = c3.selectbox("Kitchen quality", q_opts, index=q_opts.index(d["KitchenQual"]) if d["KitchenQual"] in q_opts else 2,
                          format_func=lambda v: QUAL_NAMES.get(v, v))
        eq_opts = [q for q in ["Ex", "Gd", "TA", "Fa", "Po"] if q in opts["ExterQual"]]
        eq = c4.selectbox("Exterior quality", eq_opts, index=eq_opts.index(d["ExterQual"]) if d["ExterQual"] in eq_opts else 1,
                          format_func=lambda v: QUAL_NAMES.get(v, v))
        c1, c2, c3 = st.columns(3)
        yb = c1.number_input("Year built", 1850, 2010, int(d["YearBuilt"]), step=1)
        yr = c2.number_input("Year remodeled", 1850, 2010, int(max(d["YearRemodAdd"], d["YearBuilt"])), step=1,
                             help="Same as year built if never remodeled")
        ys = c3.number_input("Year of sale (2006–2010)", 2006, 2010, int(d["YrSold"]), step=1,
                             help="The model only knows the 2006–2010 market")

        st.markdown("**Size**")
        c1, c2, c3 = st.columns(3)
        gla = c1.number_input("Living area (sq ft)", 300, 8000, int(d["GrLivArea"]), step=50)
        lot = c2.number_input("Lot area (sq ft)", 1000, 250000, int(d["LotArea"]), step=250)
        bsmt = c3.number_input("Basement area (sq ft)", 0, 7000, int(d["TotalBsmtSF"]), step=50, help="0 = no basement")

        st.markdown("**Rooms**")
        c1, c2, c3, c4, c5 = st.columns(5)
        bed = c1.number_input("Bedrooms", 0, 10, int(d["BedroomAbvGr"]))
        fb = c2.number_input("Full baths", 0, 5, int(d["FullBath"]))
        hb = c3.number_input("Half baths", 0, 3, int(d["HalfBath"]))
        kit = c4.number_input("Kitchens", 0, 3, int(d["KitchenAbvGr"]))
        rooms = c5.number_input("Total rooms", 1, 16, int(d["TotRmsAbvGrd"]), help="Above grade, excluding bathrooms")

        st.markdown("**Garage & amenities**")
        c1, c2, c3 = st.columns(3)
        gc = c1.number_input("Garage (cars)", 0, 5, int(d["GarageCars"]))
        ga = c2.number_input("Garage area (sq ft)", 0, 1500, int(d["GarageArea"]), step=10, help="Set 0 to auto-size from capacity")
        fp = c3.number_input("Fireplaces", 0, 4, int(d["Fireplaces"]))
        c1, c2, _ = st.columns(3)
        ca = c1.toggle("Central air conditioning", bool(d["HasCentralAir"]))
        pool = c2.toggle("Swimming pool", bool(d["HasPool"]))
        submitted = st.form_submit_button(submit_label, type="primary", width="stretch")
    if not submitted:
        return None
    out = {"Neighborhood": nb, "BldgType": bt, "HouseStyle": hs, "OverallQual": oq, "OverallCond": oc,
           "KitchenQual": kq, "ExterQual": eq, "YearBuilt": int(yb), "YearRemodAdd": int(yr), "YrSold": int(ys),
           "GrLivArea": float(gla), "LotArea": float(lot), "TotalBsmtSF": float(bsmt), "BedroomAbvGr": int(bed),
           "FullBath": int(fb), "HalfBath": int(hb), "KitchenAbvGr": int(kit), "TotRmsAbvGrd": int(rooms),
           "GarageCars": int(gc), "Fireplaces": int(fp), "HasCentralAir": ca, "HasPool": pool}
    if gc > 0 and ga > 0:
        out["GarageArea"] = float(ga)
    elif gc == 0:
        out["GarageArea"] = 0.0
    return out


def show_input_errors(err) -> None:
    items = "".join(f"<li>{html.escape(e)}</li>" for e in err.errors)
    md(f'<div class="hw-note">{icon("alert", 18)}<div><b>Please fix the following</b><ul style="margin:6px 0 0 18px;padding:0">{items}</ul></div></div>')


def house_sub(rec: pd.Series) -> str:
    gla = rec.get("GrLivArea")
    return (f"{float(gla):,.0f} sq ft · Q{int(rec.get('OverallQual'))} · "
            f"{int(rec.get('GarageCars') or 0)}-car · {NEIGHBORHOOD_NAMES.get(rec.get('Neighborhood'), rec.get('Neighborhood'))}")


def property_picker(df: pd.DataFrame, key: str, label: str = "Pick a home from the dataset") -> int:
    ids = df[ID_COL].tolist()
    lookup = df.set_index(ID_COL)

    def fmt(i):
        r = lookup.loc[i]
        return (f"#{i} · {NEIGHBORHOOD_NAMES.get(r.Neighborhood, r.Neighborhood)} · Q{r.OverallQual} · "
                f"{r.GrLivArea:,} sq ft · sold ${r.SalePrice:,.0f}")
    return st.selectbox(label, ids, key=key, format_func=fmt)


def safe_path(p: Path) -> str:
    try:
        return str(p.relative_to(PATHS.root))
    except ValueError:
        return str(p)
