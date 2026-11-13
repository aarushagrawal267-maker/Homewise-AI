"""Renovation simulator, property comparison, similar-property finder, 3D scenes, experiment log."""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import ID_COL, TARGET
from src.experiment_tracker import load_experiments, log_experiment
from src.property_comparison import comparison_table, explanation_difference, price_differences, radar_scores
from src.renovation_simulator import scenarios_frame, sensitivity_curve, simulate, simulate_many
from src.similar_properties import SimilarPropertyFinder
from src.three_scenes import house_scene_html, house_spec, market_galaxy_html


def test_scenario_difference(engine, valid_inputs):
    r = simulate(engine, valid_inputs, {"OverallQual": 9}, "Quality")
    assert r.difference == pytest.approx(r.modified_price - r.original_price)
    assert r.percent == pytest.approx(r.difference / r.original_price * 100)
    assert r.modified_price > r.original_price  # ridge learned a positive quality association
    assert "OverallQual" in r.changed_fields["field"].tolist()


def test_scenario_requires_change(engine, valid_inputs):
    with pytest.raises(ValueError):
        simulate(engine, valid_inputs, {}, "none")


def test_multiple_scenarios(engine, valid_inputs):
    res = simulate_many(engine, valid_inputs, {"A": {"GrLivArea": 2200}, "B": {"GarageCars": 3}, "C": {}})
    assert [r.name for r in res] == ["A", "B"]
    df = scenarios_frame(res)
    assert len(df) == 2 and np.isfinite(df["Difference"]).all()
    a = res[0].changed_fields.set_index("field")
    assert {"GrLivArea", "1stFlrSF"} <= set(a.index)  # consistency rules applied


def test_sensitivity_curve(engine, valid_inputs):
    c = sensitivity_curve(engine, valid_inputs, "OverallQual", [3, 5, 7, 9])
    assert c["estimate"].is_monotonic_increasing


def test_similar_property_retrieval(train_df):
    f = SimilarPropertyFinder(train_df)
    rec = train_df.iloc[[10]].drop(columns=[ID_COL, TARGET])
    res = f.find(rec, k=5)
    assert len(res) == 5 and res["distance"].is_monotonic_increasing
    assert res.iloc[0][ID_COL] == train_df.iloc[10][ID_COL]  # a home is its own nearest neighbour
    excl = f.find(rec, k=5, exclude_id=int(train_df.iloc[10][ID_COL]))
    assert int(train_df.iloc[10][ID_COL]) not in excl[ID_COL].tolist()
    assert (excl["similarity"] <= 100).all() and (excl["similarity"] > 0).all()
    assert len(f.distinguishing_features(rec, excl)) > 0


def test_property_comparison(engine, train_df):
    a = train_df.iloc[[0]].drop(columns=[ID_COL, TARGET]).reset_index(drop=True)
    b = train_df.iloc[[1]].drop(columns=[ID_COL, TARGET]).reset_index(drop=True)
    ests = [float(engine.predict_frame(a)[0]), float(engine.predict_frame(b)[0])]
    t = comparison_table([a, b], ["A", "B"], ests, [208500.0, 181500.0])
    assert t.loc["Living area (sq ft)", "A"] == 1710
    d = price_differences(["A", "B"], ests)
    assert d.iloc[0]["difference"] == pytest.approx(ests[1] - ests[0])
    rs = radar_scores([a, b], train_df)
    assert rs.shape[0] == 2 and ((rs >= 0) & (rs <= 100)).all().all()
    diff = explanation_difference(pd.Series({"x": 0.1, "y": 0.0}), pd.Series({"x": 0.3, "y": -0.1}))
    assert diff.iloc[0]["feature"] == "x"
    with pytest.raises(ValueError):
        comparison_table([a], ["A"], ests[:1])


def test_3d_scenes_render_valid_html(train_df):
    spec = house_spec(train_df.iloc[0], "A", "$200,000", "sub")
    assert spec["garage_cars"] == 2 and spec["second_sf"] == 854
    html = house_scene_html([spec])
    assert "<script type='module'>" in html and "loadThree" in html and '"first_sf": 856.0' in html
    gal = market_galaxy_html(train_df.head(50), highlight={"GrLivArea": 1500, "YearBuilt": 2000, "price": 2e5})
    assert gal.count("Id ") >= 50 and "</script>" in gal


def test_experiment_log_is_append_only(tmp_path):
    p = tmp_path / "exp.jsonl"
    log_experiment({"model": "ridge", "params": {"alpha": 1.0}}, p)
    log_experiment({"model": "lasso", "params": {"alpha": np.float64(0.001)}}, p)
    rows = load_experiments(p)
    assert [r["model"] for r in rows] == ["ridge", "lasso"]
    assert rows[0]["experiment_id"] != rows[1]["experiment_id"]
    assert all(json.loads(line) for line in p.read_text().splitlines())
