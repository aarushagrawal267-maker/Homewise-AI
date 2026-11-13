"""Dataset loading and validation."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DatasetMissingError, dataset_status, load_train, split_features_target
from src.data_validation import (
    EXPECTED_FEATURES, identify_feature_types, missing_summary, validate_dataset, validate_train_test,
)
from src.eda import automatic_insights, build_eda_summary, target_statistics


def test_csv_loading_shapes(train_df, test_df):
    assert train_df.shape == (1460, 81)
    assert test_df.shape == (1459, 80)
    assert "SalePrice" in train_df and "SalePrice" not in test_df


def test_checksums_match_reference(train_df):
    st = dataset_status()
    assert st["train.csv"]["checksum_ok"] and st["test.csv"]["checksum_ok"]


def test_column_validation(train_df):
    assert len(EXPECTED_FEATURES) == 79
    rep = validate_dataset(train_df, "train", expect_target=True)
    assert rep.ok, rep.errors
    assert rep.missing_expected_columns == []
    broken = validate_dataset(train_df.drop(columns=["GrLivArea"]), "train", expect_target=True)
    assert not broken.ok and "GrLivArea" in broken.missing_expected_columns


def test_target_validation(train_df):
    rep = validate_dataset(train_df, "train", expect_target=True)
    assert rep.target_summary["min"] > 0
    bad = train_df.copy()
    bad.loc[0, "SalePrice"] = -5
    assert not validate_dataset(bad, "train", expect_target=True).ok
    missing = train_df.drop(columns=["SalePrice"])
    assert not validate_dataset(missing, "train", expect_target=True).ok


def test_missing_value_detection(train_df):
    miss = missing_summary(train_df)
    assert miss.loc["PoolQC", "missing"] == 1453
    assert miss.loc["LotFrontage", "missing"] == 259
    assert "GrLivArea" not in miss.index


def test_duplicate_detection(train_df):
    rep = validate_dataset(train_df, "train", expect_target=True)
    assert rep.duplicate_rows == 0 and rep.duplicate_ids == 0
    dup = pd.concat([train_df, train_df.iloc[[0]].assign(Id=99999)], ignore_index=True)
    rep2 = validate_dataset(dup, "dup", expect_target=True)
    assert rep2.duplicate_rows == 1
    dup_id = pd.concat([train_df, train_df.iloc[[0]]], ignore_index=True)
    assert validate_dataset(dup_id, "dup", expect_target=True).duplicate_ids == 1


def test_feature_type_identification(train_df):
    num, cat = identify_feature_types(train_df)
    assert "MSSubClass" in cat, "numerically coded category must be categorical"
    assert "GrLivArea" in num and "Neighborhood" in cat
    assert "Id" not in num and "SalePrice" not in num
    assert len(num) + len(cat) == 79


def test_train_test_structure(train_df, test_df):
    rep = validate_train_test(train_df, test_df)
    assert rep["schema_match"] and rep["id_overlap"] == 0 and rep["ok"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetMissingError):
        load_train(tmp_path / "nope.csv")


def test_split_features_target(train_df):
    X, y = split_features_target(train_df)
    assert "Id" not in X and "SalePrice" not in X and len(y) == 1460


def test_eda_is_computed_from_data(train_df):
    s = target_statistics(train_df["SalePrice"])
    assert s["median"] == pytest.approx(163000)
    assert s["mean"] == pytest.approx(train_df["SalePrice"].mean())
    ins = automatic_insights(train_df)
    assert len(ins) >= 6 and all(np.isfinite(i["metric"]) for i in ins)
    summ = build_eda_summary(train_df)
    assert summ["overview"]["train_records"] == 1460
