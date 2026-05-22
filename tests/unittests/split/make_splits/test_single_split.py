import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clinicadl.split.make_splits import make_split

TSV_DIR = Path(__file__).parents[2] / "resources" / "tsv"
DF = pd.read_csv(TSV_DIR / "test_df.tsv", sep="\t")


def test_good_split(tmp_path):
    stratification = ["age", "sex", "test", "diagnosis"]
    split_dir = make_split(
        TSV_DIR / "test_df.tsv",
        output_dir=tmp_path,
        subset_name="test",
        stratification=stratification,
        p_categorical_threshold=0.9,
        p_continuous_threshold=0.9,
        n_test=15,
        seed=0,
    )

    assert split_dir == tmp_path / "split"
    assert (split_dir / "single_split_config.json").is_file
    with (split_dir / "single_split_config.json").open(mode="r") as file:
        dict_ = json.load(file)
    assert dict_["subset_name"] == "test"
    assert dict_["stratification"] == stratification
    assert dict_["longitudinal"] is False
    assert dict_["n_test"] == 15
    assert dict_["seed"] == 0
    assert np.isclose(dict_["p_categorical_threshold"], 0.9, rtol=1e-09, atol=1e-09)
    assert np.isclose(dict_["p_continuous_threshold"], 0.9, rtol=1e-09, atol=1e-09)

    test_df = pd.read_csv(split_dir / "test_baseline.tsv", sep="\t")
    assert len(test_df) == 15
    assert set(test_df.columns) == set(
        ["participant_id", "session_id"] + stratification
    )
    assert test_df.iloc[11][["participant_id", "session_id"]].to_list() == [
        "sub-076",
        "ses-M006",
    ]

    train_baseline_df = pd.read_csv(split_dir / "train_baseline.tsv", sep="\t")
    assert len(train_baseline_df) == 25
    assert set(train_baseline_df.columns) == set(
        ["participant_id", "session_id"] + stratification
    )
    assert train_baseline_df.iloc[7][["participant_id", "session_id"]].to_list() == [
        "sub-013",
        "ses-M000",
    ]

    train_df = pd.read_csv(split_dir / "train.tsv", sep="\t")
    assert len(train_df) == 39
    assert set(train_df.columns) == set(
        ["participant_id", "session_id"] + stratification
    )
    assert train_df.iloc[22][["participant_id", "session_id"]].to_list() == [
        "sub-045",
        "ses-M018",
    ]

    continuous_stats = pd.read_csv(split_dir / "split_continuous_stats.tsv", sep="\t")
    categorical_stats = pd.read_csv(split_dir / "split_categorical_stats.tsv", sep="\t")
    ref_continuous_stats = pd.read_csv(TSV_DIR / "ref_continuous_stats.tsv", sep="\t")
    ref_categorical_stats = pd.read_csv(TSV_DIR / "ref_categorical_stats.tsv", sep="\t")

    assert (categorical_stats == ref_categorical_stats).all().all()
    assert (continuous_stats == ref_continuous_stats).all().all()

    # test other args
    split_dir = make_split(
        DF,
        output_dir=tmp_path,
        subset_name="val",
        stratification=True,
        longitudinal=True,
        n_test=0.25,
        seed=1,
    )

    assert split_dir == tmp_path / "split_2"
    val_baseline_df = pd.read_csv(split_dir / "val_baseline.tsv", sep="\t")
    assert len(val_baseline_df) == 10
    assert set(val_baseline_df.columns) == {
        "participant_id",
        "session_id",
        "age",
        "sex",
    }
    val_df = pd.read_csv(split_dir / "val.tsv", sep="\t")
    assert len(val_df) == 16
    assert set(val_df.columns) == {
        "participant_id",
        "session_id",
        "age",
        "sex",
    }

    # test other args
    split_dir = make_split(
        split_dir / "train.tsv",
        output_dir=None,
        stratification=False,
    )
    assert split_dir == tmp_path / "split_2" / "split"
    assert (split_dir / "test_baseline.tsv").exists()
    assert not (split_dir / "split_continuous_stats.tsv").exists()
    assert not (split_dir / "split_categorical_stats.tsv").exists()


def test_special_cases(tmp_path):
    # no output dir
    with pytest.raises(
        ValueError,
        match="If you pass a DataFrame, you must specify the output directory.",
    ):
        make_split(
            DF,
            output_dir=None,
        )

    # bad thresholds
    with pytest.raises(
        ValueError, match="'p_categorical_threshold' must be between 0 and 1*"
    ):
        make_split(
            DF,
            output_dir=tmp_path,
            stratification=True,
            p_categorical_threshold=1.1,
            n_test=3,
        )

    # n_test=0
    split_dir = make_split(
        DF,
        output_dir=tmp_path,
        stratification=True,
        n_test=0,
    )
    assert (
        (
            pd.read_csv(split_dir / "train.tsv", sep="\t")
            == DF[["participant_id", "session_id", "age", "sex"]]
        )
        .all()
        .all()
    )
    assert (
        (
            pd.read_csv(split_dir / "test_baseline.tsv", sep="\t")
            == pd.DataFrame(columns=["participant_id", "session_id", "age", "sex"])
        )
        .all()
        .all()
    )

    # not enough tries
    with pytest.raises(RuntimeError, match="Unable to find a valid split after*"):
        split_dir = make_split(
            DF,
            output_dir=tmp_path,
            stratification=True,
            n_test=10,
            p_categorical_threshold=1.0,
            p_continuous_threshold=1.0,
            n_try_max=10,
        )

    # stratification columns
    with pytest.raises(
        KeyError, match="Invalid stratification columns (not found in the dataframe)*"
    ):
        split_dir = make_split(
            DF,
            output_dir=tmp_path,
            stratification=["abc"],
        )
