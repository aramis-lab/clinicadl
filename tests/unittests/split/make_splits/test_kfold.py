import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from clinicadl.split.make_splits import make_kfold

TSV_DIR = Path(__file__).parents[2] / "resources" / "tsv"
DF = pd.read_csv(TSV_DIR / "test_df.tsv", sep="\t")


def test_good_split(tmp_path):
    split_dir = make_kfold(
        TSV_DIR / "test_df.tsv",
        output_dir=tmp_path,
        subset_name="val",
        stratification="diagnosis",
        n_splits=3,
        seed=0,
    )

    assert split_dir == tmp_path / "3_fold"
    assert (split_dir / "split-0").is_dir()
    assert (split_dir / "split-1").is_dir()
    assert (split_dir / "split-2").is_dir()
    assert not (split_dir / "split-3").is_dir()

    assert (split_dir / "kfold_config.json").is_file
    with (split_dir / "kfold_config.json").open(mode="r") as file:
        dict_ = json.load(file)
    assert dict_["subset_name"] == "val"
    assert dict_["stratification"] == "diagnosis"
    assert dict_["longitudinal"] is False
    assert dict_["n_splits"] == 3
    assert dict_["seed"] == 0

    val_df = pd.read_csv(split_dir / "split-0" / "val_baseline.tsv", sep="\t")
    assert len(val_df) == 14
    assert set(val_df.columns) == {"participant_id", "session_id", "diagnosis"}
    assert val_df.iloc[11][["participant_id", "session_id"]].to_list() == [
        "sub-067",
        "ses-M006",
    ]

    val_sets = [
        set(
            pd.read_csv(
                split_dir / f"split-{split}" / "val_baseline.tsv", sep="\t"
            ).itertuples(index=False)
        )
        for split in range(3)
    ]
    assert all(val_sets[0].isdisjoint(val_set) for val_set in val_sets[1:])

    train_baseline_df = pd.read_csv(
        split_dir / "split-0" / "train_baseline.tsv", sep="\t"
    )
    assert len(train_baseline_df) == 26
    assert set(train_baseline_df.columns) == {
        "participant_id",
        "session_id",
        "diagnosis",
    }
    assert train_baseline_df.iloc[7][["participant_id", "session_id"]].to_list() == [
        "sub-025",
        "ses-M006",
    ]

    train_df = pd.read_csv(split_dir / "split-0" / "train.tsv", sep="\t")
    assert len(train_df) == 40
    assert set(train_df.columns) == {"participant_id", "session_id", "diagnosis"}
    assert train_df.iloc[22][["participant_id", "session_id"]].to_list() == [
        "sub-046",
        "ses-M006",
    ]

    # test other args
    split_dir = make_kfold(
        DF,
        output_dir=tmp_path,
        subset_name="val",
        stratification=True,
        longitudinal=True,
        n_splits=5,
        seed=1,
    )

    assert split_dir == tmp_path / "5_fold"
    val_baseline_df = pd.read_csv(split_dir / "split-0" / "val_baseline.tsv", sep="\t")
    assert len(val_baseline_df) == 8
    assert set(val_baseline_df.columns) == {
        "participant_id",
        "session_id",
        "sex",
    }
    val_df = pd.read_csv(split_dir / "split-4" / "val.tsv", sep="\t")
    assert len(val_df) == 13
    assert set(val_df.columns) == {"participant_id", "session_id", "sex"}

    # test other args
    shutil.copy(TSV_DIR / "test_df.tsv", tmp_path / "test_df.tsv")
    split_dir = make_kfold(
        tmp_path / "test_df.tsv",
        output_dir=None,
        stratification=False,
        n_splits=4,
    )
    assert split_dir == tmp_path / "4_fold"
    assert (split_dir / "split-3" / "validation_baseline.tsv").exists()


def test_special_cases(tmp_path):
    # no output dir
    with pytest.raises(ValueError, match="You must specify the output directory."):
        make_kfold(
            DF,
            output_dir=None,
        )

    # number of splits
    with pytest.raises(ValidationError, match="'n_splits' must be at least 2."):
        make_kfold(
            DF,
            output_dir=tmp_path,
            stratification=True,
            n_splits=1,
        )

    # stratification
    with pytest.raises(
        ValueError,
        match="Stratification can only be performed on a single column for K-Fold splitting.*",
    ):
        make_kfold(
            DF,
            output_dir=tmp_path,
            stratification=["age", "sex"],
        )

    with pytest.raises(
        KeyError, match="Stratification column 'abc' not found in the dataset."
    ):
        make_kfold(
            DF,
            output_dir=tmp_path,
            stratification="abc",
        )

    with pytest.raises(
        ValueError,
        match="Continuous variables cannot be used for stratification in K-Fold splitting.",
    ):
        make_kfold(
            DF,
            output_dir=tmp_path,
            stratification="age",
        )
