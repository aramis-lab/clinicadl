import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from clinicadl.splitter.make_splits import make_kfold


def remove_non_empty_dir(dir_path: Path):
    """
    Remove a non-empty directory using only pathlib.

    Parameters
    ----------
    dir_path : Path
        Path to the directory to remove.
    """
    if dir_path.exists() and dir_path.is_dir():
        for item in dir_path.iterdir():  # Iterate through directory contents
            if item.is_dir():
                remove_non_empty_dir(item)  # Recursively remove subdirectories
            else:
                item.unlink()  # Remove files
        dir_path.rmdir()  # Remove the now-empty directory
    else:
        print(f"{dir_path} does not exist or is not a directory.")


CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
TMP_DIR = Path(__file__).parents[2] / "resources" / "tmp"

DF_PATH = CAPS_DIR / "tsv" / "test_df.tsv"
DF = pd.read_csv(DF_PATH, sep="\t")


def test_good_split():
    split_dir = make_kfold(
        DF_PATH,
        output_dir=TMP_DIR,
        subset_name="val",
        stratification="diagnosis",
        n_splits=3,
        seed=0,
    )

    assert split_dir == TMP_DIR / "3_fold"
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
        output_dir=TMP_DIR,
        subset_name="val",
        stratification=True,
        longitudinal=True,
        n_splits=5,
        seed=1,
    )

    assert split_dir == TMP_DIR / "5_fold"
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
    shutil.copy(DF_PATH, TMP_DIR / "test_df.tsv")
    split_dir = make_kfold(
        TMP_DIR / "test_df.tsv",
        output_dir=None,
        stratification=False,
        n_splits=4,
    )
    assert split_dir == TMP_DIR / "4_fold"
    assert (split_dir / "split-3" / "validation_baseline.tsv").exists()

    remove_non_empty_dir(TMP_DIR)


def test_special_cases():
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
            output_dir=TMP_DIR,
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
            output_dir=TMP_DIR,
            stratification=["age", "sex"],
        )

    with pytest.raises(
        KeyError, match="Stratification column 'abc' not found in the dataset."
    ):
        make_kfold(
            DF,
            output_dir=TMP_DIR,
            stratification="abc",
        )

    with pytest.raises(
        ValueError,
        match="Continuous variables cannot be used for stratification in K-Fold splitting.",
    ):
        make_kfold(
            DF,
            output_dir=TMP_DIR,
            stratification="age",
        )

    remove_non_empty_dir(TMP_DIR)
