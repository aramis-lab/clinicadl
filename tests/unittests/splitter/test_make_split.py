import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.preprocessing import PreprocessingT1, PreprocessingT2
from clinicadl.splitter.make_splits import make_kfold, make_split
from clinicadl.transforms import Transforms
from clinicadl.utils.enum import Preprocessing
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)


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


caps_dir = Path(__file__).parents[1] / "ressources" / "caps_example"

sub_ses_t1 = caps_dir / "subjects_t1.tsv"
sub_ses_df = pd.read_csv(sub_ses_t1, sep="\t")

split_dir = caps_dir / "split"
train_path = split_dir / "train.tsv"


def test_good_split():
    n_test = 15
    stratification = ["age", "sex", "test", "diagnosis"]
    subset_name = "test_test"

    split_dir = make_split(
        sub_ses_t1,
        output_dir=caps_dir / "test",
        subset_name=subset_name,
        stratification=stratification,
        n_test=n_test,
    )

    train_path = split_dir / "train_baseline.tsv"
    test_path = split_dir / f"{subset_name}_baseline.tsv"

    assert train_path.exists()
    assert test_path.exists()

    assert (split_dir / "single_split_config.json").is_file
    with (split_dir / "single_split_config.json").open(mode="r") as file:
        dict_ = json.load(file)

    assert dict_["json_name"] == "single_split_config.json"
    assert dict_["split_dir"] == str(split_dir)
    assert dict_["subset_name"] == subset_name
    assert dict_["stratification"] == stratification
    assert dict_["valid_longitudinal"] is False
    assert dict_["ignore_demographics"] is False
    assert dict_["n_test"] == n_test
    assert np.isclose(dict_["p_categorical_threshold"], 0.5, rtol=1e-09, atol=1e-09)
    assert np.isclose(dict_["p_continuous_threshold"], 0.5, rtol=1e-09, atol=1e-09)

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    assert len(test_df) == 15
    assert set(stratification).issubset(set(test_df.columns))

    split_dir_bis = make_split(sub_ses_t1, n_test=n_test)

    assert split_dir_bis == sub_ses_t1.parent / "split"

    split_dir_bis = make_split(sub_ses_t1, n_test=n_test, ignore_demographics=True)

    assert split_dir_bis == sub_ses_t1.parent / "split_2"

    remove_non_empty_dir(split_dir)
    remove_non_empty_dir(split_dir_bis)


def test_bad_split():
    with pytest.raises(ClinicaDLTSVError):
        make_split(caps_dir / "test.tsv", n_test=15)

    with pytest.raises(ClinicaDLTSVError):
        make_split(caps_dir / "subject_false.tsv", n_test=2)

    with pytest.raises(ValueError):
        make_split(sub_ses_t1, p_categorical_threshold=12, n_test=2)

    with pytest.raises(ValueError):
        make_split(sub_ses_t1, n_test=100)


def test_good_kfold():
    # fold_dir = make_kfold(train_path, stratification=["sex"], n_splits=2)
    assert True


def test_bad_kfold():
    assert True
