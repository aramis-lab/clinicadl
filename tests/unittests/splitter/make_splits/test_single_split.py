import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes.preprocessing import PETLinear, T1Linear
from clinicadl.splitter.make_splits import make_kfold, make_split
from clinicadl.tsvtools.utils import extract_baseline
from clinicadl.utils.exceptions import (
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


caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
data = pd.read_csv(caps_dir / "labels.tsv", sep="\t")

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
    assert dict_["n_test"] == n_test
    assert np.isclose(dict_["p_categorical_threshold"], 0.5, rtol=1e-09, atol=1e-09)
    assert np.isclose(dict_["p_continuous_threshold"], 0.5, rtol=1e-09, atol=1e-09)

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    assert len(test_df) == 15
    assert set(stratification).issubset(set(test_df.columns))

    assert (split_dir / "split_continuous_stats.tsv").is_file()
    assert (split_dir / "split_categorical_stats.tsv").is_file()

    split_dir_bis = make_split(sub_ses_t1, n_test=n_test)

    assert split_dir_bis == sub_ses_t1.parent / "split"

    split_dir_bis_bis = make_split(sub_ses_t1, n_test=n_test, stratification=False)

    assert split_dir_bis_bis == sub_ses_t1.parent / "split_2"

    remove_non_empty_dir(split_dir)
    remove_non_empty_dir(split_dir_bis)
    remove_non_empty_dir(split_dir_bis_bis)
