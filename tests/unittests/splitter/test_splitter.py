import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datatype.preprocessing import PETLinear
from clinicadl.splitter.splitter.kfold import KFold, KFoldConfig
from clinicadl.splitter.splitter.single_split import SingleSplit, SingleSplitConfig
from clinicadl.splitter.splitter.splitter import SubjectsSessionsSplit
from clinicadl.utils.exceptions import ClinicaDLTSVError

caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
data = pd.read_csv(caps_dir / "labels.tsv", sep="\t")

split_dir = caps_dir / "split_test" / "split"
fold_path = split_dir / "2_fold"
data = pd.read_csv(caps_dir / "labels.tsv")


def test_single_splitter():
    config = SingleSplitConfig(split_dir=split_dir)

    assert config.subset_name == "test"
    assert config.stratification is False
    assert config.valid_longitudinal is False
    assert np.isclose(config.p_categorical_threshold, 0.8, rtol=1e-09, atol=1e-09)
    assert np.isclose(config.p_categorical_threshold, 0.8, rtol=1e-09, atol=1e-09)
    assert config.json_name == "single_split_config.json"
    assert config.n_test == 100

    with pytest.raises(ValidationError):
        SingleSplitConfig(split_dir=split_dir, p_categorical_threshold=12)


def test_single_split():
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            tracer="18FAV45",
            suvr_reference_region="pons2",
            use_uncropped_image=True,
        ),
        data=data,
    )
    splitter = SingleSplit(split_dir=split_dir)

    with pytest.raises(ClinicaDLTSVError):
        splitter.get_splits(
            dataset=caps_dataset,
        )

    with pytest.raises(FileNotFoundError):
        splitter._read_split(Path("doesnt_exist"))

    with pytest.raises(FileNotFoundError):
        splitter._read_split(caps_dir / "test")

    with pytest.raises(FileNotFoundError):
        SingleSplit("doesnt_exist")


def test_kfold_splitter():
    config = KFoldConfig(split_dir=fold_path)

    assert config.subset_name == "validation"
    assert config.stratification is False
    assert config.valid_longitudinal is False
    assert config.json_name == "kfold_config.json"
    assert config.n_splits == 5


def test_kfold():
    kfold = KFold(split_dir=fold_path)
    config = kfold.config
    assert config.subset_name == "validation"
    assert config.stratification == "sex"
    assert config.valid_longitudinal is False
    assert config.json_name == "kfold_config.json"
    assert config.n_splits == 2

    assert isinstance(kfold.subjects_sessions_split[0], SubjectsSessionsSplit)

    with pytest.raises(ClinicaDLTSVError):
        splits = list(kfold.get_splits(dataset=CapsDataset(caps_dir, PETLinear())))

    with pytest.raises(FileNotFoundError):
        kfold._read_split(Path("doesnt_exist"))

    with pytest.raises(FileNotFoundError):
        kfold._read_split(caps_dir / "test")

    with pytest.raises(FileNotFoundError):
        KFold("doesnt_exist")

    with pytest.raises(FileNotFoundError):
        KFold(caps_dir / "test")
