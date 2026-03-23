from pathlib import Path

import pandas as pd
import pytest
import torchio as tio

from clinicadl.data.datasets import *
from clinicadl.data.datasets.factory import (
    get_dataset_from_dict,
    get_dataset_from_json,
    get_dataset_from_json_safely,
)
from clinicadl.data.datatypes import T1Linear
from clinicadl.transforms import TransformsHandler

from .utils import subset_df

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
DATAFRAME = pd.read_csv(CAPS_DIR / "tsv" / "labels.tsv", sep="\t")

CAPS_DATASET = CapsDataset(
    CAPS_DIR,
    datatype=T1Linear(use_uncropped_image=True),
    data=subset_df(DATAFRAME, [("sub-010", "ses-M003"), ("sub-000", "ses-M000")]),
)
CAPS_DATASET.read_tensor_conversion()


def sub_data(participants_sessions: list[tuple[str, str]]) -> pd.DataFrame:
    df = subset_df(DATAFRAME, participants_sessions)
    df["abc"] = 0
    return df


MANDATORY_ARGS = {
    "CapsDataset": {
        "directory": CAPS_DIR,
        "data": sub_data(
            [
                ("sub-000", "ses-M000"),
                ("sub-010", "ses-M003"),
            ]
        ),
        "datatype": T1Linear(use_uncropped_image=True),
    },
    "ConcatDataset": {"datasets": [CAPS_DATASET, CAPS_DATASET]},
    "PairedDataset": {"datasets": [CAPS_DATASET, CAPS_DATASET]},
    "UnpairedDataset": {"datasets": [CAPS_DATASET, CAPS_DATASET]},
}


@pytest.mark.parametrize(
    "dataset",
    [
        CapsDataset,
        ConcatDataset,
        PairedDataset,
        UnpairedDataset,
    ],
)
def test_dataset_from_dict(dataset):
    d = dataset(**MANDATORY_ARGS[dataset.__name__])
    dict_ = d.to_dict()
    d = get_dataset_from_dict(dict_)
    assert isinstance(d, dataset)

    if dataset is CapsDataset:
        assert d.config.datatype.name == "t1-linear"


@pytest.mark.parametrize(
    "dataset",
    [
        CapsDataset,
        ConcatDataset,
        PairedDataset,
        UnpairedDataset,
    ],
)
def test_dataset_from_json(tmp_path, dataset):
    d = dataset(**MANDATORY_ARGS[dataset.__name__])
    d.to_json(tmp_path / "dataset.json")
    d = get_dataset_from_json(tmp_path / "dataset.json")
    assert isinstance(d, dataset)

    if dataset is CapsDataset:
        assert d.config.datatype.name == "t1-linear"


def test_dataset_from_json_safely(tmp_path):
    dataset = CapsDataset(
        **MANDATORY_ARGS["CapsDataset"],
        transforms=TransformsHandler(image_transforms=[tio.ZNormalization()]),
        columns={"abc": lambda x: str(x)},
    )
    dataset.to_json(tmp_path / "dataset.json")

    assert get_dataset_from_json_safely(tmp_path / "dataset.json") == (None, [])
    obj, fields = get_dataset_from_json_safely(
        tmp_path / "dataset.json", default=dataset
    )
    assert isinstance(obj, CapsDataset)
    assert isinstance(obj.transforms.image_transforms.transforms[0], tio.ZNormalization)
    assert obj.config.columns["abc"](0) == "0"
    assert fields == ["transforms", "columns"]
