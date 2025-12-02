from pathlib import Path

import pandas as pd
import pytest

from clinicadl.data.datasets import *
from clinicadl.data.datasets.factory import get_dataset_from_dict, get_dataset_from_json
from clinicadl.data.datatypes import T1Linear

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
    return subset_df(DATAFRAME, participants_sessions)


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
    ],
)
def test_dataset_from_dict(dataset):
    d = dataset(**MANDATORY_ARGS[dataset.__name__])
    dict_ = d.to_dict()
    d = get_dataset_from_dict(dict_)
    assert isinstance(d, dataset)

    if dataset is CapsDataset:
        assert d.config.datatype.key == "t1-linear"


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
        assert d.config.datatype.key == "t1-linear"
