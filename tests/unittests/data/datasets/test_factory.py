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
from clinicadl.io import T1Linear
from clinicadl.transforms import TransformsHandler

CAPS_DIR = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "caps"
DATA = pd.DataFrame(
    {
        "participant_id": ["sub-010", "sub-000"],
        "session_id": ["ses-M003", "ses-M000"],
        "abc": ["x", "y"],
    }
)

BIDS_DATASET = BidsDataset(
    CAPS_DIR,
    file_type=T1Linear(use_uncropped_image=True),
    data=DATA,
)

MANDATORY_ARGS = {
    "BidsDataset": {
        "bids": CAPS_DIR,
        "data": DATA,
        "file_type": T1Linear(use_uncropped_image=True),
    },
    "ConcatDataset": {"datasets": [BIDS_DATASET, BIDS_DATASET]},
    "PairedDataset": {"datasets": [BIDS_DATASET, BIDS_DATASET]},
    "UnpairedDataset": {"datasets": [BIDS_DATASET, BIDS_DATASET]},
}


@pytest.mark.parametrize(
    "dataset",
    [
        BidsDataset,
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

    if isinstance(d, BidsDataset):
        assert d.config.file_type.suffix.pattern == "T1w"


@pytest.mark.parametrize(
    "dataset",
    [
        BidsDataset,
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

    if isinstance(d, BidsDataset):
        assert d.config.file_type.suffix.pattern == "T1w"


def test_dataset_from_json_safely(tmp_path):
    dataset = BidsDataset(
        **MANDATORY_ARGS["BidsDataset"],
        transforms=TransformsHandler(image_transforms=[tio.ZNormalization()]),
    )
    dataset.to_json(tmp_path / "dataset.json")

    assert get_dataset_from_json_safely(tmp_path / "dataset.json") == (None, [])
    obj, fields = get_dataset_from_json_safely(
        tmp_path / "dataset.json", default=dataset
    )
    assert isinstance(obj, BidsDataset)
    assert isinstance(obj.transforms.image_transforms.transforms[0], tio.ZNormalization)
    assert fields == ["transforms"]
