import re
from pathlib import Path
from typing import Iterable
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.datasets import (
    Dataset,
    PairedDataset,
    TensorDataset,
)
from clinicadl.data.structures import Sample
from clinicadl.io.bids import BidsFileType
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.extraction import Slice

TENSORS = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "tensors"


class MyDataset(Dataset):
    def __init__(
        self,
        participants_sessions: Iterable[tuple[str, str]],
        suffix: str = "T1w",
        **kwargs,
    ):
        self.suffix = suffix
        self._df = pd.DataFrame(
            {
                "participant_id": [pair[0] for pair in participants_sessions],
                "session_id": [pair[1] for pair in participants_sessions],
            }
        )
        for arg, values in kwargs.items():
            self._df[arg] = values

    def __len__(self):
        return len(self._df)

    def train(self):
        pass

    def eval(self):
        pass

    def __getitem__(self, idx):
        return Sample(
            participant=self.df.iloc[idx]["participant_id"],
            session=self.df.iloc[idx]["session_id"],
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
            image_path="x",
            file_type=BidsFileType(data_type="anat", suffix=self.suffix),
        )

    def get_sample_info(self, idx, column):
        return self.df.iloc[idx][column]


def test_checks():
    dataset_1 = MyDataset([("sub-000", "ses-M000"), ("sub-001", "ses-M000")])
    dataset_2 = MyDataset([("sub-000", "ses-M000"), ("sub-000", "ses-M000")])
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "Datasets passed to PairedDataset cannot contain duplicated (participant, session) pairs, but some were founds in dataset 1:"
        ),
    ):
        PairedDataset([dataset_1, dataset_2])

    dataset_2 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-001", "ses-M000"), ("sub-001", "ses-M001")]
    )
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "To pair datasets, they must have exactly the same (participant, session) pairs. Differences were found for between dataset 0 and dataset 1:"
        ),
    ):
        PairedDataset([dataset_1, dataset_2])

    with pytest.raises(
        ValidationError,
        match=re.escape(
            "PairedDataset only accepts datasets of the same length. Dataset 0 is 6 samples long, whereas dataset 1 is 2."
        ),
    ):
        PairedDataset(
            [
                TensorDataset(
                    TENSORS / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json",
                    transforms=TransformsHandler(extraction=Slice()),
                ),
                TensorDataset(
                    TENSORS / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json"
                ),
            ]
        )

    dataset_2 = MyDataset([("sub-001", "ses-M000"), ("sub-000", "ses-M000")])
    with pytest.raises(
        RuntimeError,
        match=r"Sample 0 is associated to \('sub-00.*', 'ses-M000'\) in one dataset, but \('sub-00.*', 'ses-M000'\) in another. "
        r"Make sure that the \(participant, session\) are consistent across your datasets.",
    ):
        PairedDataset([dataset_1, dataset_2])

    dataset_1 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-001", "ses-M000")], age=[1.0, 1.0]
    )
    dataset_2 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-001", "ses-M000")], age=[0.0, 1.0]
    )
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "For (sub-000, ses-M000), different values found for 'age' across the datasets forming the PairedDataset: [1. 0.]"
        ),
    ):
        PairedDataset([dataset_1, dataset_2])

    with pytest.raises(
        ValidationError,
        match=re.escape("PairedDataset requires at least 2 datasets to join!"),
    ):
        PairedDataset([dataset_1])


def test_df():
    dataset_1 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-001", "ses-M000")],
        age=[0.0, 1.0],
        diagnosis=["AD", "CN"],
    )
    dataset_2 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-001", "ses-M000")], age=[0.0, 1.0]
    )

    paired = PairedDataset([dataset_1, dataset_2])
    pd.testing.assert_frame_equal(
        paired.df.fillna(-1),
        pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-001"],
                "session_id": ["ses-M000", "ses-M000"],
                "age": [0.0, 1.0],
                "diagnosis": ["AD", "CN"],
            }
        ),
    )


def test_get_participant_session_couples():
    dataset_1 = MyDataset([("sub-000", "ses-M000"), ("sub-001", "ses-M000")])
    dataset_2 = MyDataset([("sub-000", "ses-M000"), ("sub-001", "ses-M000")])
    paired = PairedDataset([dataset_1, dataset_2])
    assert paired.get_participant_session_couples() == set(
        [
            ("sub-000", "ses-M000"),
            ("sub-001", "ses-M000"),
        ]
    )


def test_get_sample_info():
    dataset_1 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-001", "ses-M000")],
        age=[0.0, 1.0],
        diagnosis=["AD", "CN"],
    )
    dataset_2 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-001", "ses-M000")],
        age=[0.0, 1.0],
        category=["A", "B"],
    )
    paired = PairedDataset([dataset_1, dataset_2])
    dataset_2.df["diagnosis"] = ["AD", "AD"]
    assert paired.get_sample_info(0, "age") == 0
    assert paired.get_sample_info(0, "category") == "A"
    with pytest.raises(
        KeyError, match="No column named 'abc' in any dataset of the PairedDataset."
    ):
        paired.get_sample_info(0, "abc")
    with pytest.raises(
        RuntimeError,
        match=r"Multiple values found for 'diagnosis' for sample 1 in the datasets. Got .* and .*",
    ):
        paired.get_sample_info(1, "diagnosis")


def test_train_val():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    bids_1.train = Mock()
    bids_1.eval = Mock()
    bids_2.train = Mock()
    bids_2.eval = Mock()

    multimodal_dataset = PairedDataset([bids_1, bids_2])
    multimodal_dataset.eval()
    bids_1.eval.assert_called_once()
    bids_2.eval.assert_called_once()
    multimodal_dataset.train()
    bids_1.train.assert_called_once()
    bids_2.train.assert_called_once()


def test_len():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    paired = PairedDataset([bids_1, bids_2])
    assert len(paired) == 2


def test__getitem__():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ],
        suffix="flair",
    )
    paired = PairedDataset([bids_1, bids_2])
    assert paired[1][0].participant == "sub-010"
    assert paired[1][0].file_type[0].suffix.pattern == "T1w"
    assert paired[1][1].participant == "sub-010"
    assert paired[1][1].file_type[0].suffix.pattern == "flair"


def test_subset():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
        ],
        age=[1, 2, 3],
    )
    pared = PairedDataset([bids_1, bids_2])
    subset = pared.subset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    assert len((subset)) == 2
    pd.testing.assert_frame_equal(
        subset.df,
        pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-010"],
                "session_id": ["ses-M000", "ses-M003"],
                "age": [1, 3],
            }
        ),
    )


def test_from_json_to_json(tmp_path):
    tensors = TensorDataset(
        TENSORS / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json"
    )
    paired = PairedDataset([tensors, tensors])

    paired.to_json(tmp_path / "dataset.json")
    paired = PairedDataset.from_json(tmp_path / "dataset.json")

    assert len(paired) == 2
    assert paired[1][0].participant == "sub-010"

    bids = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    multimodal_dataset = PairedDataset([bids, bids])
    multimodal_dataset.to_json(tmp_path / "dataset.json", overwrite=True)
