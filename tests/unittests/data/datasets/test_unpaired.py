import re
from pathlib import Path
from typing import Iterable
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.datasets import Dataset, TensorDataset, UnpairedDataset
from clinicadl.data.structures import Sample
from clinicadl.io.bids import BidsFileType

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
            participant_id=self.df.iloc[idx]["participant_id"],
            session_id=self.df.iloc[idx]["session_id"],
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
            image_path="x",
            file_type=BidsFileType(data_type="anat", suffix=self.suffix),
        )

    def get_sample_info(self, idx, column):
        return self.df.iloc[idx][column]


def test_checks():
    dataset_1 = MyDataset([("sub-000", "ses-M000")])
    with pytest.raises(
        ValidationError,
        match=re.escape("UnpairedDataset requires at least 2 datasets to join!"),
    ):
        UnpairedDataset([dataset_1])

    dataset_1 = MyDataset([("sub-000", "ses-M000")], dataset_id=[0])
    with pytest.raises(
        ValidationError,
        match=re.escape(
            "'dataset_id' is a protected name. It cannot be in the DataFrames of the underlying datasets."
        ),
    ):
        UnpairedDataset([dataset_1, dataset_1])


def test_df():
    ref_df = pd.DataFrame(
        {
            "dataset_id": [0, 0, 1, 1, 1],
            "participant_id": ["sub-000", "sub-010", "sub-000", "sub-010", "sub-999"],
            "session_id": ["ses-M000", "ses-M000", "ses-M001", "ses-M000", "ses-M999"],
            "age": [1, 2, 1, 2, 3],
            "diagnosis": ["nan", "nan", "CN", "AD", "MCI"],
        }
    )
    dataset_1 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-010", "ses-M000")], age=[1, 2]
    )
    dataset_2 = MyDataset(
        [("sub-000", "ses-M001"), ("sub-010", "ses-M000"), ("sub-999", "ses-M999")],
        diagnosis=["CN", "AD", "MCI"],
        age=[1, 2, 3],
    )
    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=True)
    assert unpaired.df.fillna("nan").equals(ref_df)
    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=False)
    assert unpaired.df.fillna("nan").equals(ref_df)


def test_train_val():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-001", "ses-M001"),
        ]
    )
    bids_1.train = Mock()
    bids_1.eval = Mock()
    bids_2.train = Mock()
    bids_2.eval = Mock()

    multimodal_dataset = UnpairedDataset([bids_1, bids_2])
    multimodal_dataset.eval()
    bids_1.eval.assert_called_once()
    bids_2.eval.assert_called_once()
    multimodal_dataset.train()
    bids_1.train.assert_called_once()
    bids_2.train.assert_called_once()


def test_len():
    dataset_1 = MyDataset([("sub-000", "ses-M000"), ("sub-010", "ses-M000")])
    dataset_2 = MyDataset(
        [("sub-000", "ses-M001"), ("sub-010", "ses-M000"), ("sub-999", "ses-M999")],
    )
    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=True)
    assert len(unpaired) == 3
    unpaired = UnpairedDataset([dataset_1, dataset_2])
    assert len(unpaired) == 2


def test_get_participant_session_couples():
    dataset_1 = MyDataset([("sub-000", "ses-M000"), ("sub-010", "ses-M000")])
    dataset_2 = MyDataset(
        [("sub-000", "ses-M001"), ("sub-010", "ses-M000"), ("sub-999", "ses-M999")],
    )
    ref_set = {
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M000"),
        ("sub-000", "ses-M001"),
        ("sub-999", "ses-M999"),
    }
    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=False)
    assert unpaired.get_participant_session_couples() == ref_set
    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=True)
    assert unpaired.get_participant_session_couples() == ref_set


def test_get_sample_info():
    dataset_1 = MyDataset([("sub-000", "ses-M000")], age=[1])
    dataset_2 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M000"),
        ],
        age=[2, 3],
        diagnosis=["AD", "CN"],
    )
    unpaired = UnpairedDataset([dataset_1, dataset_2])
    assert unpaired.get_sample_info(0, "age") == (1, 2)
    unpaired.set_epoch(4)
    assert unpaired.get_sample_info(0, "age") == (1, 3)

    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=True)
    assert unpaired.get_sample_info(0, "diagnosis") == (None, "AD")

    with pytest.raises(
        KeyError,
        match="No column named 'abc' in any DataFrame of the datasets forming the UnpairedDataset.",
    ):
        unpaired.get_sample_info(0, "abc")


def test__getitem__():
    dataset_1 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-001", "ses-M000")], age=[1, 2]
    )
    dataset_2 = MyDataset(
        [("sub-000", "ses-M000"), ("sub-002", "ses-M000"), ("sub-003", "ses-M000")],
    )

    # oversample
    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=True)
    assert unpaired.mapping.equals(
        pd.DataFrame(
            {
                0: [1, 1, 0],
                1: [2, 1, 0],
            }
        ).rename_axis(columns="dataset_id", index="idx")
    )
    assert unpaired[1][0].participant_id == "sub-001"
    assert unpaired[1][1].participant_id == "sub-002"

    unpaired.set_epoch(1)
    assert unpaired.mapping.equals(
        pd.DataFrame(
            {
                0: [0, 0, 1],
                1: [0, 2, 1],
            }
        ).rename_axis(columns="dataset_id", index="idx")
    )
    assert unpaired[1][0].participant_id == "sub-000"
    assert unpaired[1][1].participant_id == "sub-003"

    # undersample
    unpaired = UnpairedDataset([dataset_1, dataset_2])
    assert unpaired.mapping.equals(
        pd.DataFrame(
            {
                0: [1, 0],
                1: [2, 1],
            }
        ).rename_axis(columns="dataset_id", index="idx")
    )
    assert unpaired[0][0].participant_id == "sub-001"
    assert unpaired[0][1].participant_id == "sub-003"

    unpaired.set_epoch(1)
    assert unpaired.mapping.equals(
        pd.DataFrame(
            {
                0: [0, 1],
                1: [0, 2],
            }
        ).rename_axis(columns="dataset_id", index="idx")
    )


def test_subset():
    dataset_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-001", "ses-M000"),
        ],
        age=[1, 2],
    )
    dataset_2 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-002", "ses-M000"),
            ("sub-003", "ses-M000"),
        ],
        age=[1, 3, 4],
    )
    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=True)
    subset = unpaired.subset(
        [
            ("sub-000", "ses-M000"),
            ("sub-003", "ses-M000"),
        ]
    )
    pd.testing.assert_frame_equal(
        subset.df,
        pd.DataFrame(
            {
                "dataset_id": [0, 1, 1],
                "participant_id": ["sub-000", "sub-000", "sub-003"],
                "session_id": ["ses-M000"] * 3,
                "age": [1, 1, 4],
            }
        ),
    )
    assert len(subset) == 2
    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=False)
    subset = unpaired.subset(
        [
            ("sub-000", "ses-M000"),
            ("sub-003", "ses-M000"),
        ]
    )
    assert len(subset) == 1


def test_from_json_to_json(tmp_path):
    tensors = TensorDataset(
        TENSORS / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json"
    )
    paired = UnpairedDataset(
        [tensors, tensors.subset([("sub-000", "ses-M000")])], oversample=True
    )

    paired.to_json(tmp_path / "dataset.json")
    paired = UnpairedDataset.from_json(tmp_path / "dataset.json")

    assert len(paired) == 2
    assert paired[1][0].participant_id == "sub-000"

    bids = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    multimodal_dataset = UnpairedDataset([bids, bids])
    multimodal_dataset.to_json(tmp_path / "dataset.json", overwrite=True)
