from pathlib import Path
from typing import Iterable
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.datasets import ConcatDataset, Dataset, TensorDataset
from clinicadl.data.structures import Sample
from clinicadl.io import BidsFileType

TENSORS = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "tensors"


class MyDataset(Dataset):
    def __init__(
        self,
        participants_sessions: Iterable[tuple[str, str]],
        suffix: str = "T1w",
        image_shape: int = 3,
        additional_col: bool = False,
    ):
        self.suffix = suffix
        self.image_shape = image_shape
        self._df = pd.DataFrame(
            {
                "participant_id": [pair[0] for pair in participants_sessions],
                "session_id": [pair[1] for pair in participants_sessions],
            }
        )
        if additional_col:
            self._df["age"] = range(len(participants_sessions))

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
            image=tio.ScalarImage(
                tensor=torch.randn(
                    1, self.image_shape, self.image_shape, self.image_shape
                )
            ),
            image_path="x",
            file_type=BidsFileType(data_type="anat", suffix=self.suffix),
        )

    def get_sample_info(self, idx, column):
        return self.df.iloc[idx][column]


def test_checks():
    dataset = MyDataset([("sub-000", "ses-M000")])
    dataset.df["dataset_id"] = 0
    with pytest.raises(
        ValidationError,
        match="'dataset_id' is a protected name. It cannot be in the DataFrames of the underlying datasets.",
    ):
        ConcatDataset([dataset, dataset])


def test_get_participant_session_couples():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-010", "ses-M012"),
            ("sub-999", "ses-M999"),
        ]
    )
    multimodal_dataset = ConcatDataset(d for d in [bids_1, bids_2])
    assert multimodal_dataset.get_participant_session_couples() == set(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
            ("sub-999", "ses-M999"),
        ]
    )


def test_get_sample_info():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-010", "ses-M012"),
            ("sub-999", "ses-M999"),
        ]
    )
    multimodal_dataset = ConcatDataset([bids_1, bids_2])
    assert multimodal_dataset.get_sample_info(2, "participant_id") == "sub-010"
    with pytest.raises(
        KeyError,
        match="No column named 'diagnosis' in the metadata DataFrame of the dataset from which the sample is taken.",
    ):
        multimodal_dataset.get_sample_info(0, "diagnosis")


def test_len():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-010", "ses-M012"),
        ]
    )
    multimodal_dataset = ConcatDataset([bids_1, bids_2])
    assert len(multimodal_dataset) == 3


def test_train_val():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-010", "ses-M012"),
        ]
    )
    bids_1.train = Mock()
    bids_1.eval = Mock()
    bids_2.train = Mock()
    bids_2.eval = Mock()

    multimodal_dataset = ConcatDataset([bids_1, bids_2])
    multimodal_dataset.eval()
    bids_1.eval.assert_called_once()
    bids_2.eval.assert_called_once()
    multimodal_dataset.train()
    bids_1.train.assert_called_once()
    bids_2.train.assert_called_once()


def test_df():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-999", "ses-M999"),
        ],
        additional_col=True,
    )
    multimodal_dataset = ConcatDataset([bids_1, bids_2])
    pd.testing.assert_frame_equal(
        multimodal_dataset.df.fillna(-1),
        pd.DataFrame(
            {
                "dataset_id": [0, 1, 1],
                "participant_id": ["sub-000", "sub-000", "sub-999"],
                "session_id": ["ses-M000", "ses-M000", "ses-M999"],
                "age": [-1, 0.0, 1.0],
            }
        ),
    )


def test__getitem__():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-000", "ses-M000"),
            ("sub-999", "ses-M999"),
        ],
        suffix="flair",
    )
    multimodal_dataset = ConcatDataset([bids_1, bids_2])
    assert multimodal_dataset[0].participant == "sub-000"
    assert multimodal_dataset[0].file_type[0].suffix.pattern == "T1w"
    assert multimodal_dataset[2].participant == "sub-999"
    assert multimodal_dataset[2].file_type[0].suffix.pattern == "flair"


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
            ("sub-010", "ses-M003"),
            ("sub-999", "ses-M999"),
        ],
        additional_col=True,
    )
    multimodal_dataset = ConcatDataset([bids_1, bids_2])
    subset = multimodal_dataset.subset(
        [
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
        ]
    )
    assert len((subset)) == 3
    pd.testing.assert_frame_equal(
        subset.df.fillna(-1),
        pd.DataFrame(
            {
                "dataset_id": [0, 0, 1],
                "participant_id": ["sub-000", "sub-010", "sub-010"],
                "session_id": ["ses-M003", "ses-M003", "ses-M003"],
                "age": [-1, -1, 0.0],
            }
        ),
    )

    subset = multimodal_dataset.subset(
        [
            ("sub-999", "ses-M999"),
        ]
    )
    assert len(subset.datasets) == 1

    with pytest.raises(
        RuntimeError,
        match=r"No \(participant, session\) pairs are in the dataset. This would lead to an empty dataset!",
    ):
        multimodal_dataset.subset(
            [
                ("sub-010", "ses-M012"),
            ]
        )


def test_sanity_check():
    bids_1 = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    bids_2 = MyDataset(
        [
            ("sub-010", "ses-M003"),
        ],
        image_shape=2,
    )
    multimodal_dataset = ConcatDataset([bids_1, bids_2])
    with pytest.raises(
        RuntimeError,
        match="Different spatial shape found in the dataset:",
    ):
        multimodal_dataset.sanity_check(spatial_checks=["global_shape"])


def test_from_json_to_json(tmp_path):
    tensors = TensorDataset(
        TENSORS / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json"
    )
    multimodal_dataset = ConcatDataset([tensors, tensors])

    multimodal_dataset.to_json(tmp_path / "dataset.json")
    multimodal_dataset = ConcatDataset.from_json(tmp_path / "dataset.json")

    assert len(multimodal_dataset) == 4
    assert multimodal_dataset[3].participant == "sub-010"
    assert multimodal_dataset[3].session == "ses-M003"

    bids = MyDataset(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    multimodal_dataset = ConcatDataset([bids, bids])
    multimodal_dataset.to_json(tmp_path / "dataset.json", overwrite=True)
