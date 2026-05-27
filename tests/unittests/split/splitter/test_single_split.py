from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.datasets import (
    ConcatDataset,
    Dataset,
    PairedDataset,
    UnpairedDataset,
)
from clinicadl.data.structures import Sample
from clinicadl.io import BidsFileType
from clinicadl.split.splitter import SingleSplit

SPLIT_DIR = Path(__file__).parents[2] / "resources" / "split"
BAD_SPLIT_1 = Path(__file__).parents[2] / "resources" / "bad_split"
BAD_SPLIT_2 = Path(__file__).parents[2] / "resources" / "bad_split_2"


class MyDataset(Dataset):
    def __init__(self, participants_sessions: Iterable[tuple[str, str]], **kwargs):
        self._df = pd.DataFrame(
            {
                "participant_id": [pair[0] for pair in participants_sessions],
                "session_id": [pair[1] for pair in participants_sessions],
            }
        )
        for col, value in kwargs.items():
            self._df[col] = value

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
            file_type=BidsFileType(data_type="anat", suffix="T1w"),
        )

    def get_sample_info(self, idx, column):
        return self.df.iloc[idx][column]


DATASET_1 = MyDataset(
    participants_sessions=[("sub-000", "ses-M000"), ("sub-010", "ses-M003")]
)
DATASET_2 = MyDataset(
    participants_sessions=[
        ("sub-000", "ses-M000"),
        ("sub-100", "ses-M012"),
        ("sub-999", "ses-M099"),
        ("sub-999", "ses-M999"),
    ]
)

SPLITTER = SingleSplit(SPLIT_DIR)


def test_single_split():
    split = SPLITTER.get_split(DATASET_2)
    assert split.index == 0
    assert split.split_dir == SPLIT_DIR
    assert len(split.train_dataset) == 2
    assert len(split.val_dataset) == 1
    assert set(split.val_dataset.get_participant_session_couples()) == {
        ("sub-999", "ses-M099"),
    }

    # test errors
    with pytest.raises(FileNotFoundError, match="No such directory:*"):
        SingleSplit(SPLIT_DIR / "abc")

    with pytest.raises(FileNotFoundError, match="Required file missing:*"):
        SingleSplit(BAD_SPLIT_1)

    with pytest.raises(FileNotFoundError, match="No configuration file found in*"):
        SingleSplit(BAD_SPLIT_2)

    # eval dataset
    eval_dataset = deepcopy(DATASET_2)
    eval_dataset.df["x"] = "x"
    split = SPLITTER.get_split(DATASET_2, eval_dataset=eval_dataset)
    assert len(split.train_dataset) == 2
    assert len(split.val_dataset) == 1
    assert "x" not in split.train_dataset.df
    assert "x" in split.val_dataset.df


def test_single_split_concat():
    multimodal_dataset = ConcatDataset([DATASET_1, DATASET_2])
    split = SPLITTER.get_split(multimodal_dataset)
    assert len(split.train_dataset) == 3
    assert len(split.val_dataset) == 2


def test_single_split_paired():
    paired = PairedDataset([DATASET_1, DATASET_1])
    split = SPLITTER.get_split(paired)
    assert len(split.train_dataset) == 1
    assert len(split.val_dataset) == 1


def test_single_split_unpaired():
    unpaired = UnpairedDataset([DATASET_1, DATASET_1])
    split = SPLITTER.get_split(unpaired)
    assert len(split.train_dataset) == 1
    assert len(split.val_dataset) == 1
