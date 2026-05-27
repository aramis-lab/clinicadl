import platform
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytest
import torch
import torchio as tio
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from clinicadl.data.dataloader import MergeBatchesCollate
from clinicadl.data.datasets import Dataset
from clinicadl.data.datasets.bids import BidsDataset
from clinicadl.data.structures import Sample
from clinicadl.io import BidsFileType
from clinicadl.split.split import Split

SPLIT_DIR = Path(__file__).parents[1] / "resources" / "split"
BIDS_DIR = Path(__file__).parents[1] / "resources" / "bids"


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


TRAIN_DATASET = MyDataset(
    [("sub-000", "ses-M000"), ("sub-001", "ses-M000")], age=[1, 2]
)
VAL_DATASET = MyDataset([("sub-002", "ses-M000"), ("sub-003", "ses-M000")])


def test_build_loaders():
    split = Split(
        index=0,
        split_dir=SPLIT_DIR,
        train_dataset=TRAIN_DATASET,
        val_dataset=VAL_DATASET,
    )

    with pytest.raises(
        RuntimeError,
        match="The split has no training dataloader defined. Please run 'build_train_loader'",
    ):
        split.train_loader

    with pytest.raises(
        RuntimeError,
        match="The split has no validation dataloader defined. Please run 'build_val_loader'",
    ):
        split.val_loader

    # split.parallelism(dp_degree=2, rank=0)
    # assert split.train_loader.sampler.num_replicas == 2
    # assert split.val_loader.sampler.num_replicas == 2

    split.build_train_loader(
        batch_size=2,
        sampling_weights="age",
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        collate_fn=(collate := MergeBatchesCollate()),
    )
    assert split.train_loader.batch_size == 2
    assert not split.train_loader.pin_memory
    assert split.train_loader.drop_last
    assert isinstance(split.train_loader.sampler, WeightedRandomSampler)
    assert split.train_loader.collate_fn is collate

    split.build_val_loader(
        batch_size=2,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        collate_fn=(collate := MergeBatchesCollate()),
    )
    assert split.val_loader.batch_size == 2
    assert not split.val_loader.pin_memory
    assert split.val_loader.drop_last
    assert isinstance(split.val_loader.sampler, DistributedSampler)
    assert split.val_loader.collate_fn is collate

    # # error
    # with pytest.raises(ValueError):
    #     split.parallelism(dp_degree=2, rank=2)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Avoid persistent_workers on macOS"
)
def test_workers():
    split = Split(
        index=0,
        split_dir=SPLIT_DIR,
        train_dataset=TRAIN_DATASET,
        val_dataset=VAL_DATASET,
    )
    split.build_train_loader(
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert split.train_loader.num_workers == 1
    assert split.train_loader.prefetch_factor == 2
    assert split.train_loader.persistent_workers

    split.build_val_loader(
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert split.train_loader.num_workers == 1
    assert split.train_loader.prefetch_factor == 2
    assert split.train_loader.persistent_workers


def test_to_json_from_json(tmp_path):
    train_dataset = BidsDataset(
        BIDS_DIR,
        BidsFileType(data_type="anat", suffix="T1w"),
        data=pd.DataFrame(
            {
                "participant_id": ["sub-000"],
                "session_id": ["ses-M000"],
            }
        ),
    )
    val_dataset = BidsDataset(
        BIDS_DIR,
        BidsFileType(data_type="anat", suffix="T1w"),
        data=pd.DataFrame(
            {
                "participant_id": ["sub-010"],
                "session_id": ["ses-M003"],
            }
        ),
    )
    split = Split(
        index=0,
        split_dir=SPLIT_DIR,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    split.build_train_loader(batch_size=2)
    split.build_val_loader(num_workers=1)
    split.to_json(tmp_path / "split.json")
    split = Split.from_json(tmp_path / "split.json")
    assert split.index == 0
    assert split.split_dir == SPLIT_DIR
    assert split.train_dataset.config.file_type.suffix.pattern == "T1w"
    assert split.val_loader.num_workers == 1

    split = Split(
        index=0,
        split_dir=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    split.to_json(tmp_path / "split.json", overwrite=True)
    split = Split.from_json(tmp_path / "split.json")
