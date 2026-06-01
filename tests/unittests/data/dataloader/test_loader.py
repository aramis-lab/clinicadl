import platform
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
import torchio as tio
from pydantic import ValidationError
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from clinicadl.data.dataloader import (
    CollateFn,
    DataLoader,
    MergeBatchesCollate,
    ToBatchCollate,
)
from clinicadl.data.dataloader.batch import Batch
from clinicadl.data.dataloader.loader import (
    DataLoaderConfig,
    get_dataloader_from_json_safely,
)
from clinicadl.data.datasets import (
    BidsDataset,
    Dataset,
    UnpairedDataset,
)
from clinicadl.data.structures import Sample
from clinicadl.io.bids import BidsFileType
from clinicadl.utils.seed import pl_worker_init_function

BAD_INPUTS = [
    {"batch_size": 0},
    {"sampling_weights": [0, 1, 2.0]},
    {"num_workers": 0, "prefetch_factor": 1},
    {"prefetch_factor": 1},
    {"num_workers": 0, "persistent_workers": True},
    {"persistent_workers": True},
    {"collate_fn": lambda x: x},
]
GOOD_INPUTS = [
    {
        "batch_size": 1,
        "sampling_weights": "age",
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "num_workers": 1,
        "prefetch_factor": 1,
        "persistent_workers": True,
    },
    {"sampling_weights": None, "num_workers": 1},
    {"collate_fn": ToBatchCollate()},
]

BIDS_DIR = Path(__file__).parents[2] / "resources" / "bids"
DATA = pd.read_csv(BIDS_DIR / "participantsXsessions.tsv", sep="\t").drop(7)
DATA["age"] = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 10.0]

FILE_TYPE = BidsFileType(
    data_type="pet", suffix="pet", without_entities={"desc": "Crop"}
)

BIDS = BidsDataset(
    BIDS_DIR,
    file_type=FILE_TYPE,
    columns=["age"],
    data=DATA,
)
BIDS_WITHOUT_LABEL = BidsDataset(
    BIDS_DIR,
    file_type=FILE_TYPE,
    data=DATA,
)

SAMPLE = Sample(
    image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
    participant="x",
    session="x",
    file_type=FILE_TYPE,
    image_path="x",
)


class Collate(CollateFn):
    def __call__(self, samples):
        return samples


class MyDataset(Dataset[Sample]):
    _df = pd.DataFrame()

    def __getitem__(self, idx: int) -> Any:
        return Sample(
            image=tio.ScalarImage(tensor=torch.randn(1, 1, 1, 1)),
            participant=str(idx),
            session="x",
            file_type=FILE_TYPE,
            image_path="x",
        )

    def __len__(self):
        return 7

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def get_sample_info(self, idx: int, column: str) -> Any:
        if column == "sex":
            raise KeyError()
        elif column == "session_id":
            return "ses-M000"
        return [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 10.0][idx]


class PairedDataset(MyDataset):
    def __getitem__(self, idx: int) -> Any:
        return (super().__getitem__(idx), super().__getitem__(idx))


@pytest.mark.parametrize("args", GOOD_INPUTS)
def test_good_inputs(args: dict):
    c = DataLoader(dataset=MyDataset(), **args)
    for arg, value in args.items():
        assert getattr(c.config, arg) == value


@pytest.mark.parametrize("args", BAD_INPUTS)
def test_bad_inputs(args: dict):
    with pytest.raises(ValidationError):
        DataLoader(dataset=MyDataset(), **args)


def test_dataloader():
    dataloader = DataLoader(
        MyDataset(),
        batch_size=2,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )
    assert dataloader.batch_size == 2
    assert dataloader.drop_last
    assert dataloader.pin_memory
    assert dataloader.worker_init_fn == pl_worker_init_function
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert not dataloader.sampler.shuffle
    assert dataloader.sampler.num_replicas == 1
    assert dataloader.sampler.rank == 0
    batch = next(iter(dataloader))
    assert isinstance(batch, Batch)
    assert batch[0].participant == "0"

    # check sampler
    dataloader = DataLoader(
        MyDataset(),
        sampling_weights="age",
    )

    torch.manual_seed(0)
    assert isinstance(dataloader.sampler, WeightedRandomSampler)
    torch.testing.assert_close(
        dataloader.sampler.weights,
        torch.tensor([0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 10.0], dtype=torch.float64),
    )
    assert dataloader.sampler.num_samples == 7
    assert dataloader.sampler.replacement
    batch = next(iter(dataloader))
    assert isinstance(batch, Batch)
    assert batch[0].participant == "6"

    # multi batches
    dataloader = DataLoader(
        PairedDataset(),
        shuffle=False,
    )
    batch = next(iter(dataloader))
    assert isinstance(batch, (list, tuple))  # depends on the OS?
    assert batch[0][0].participant == "0"
    assert batch[1][0].participant == "0"

    # checks
    with pytest.raises(
        KeyError, match="Failed to get the column 'sex' in the dataframe*"
    ):
        DataLoader(
            MyDataset(),
            sampling_weights="sex",
        )

    with pytest.raises(
        ValueError, match="Got 'session_id' for 'sampling_weights' but cannot convert*"
    ):
        DataLoader(
            MyDataset(),
            sampling_weights="session_id",
        )

    # dataloader_config = DataLoaderConfig(
    #     sampling_weights="age",
    # )
    # with pytest.raises(ValueError, match="For data parallelism*"):
    #     dataloader_config.get_object(BIDS, rank=0)

    with pytest.raises(
        ValueError, match="Can't use 'sampling_weights' with UnpairedDataset."
    ):
        DataLoader(UnpairedDataset([MyDataset(), MyDataset()]), sampling_weights="age")

    # with pytest.raises(
    #     ValueError,
    #     match="'rank' must be strictly smaller than 'dp_degree'. Got dp_degree=2 and rank=2",
    # ):
    #     dataloader_config.get_object(BIDS, rank=2, dp_degree=2)

    # set epoch
    dataloader = DataLoader(
        UnpairedDataset([MyDataset(), MyDataset()]), batch_size=2, collate_fn=Collate()
    )
    dataloader.set_epoch(5)
    batch = next(iter(dataloader))
    assert isinstance(batch, (list, tuple))
    assert batch[0][0].participant == "2"
    assert batch[0][1].participant == "2"
    dataloader.set_epoch(6)
    batch = next(iter(dataloader))
    assert batch[0][0].participant == "6"
    assert batch[0][1].participant == "5"


def test_dataloader_config():
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
        collate_fn=None,
        sampling_weights=None,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
    )
    dataloader = dataloader_config.get_object(MyDataset())
    assert isinstance(dataloader, DataLoader)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Avoid persistent_workers on macOS"
)
def test_workers():
    dataloader = DataLoader(
        MyDataset(),
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert dataloader.num_workers == 1
    assert dataloader.prefetch_factor == 2
    assert dataloader.persistent_workers


# def test_ddp():
#     bids = BidsDataset(
#         BIDS_DIR,
#         file_type=FILE_TYPE,
#         data=DATA,
#         columns=["age"],
#     )
#     dataloader_config = DataLoaderConfig(
#         batch_size=2,
#         shuffle=False,
#     )

#     dataloader = iter(dataloader_config.get_object(bids, dp_degree=2, rank=0))
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M000"
#     assert batch[0].participant == "sub-000"
#     assert batch[1].session == "ses-M003"
#     assert batch[1].participant == "sub-010"
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M000"
#     assert batch[0].participant == "sub-100"
#     assert batch[1].session == "ses-M099"
#     assert batch[1].participant == "sub-999"
#     with pytest.raises(StopIteration):
#         next(dataloader)

#     dataloader = iter(dataloader_config.get_object(bids, dp_degree=2, rank=1))
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M003"
#     assert batch[0].participant == "sub-000"
#     assert batch[1].session == "ses-M012"
#     assert batch[1].participant == "sub-010"
#     batch = next(dataloader)
#     assert len(batch) == 2  # extra indices added
#     assert batch[0].session == "ses-M012"
#     assert batch[0].participant == "sub-100"
#     assert batch[1].session == "ses-M000"
#     assert batch[1].participant == "sub-000"
#     with pytest.raises(StopIteration):
#         next(dataloader)

#     # shuffling
#     dataloader_config = DataLoaderConfig(
#         batch_size=2,
#         shuffle=True,
#     )

#     dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=0)
#     dataloader.set_epoch(5)
#     dataloader = iter(dataloader)
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M003"
#     assert batch[0].participant == "sub-000"
#     assert batch[1].session == "ses-M000"
#     assert batch[1].participant == "sub-100"
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M099"
#     assert batch[0].participant == "sub-999"
#     assert batch[1].session == "ses-M012"
#     assert batch[1].participant == "sub-010"
#     with pytest.raises(StopIteration):
#         next(dataloader)

#     dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=1)
#     dataloader.set_epoch(5)
#     dataloader = iter(dataloader)
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M000"
#     assert batch[0].participant == "sub-000"
#     assert batch[1].session == "ses-M003"
#     assert batch[1].participant == "sub-010"
#     batch = next(dataloader)
#     assert len(batch) == 2  # extra indices added
#     assert batch[0].session == "ses-M012"
#     assert batch[0].participant == "sub-100"
#     assert batch[1].session == "ses-M003"
#     assert batch[1].participant == "sub-000"
#     with pytest.raises(StopIteration):
#         next(dataloader)

#     # weighting
#     dataloader_config = DataLoaderConfig(
#         batch_size=2,
#         sampling_weights="age",
#     )

#     dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=0)
#     torch.manual_seed(0)
#     assert dataloader.sampler.num_samples == 4
#     dataloader = iter(dataloader)
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M099"
#     assert batch[0].participant == "sub-999"
#     assert batch[1].session == "ses-M012"
#     assert batch[1].participant == "sub-100"
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M099"
#     assert batch[0].participant == "sub-999"
#     assert batch[1].session == "ses-M099"
#     assert batch[1].participant == "sub-999"
#     with pytest.raises(StopIteration):
#         next(dataloader)

#     dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=1)
#     assert dataloader.sampler.num_samples == 3
#     dataloader = iter(dataloader)
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M000"
#     assert batch[0].participant == "sub-100"
#     assert batch[1].session == "ses-M099"
#     assert batch[1].participant == "sub-999"
#     batch = next(dataloader)
#     assert len(batch) == 1
#     assert batch[0].session == "ses-M012"
#     assert batch[0].participant == "sub-100"
#     with pytest.raises(StopIteration):
#         next(dataloader)

#     # weighting with slice
#     sub_data = (
#         DATA.set_index(["participant_id", "session_id"])
#         .loc[
#             [
#                 ("sub-000", "ses-M000"),
#                 ("sub-010", "ses-M003"),
#             ]
#         ]
#         .reset_index()
#     )
#     bids = BidsDataset(
#         BIDS_DIR,
#         file_type=BidsFileType(data_type="anat", suffix="T1w"),
#         data=sub_data,
#         transforms=TransformsHandler(extraction=Slice(slices=[0, 1])),
#         masks={"brain": BidsFileType(data_type="anat", suffix="mask")},
#     )

#     dataloader_config = DataLoaderConfig(
#         batch_size=2,
#         sampling_weights="age",
#     )
#     torch.manual_seed(2)

#     dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=0)
#     torch.testing.assert_close(
#         dataloader.sampler.weights, torch.tensor([0, 0, 1, 1], dtype=torch.float64)
#     )
#     assert dataloader.sampler.num_samples == 2
#     dataloader = iter(dataloader)
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M003"
#     assert batch[0].participant == "sub-010"
#     assert batch[0].sample_position == 1
#     assert batch[1].session == "ses-M003"
#     assert batch[1].participant == "sub-010"
#     assert batch[1].sample_position == 0
#     with pytest.raises(StopIteration):
#         next(dataloader)

#     dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=1)
#     torch.testing.assert_close(
#         dataloader.sampler.weights, torch.tensor([0, 0, 1, 1], dtype=torch.float64)
#     )
#     assert dataloader.sampler.num_samples == 2
#     dataloader = iter(dataloader)
#     batch = next(dataloader)
#     assert len(batch) == 2
#     assert batch[0].session == "ses-M003"
#     assert batch[0].participant == "sub-010"
#     assert batch[0].sample_position == 1
#     assert batch[1].session == "ses-M003"
#     assert batch[1].participant == "sub-010"
#     assert batch[1].sample_position == 1
#     with pytest.raises(StopIteration):
#         next(dataloader)


def test_serialize_deserialize(tmp_path):
    dataloader = DataLoader(
        MyDataset(),
        batch_size=2,
        shuffle=False,
    )

    d = dataloader.to_dict()
    dataloader = DataLoader.from_dict(d, MyDataset())
    assert dataloader.batch_size == 2

    dataloader.to_json(tmp_path / "dataloader.json")
    dataloader = DataLoader.from_json(tmp_path / "dataloader.json", MyDataset())
    assert dataloader.batch_size == 2

    dataloader = DataLoader(
        MyDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=MergeBatchesCollate(),
    )
    dataloader.to_json(tmp_path / "dataloader.json", overwrite=True)
    dataloader = DataLoader.from_json(tmp_path / "dataloader.json", MyDataset())
    assert isinstance(dataloader.collate_fn, MergeBatchesCollate)

    dataloader_config = DataLoaderConfig(
        batch_size=2,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
        collate_fn=Collate(),
        sampling_weights=None,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
    )
    dataloader_config.to_json(tmp_path / "dataloader.json", overwrite=True)
    assert get_dataloader_from_json_safely(tmp_path / "dataloader.json") == (None, [])
    new_config, fields = get_dataloader_from_json_safely(
        tmp_path / "dataloader.json", default=dataloader_config
    )
    assert isinstance(new_config, DataLoaderConfig)
    assert isinstance(new_config.collate_fn, Collate)
    assert fields == ["collate_fn"]
