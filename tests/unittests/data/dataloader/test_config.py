import platform
from pathlib import Path

import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from clinicadl.data.dataloader import (
    CollateFn,
    DataLoaderConfig,
    MergeBatchesCollate,
    ToBatchCollate,
    ToBatchesCollate,
)
from clinicadl.data.dataloader.batch import Batch
from clinicadl.data.dataloader.config import get_dataloader_from_json_safely
from clinicadl.data.datasets import (
    BidsDataset,
    ConcatDataset,
    PairedDataset,
    UnpairedDataset,
)
from clinicadl.io import BidsFileType
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import PadConfig
from clinicadl.transforms.extraction import Slice
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
        "sampling_weights": "sex",
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


@pytest.mark.parametrize("args", GOOD_INPUTS)
def test_good_inputs(args: dict):
    c = DataLoaderConfig(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


@pytest.mark.parametrize("args", BAD_INPUTS)
def test_bad_inputs(args: dict):
    with pytest.raises(ValidationError):
        DataLoaderConfig(**args)


def test_get_object():
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        sampling_weights="age",
        drop_last=True,
        pin_memory=True,
    )
    dataloader = dataloader_config.get_object(BIDS)
    assert dataloader.batch_size == 2
    assert dataloader.drop_last
    assert dataloader.pin_memory
    assert dataloader.worker_init_fn == pl_worker_init_function

    assert dataloader.config == dataloader_config

    # check sampler
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
    assert batch[0].participant == "sub-999"
    assert batch[0].session == "ses-M099"

    dataloader_config = DataLoaderConfig(
        shuffle=True,
    )
    dataloader = dataloader_config.get_object(PairedDataset([BIDS, BIDS_WITHOUT_LABEL]))
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert dataloader.sampler.shuffle
    assert dataloader.sampler.num_replicas == 1
    assert dataloader.sampler.rank == 0
    batch = next(iter(dataloader))
    assert isinstance(batch, (list, tuple))  # depends on the OS?
    assert batch[0][0].participant == "sub-100"
    assert batch[0][0].session == "ses-M000"
    assert batch[0].get_field("age") == torch.tensor([5.0])

    dataloader_config = DataLoaderConfig(shuffle=False, collate_fn=ToBatchCollate())
    dataloader = dataloader_config.get_object(ConcatDataset([BIDS, BIDS_WITHOUT_LABEL]))
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert not dataloader.sampler.shuffle
    assert dataloader.sampler.num_replicas == 1
    assert dataloader.sampler.rank == 0
    batch = next(iter(dataloader))
    assert isinstance(batch, Batch)
    assert batch[0].participant == "sub-000"
    assert batch[0].session == "ses-M000"

    # checks
    dataloader_config = DataLoaderConfig(
        sampling_weights="sex",
    )
    with pytest.raises(
        KeyError, match="Failed to get the column 'sex' in the dataframe*"
    ):
        dataloader_config.get_object(BIDS)

    dataloader_config = DataLoaderConfig(
        sampling_weights="session_id",
    )
    with pytest.raises(
        ValueError, match="Got 'session_id' for 'sampling_weights' but cannot convert*"
    ):
        dataloader_config.get_object(BIDS)

    dataloader_config = DataLoaderConfig(
        sampling_weights="age",
    )
    with pytest.raises(ValueError, match="For data parallelism*"):
        dataloader_config.get_object(BIDS, rank=0)

    with pytest.raises(
        ValueError, match="Can't use 'sampling_weights' with UnpairedDataset."
    ):
        dataloader_config.get_object(UnpairedDataset([BIDS, BIDS]))

    with pytest.raises(
        ValueError,
        match="'rank' must be strictly smaller than 'dp_degree'. Got dp_degree=2 and rank=2",
    ):
        dataloader_config.get_object(BIDS, rank=2, dp_degree=2)

    # tets other datasets
    dataloader = DataLoaderConfig(
        batch_size=2, collate_fn=ToBatchesCollate()
    ).get_object(UnpairedDataset([BIDS, BIDS_WITHOUT_LABEL]))
    dataloader.set_epoch(5)
    batch = next(iter(dataloader))
    assert isinstance(batch, (list, tuple))
    assert len(batch[0]) == 2
    assert (batch[0].get_field("age") == torch.tensor([1.0, 10.0])).all()

    dataloader = DataLoaderConfig(
        batch_size=5, shuffle=True, collate_fn=MergeBatchesCollate()
    ).get_object(PairedDataset([BIDS, BIDS_WITHOUT_LABEL]))
    batch = next(iter(dataloader))
    assert len(batch) == 5


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Avoid persistent_workers on macOS"
)
def test_workers():
    dataloader_config = DataLoaderConfig(
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
    )
    dataloader = dataloader_config.get_object(BIDS)
    assert dataloader.num_workers == 1
    assert dataloader.prefetch_factor == 2
    assert dataloader.persistent_workers


def test_train_eval():
    bids = BidsDataset(
        BIDS_DIR,
        file_type=FILE_TYPE,
        transforms=TransformsHandler(augmentations=[PadConfig(padding=1)]),
        data=DATA,
    )
    dataloader = iter(DataLoaderConfig().get_object(bids))
    bids.train()
    out = next(dataloader)
    assert out[0].image.shape == (1, 3, 3, 3)
    bids.eval()
    out = next(dataloader)
    assert out[0].image.shape == (1, 1, 1, 1)


def test_ddp():
    bids = BidsDataset(
        BIDS_DIR,
        file_type=FILE_TYPE,
        data=DATA,
        columns=["age"],
    )
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        shuffle=False,
    )

    dataloader = iter(dataloader_config.get_object(bids, dp_degree=2, rank=0))
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M000"
    assert batch[0].participant == "sub-000"
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-010"
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M000"
    assert batch[0].participant == "sub-100"
    assert batch[1].session == "ses-M099"
    assert batch[1].participant == "sub-999"
    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = iter(dataloader_config.get_object(bids, dp_degree=2, rank=1))
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M003"
    assert batch[0].participant == "sub-000"
    assert batch[1].session == "ses-M012"
    assert batch[1].participant == "sub-010"
    batch = next(dataloader)
    assert len(batch) == 2  # extra indices added
    assert batch[0].session == "ses-M012"
    assert batch[0].participant == "sub-100"
    assert batch[1].session == "ses-M000"
    assert batch[1].participant == "sub-000"
    with pytest.raises(StopIteration):
        next(dataloader)

    # shuffling
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        shuffle=True,
    )

    dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=0)
    dataloader.set_epoch(5)
    dataloader = iter(dataloader)
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M003"
    assert batch[0].participant == "sub-000"
    assert batch[1].session == "ses-M000"
    assert batch[1].participant == "sub-100"
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M099"
    assert batch[0].participant == "sub-999"
    assert batch[1].session == "ses-M012"
    assert batch[1].participant == "sub-010"
    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=1)
    dataloader.set_epoch(5)
    dataloader = iter(dataloader)
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M000"
    assert batch[0].participant == "sub-000"
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-010"
    batch = next(dataloader)
    assert len(batch) == 2  # extra indices added
    assert batch[0].session == "ses-M012"
    assert batch[0].participant == "sub-100"
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-000"
    with pytest.raises(StopIteration):
        next(dataloader)

    # weighting
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        sampling_weights="age",
    )

    dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=0)
    torch.manual_seed(0)
    assert dataloader.sampler.num_samples == 4
    dataloader = iter(dataloader)
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M099"
    assert batch[0].participant == "sub-999"
    assert batch[1].session == "ses-M012"
    assert batch[1].participant == "sub-100"
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M099"
    assert batch[0].participant == "sub-999"
    assert batch[1].session == "ses-M099"
    assert batch[1].participant == "sub-999"
    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=1)
    assert dataloader.sampler.num_samples == 3
    dataloader = iter(dataloader)
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M000"
    assert batch[0].participant == "sub-100"
    assert batch[1].session == "ses-M099"
    assert batch[1].participant == "sub-999"
    batch = next(dataloader)
    assert len(batch) == 1
    assert batch[0].session == "ses-M012"
    assert batch[0].participant == "sub-100"
    with pytest.raises(StopIteration):
        next(dataloader)

    # weighting with slice
    sub_data = (
        DATA.set_index(["participant_id", "session_id"])
        .loc[
            [
                ("sub-000", "ses-M000"),
                ("sub-010", "ses-M003"),
            ]
        ]
        .reset_index()
    )
    bids = BidsDataset(
        BIDS_DIR,
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
        data=sub_data,
        transforms=TransformsHandler(extraction=Slice(slices=[0, 1])),
        masks={"brain": BidsFileType(data_type="anat", suffix="mask")},
    )

    dataloader_config = DataLoaderConfig(
        batch_size=2,
        sampling_weights="age",
    )
    torch.manual_seed(2)

    dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=0)
    torch.testing.assert_close(
        dataloader.sampler.weights, torch.tensor([0, 0, 1, 1], dtype=torch.float64)
    )
    assert dataloader.sampler.num_samples == 2
    dataloader = iter(dataloader)
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M003"
    assert batch[0].participant == "sub-010"
    assert batch[0].sample_position == 1
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-010"
    assert batch[1].sample_position == 0
    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = dataloader_config.get_object(bids, dp_degree=2, rank=1)
    torch.testing.assert_close(
        dataloader.sampler.weights, torch.tensor([0, 0, 1, 1], dtype=torch.float64)
    )
    assert dataloader.sampler.num_samples == 2
    dataloader = iter(dataloader)
    batch = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M003"
    assert batch[0].participant == "sub-010"
    assert batch[0].sample_position == 1
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-010"
    assert batch[1].sample_position == 1
    with pytest.raises(StopIteration):
        next(dataloader)


def test_serialize_deserialize(tmp_path):
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        shuffle=False,
    )

    d = dataloader_config.to_dict()
    dataloader_config = DataLoaderConfig.from_dict(d)
    assert dataloader_config.batch_size == 2

    dataloader_config.to_json(tmp_path / "dataloader.json")
    dataloader_config = DataLoaderConfig.from_json(tmp_path / "dataloader.json")
    assert dataloader_config.batch_size == 2

    dataloader_config = DataLoaderConfig(
        batch_size=2,
        shuffle=False,
        collate_fn=MergeBatchesCollate(),
    )
    dataloader_config.to_json(tmp_path / "dataloader.json", overwrite=True)
    dataloader_config = DataLoaderConfig.from_json(tmp_path / "dataloader.json")
    assert isinstance(dataloader_config.collate_fn, MergeBatchesCollate)

    # without errors
    class CustomCollate(CollateFn):
        def __call__(self, samples):
            return samples

    dataloader_config = DataLoaderConfig(
        batch_size=2, shuffle=False, collate_fn=CustomCollate()
    )
    dataloader_config.to_json(tmp_path / "dataloader.json", overwrite=True)
    assert get_dataloader_from_json_safely(tmp_path / "dataloader.json") == (None, [])
    new_config, fields = get_dataloader_from_json_safely(
        tmp_path / "dataloader.json", default=dataloader_config
    )
    assert isinstance(new_config, DataLoaderConfig)
    assert isinstance(new_config.collate_fn, CustomCollate)
    assert fields == ["collate_fn"]
