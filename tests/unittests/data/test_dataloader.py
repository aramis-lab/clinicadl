from pathlib import Path

import pandas as pd
import pytest
import torch
from pydantic import ValidationError
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from clinicadl.data.dataloader.config import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.transforms import Transforms
from clinicadl.transforms.extraction import Sample, Slice
from clinicadl.utils.seed import pl_worker_init_function

BAD_INPUTS = [
    {"batch_size": 0},
    {"sampling_weights": [0, 1, 2.0]},
    {"num_workers": 0, "prefetch_factor": 1},
    {"prefetch_factor": 1},
    {"num_workers": 0, "persistent_workers": True},
    {"persistent_workers": True},
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
]

caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
data = pd.read_csv(caps_dir / "labels.tsv", sep="\t").drop(7)
data["age"] = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 10.0]


@pytest.mark.parametrize("args", GOOD_INPUTS)
def test_good_inputs(args: dict):
    c = DataLoaderConfig(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


@pytest.mark.parametrize("args", BAD_INPUTS)
def test_bad_inputs(args: dict):
    with pytest.raises(ValidationError):
        DataLoaderConfig(**args)


def test_get_dataloader():
    caps = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        label="age",
        data=data,
    )
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        sampling_weights="age",
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )
    dataloader = dataloader_config.get_dataloader(caps)
    assert dataloader.batch_size == 2
    assert dataloader.drop_last
    assert dataloader.num_workers == 1
    assert dataloader.prefetch_factor == 2
    assert dataloader.pin_memory
    assert dataloader.persistent_workers
    assert dataloader.worker_init_fn == pl_worker_init_function

    # check sampler
    assert isinstance(dataloader.sampler, WeightedRandomSampler)
    assert (
        dataloader.sampler.weights == torch.Tensor([0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 10.0])
    ).all()
    assert dataloader.sampler.num_samples == 7
    assert dataloader.sampler.replacement

    dataloader_config = DataLoaderConfig(
        shuffle=True,
    )
    dataloader = dataloader_config.get_dataloader(caps)
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert dataloader.sampler.shuffle
    assert dataloader.sampler.num_replicas == 1
    assert dataloader.sampler.rank == 0

    dataloader_config = DataLoaderConfig(
        shuffle=False,
    )
    dataloader = dataloader_config.get_dataloader(caps)
    assert isinstance(dataloader.sampler, DistributedSampler)
    assert not dataloader.sampler.shuffle
    assert dataloader.sampler.num_replicas == 1
    assert dataloader.sampler.rank == 0

    # check weights
    dataloader_config = DataLoaderConfig(
        sampling_weights="sex",
    )
    with pytest.raises(
        KeyError, match="Got 'sex' for 'sampling_weights' but there is no*"
    ):
        dataloader_config.get_dataloader(caps)

    dataloader_config = DataLoaderConfig(
        sampling_weights="session_id",
    )
    with pytest.raises(
        ValueError, match="Got 'session_id' for 'sampling_weights' but cannot convert*"
    ):
        dataloader_config.get_dataloader(caps)

    dataloader_config = DataLoaderConfig(
        sampling_weights="age",
    )
    with pytest.raises(ValueError, match="For data parallelism*"):
        dataloader_config.get_dataloader(caps, rank=0)


def test_ddp():
    caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
    caps = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        label="age",
        data=data,
        transforms=Transforms(image_transforms=[]),
    )
    caps.read_tensor_conversion("pet_all")
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        shuffle=False,
    )

    dataloader = iter(dataloader_config.get_dataloader(caps, dp_degree=2, rank=0))
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M000"
    assert batch[0].participant == "sub-000"
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-010"
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M000"
    assert batch[0].participant == "sub-100"
    assert batch[1].session == "ses-M099"
    assert batch[1].participant == "sub-999"
    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = iter(dataloader_config.get_dataloader(caps, dp_degree=2, rank=1))
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M003"
    assert batch[0].participant == "sub-000"
    assert batch[1].session == "ses-M012"
    assert batch[1].participant == "sub-010"
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2  # extra indices added
    assert batch[0].session == "ses-M012"
    assert batch[0].participant == "sub-100"
    assert batch[1].session == "ses-M000"
    assert batch[1].participant == "sub-000"
    with pytest.raises(StopIteration):
        next(dataloader)

    # shuffling
    torch.manual_seed(0)
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        shuffle=True,
    )

    dataloader = iter(dataloader_config.get_dataloader(caps, dp_degree=2, rank=0))
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M000"
    assert batch[0].participant == "sub-100"
    assert batch[1].session == "ses-M012"
    assert batch[1].participant == "sub-100"
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M003"
    assert batch[0].participant == "sub-010"
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-000"
    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = iter(dataloader_config.get_dataloader(caps, dp_degree=2, rank=1))
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M000"
    assert batch[0].participant == "sub-000"
    assert batch[1].session == "ses-M012"
    assert batch[1].participant == "sub-010"
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2  # extra indices added
    assert batch[0].session == "ses-M099"
    assert batch[0].participant == "sub-999"
    assert batch[1].session == "ses-M000"
    assert batch[1].participant == "sub-100"
    with pytest.raises(StopIteration):
        next(dataloader)

    # weighting
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        sampling_weights="age",
    )

    torch.manual_seed(0)
    dataloader = dataloader_config.get_dataloader(caps, dp_degree=2, rank=0)
    assert dataloader.sampler.num_samples == 4
    dataloader = iter(dataloader)
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M099"
    assert batch[0].participant == "sub-999"
    assert batch[1].session == "ses-M012"
    assert batch[1].participant == "sub-100"
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M099"
    assert batch[0].participant == "sub-999"
    assert batch[1].session == "ses-M099"
    assert batch[1].participant == "sub-999"
    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = dataloader_config.get_dataloader(caps, dp_degree=2, rank=1)
    assert dataloader.sampler.num_samples == 3
    dataloader = iter(dataloader)
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M012"
    assert batch[0].participant == "sub-100"
    assert batch[1].session == "ses-M099"
    assert batch[1].participant == "sub-999"
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 1
    assert batch[0].session == "ses-M012"
    assert batch[0].participant == "sub-100"
    with pytest.raises(StopIteration):
        next(dataloader)

    # weighting with slice
    sub_data = (
        data.set_index(["participant_id", "session_id"])
        .loc[
            [
                ("sub-000", "ses-M000"),
                ("sub-010", "ses-M003"),
            ]
        ]
        .reset_index()
    )
    caps = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        label="seg",
        data=sub_data,
        transforms=Transforms(extraction=Slice()),
        masks=["brain"],
    )
    caps.read_tensor_conversion("t1_without_transform")

    torch.manual_seed(0)
    dataloader_config = DataLoaderConfig(
        batch_size=2,
        sampling_weights="age",
    )

    dataloader = dataloader_config.get_dataloader(caps, dp_degree=2, rank=0)
    assert (dataloader.sampler.weights == torch.Tensor([0, 0, 1, 1])).all()
    assert dataloader.sampler.num_samples == 2
    dataloader = iter(dataloader)
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M003"
    assert batch[0].participant == "sub-010"
    assert batch[0].slice_position == 1
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-010"
    assert batch[1].slice_position == 0
    with pytest.raises(StopIteration):
        next(dataloader)

    dataloader = dataloader_config.get_dataloader(caps, dp_degree=2, rank=1)
    assert (dataloader.sampler.weights == torch.Tensor([0, 0, 1, 1])).all()
    assert dataloader.sampler.num_samples == 2
    dataloader = iter(dataloader)
    batch: list[Sample] = next(dataloader)
    assert len(batch) == 2
    assert batch[0].session == "ses-M003"
    assert batch[0].participant == "sub-010"
    assert batch[0].slice_position == 0
    assert batch[1].session == "ses-M003"
    assert batch[1].participant == "sub-010"
    assert batch[1].slice_position == 1
    with pytest.raises(StopIteration):
        next(dataloader)
