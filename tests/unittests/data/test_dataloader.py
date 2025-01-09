from pathlib import Path

import pytest
from pydantic import ValidationError

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset

BAD_INPUTS = [
    {"batch_size": 0},
    {"sampling_weights": [0, 1, 2.0]},
    {"num_workers": 0, "prefetch_factor": 1},
    {"prefetch_factor": 1},
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
    },
    {"sampling_weights": None, "num_workers": 1},
]


@pytest.mark.parametrize("args", GOOD_INPUTS)
def test_good_inputs(args: dict):
    c = DataLoaderConfig(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


@pytest.mark.parametrize("args", BAD_INPUTS)
def test_bad_inputs(args: dict):
    with pytest.raises(ValidationError):
        DataLoaderConfig(**args)


def test_get_dataloader(args: dict):
    with pytest.raises(ValidationError):
        DataLoaderConfig(**args)


caps_dir = Path(__file__).parents[1] / "ressources" / "caps_example"

# def test_get_dataloader():
#     caps_dir = Path(__file__).parents[1] / "ressources" / "caps_example"
#     dataset = CapsDataset(caps_directory=caps_dir, data=)
#     dataloader_config = DataLoaderConfig(
#         batch_size=2,
#         sampling_weights="age",
#         drop_last=True,
#         num_workers=1,
#         prefetch_factor=2,
#         pin_memory=True,
#     )
#     dataloder = dataloader_config.get_dataloader()
