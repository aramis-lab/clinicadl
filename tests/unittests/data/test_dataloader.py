import pytest
from pydantic import ValidationError

from clinicadl.data.dataloader import DataLoaderConfig

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
