import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms import Format, MergeFields
from clinicadl.transforms.config import FormatConfig, MergeFieldsConfig

BAD_INPUTS = [
    ({"dtype": int}, FormatConfig),
    ({"unsqueeze": -1}, FormatConfig),
    ({"squeeze": [0, -1]}, FormatConfig),
]
GOOD_INPUTS = [
    (
        {"unsqueeze": 0, "squeeze": True, "dtype": torch.int16, "include": ["abc"]},
        FormatConfig,
    ),
    (
        {"unsqueeze": None, "squeeze": 1, "dtype": None, "exclude": ["abc"]},
        FormatConfig,
    ),
    ({"squeeze": False, "dtype": np.dtype("float32")}, FormatConfig),
    ({"squeeze": [0, 1], "dtype": "float"}, FormatConfig),
    (
        {
            "keys": ["abc", "bcd"],
            "output_key": "abc",
            "include": ["abc"],
        },
        MergeFieldsConfig,
    ),
]


@pytest.fixture
def datapoint():
    return DataPoint(
        image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
        label=tio.LabelMap(tensor=torch.randint(0, 2, (1, 2, 2, 2))),
        participant_id="abc",
        session_id="abc",
        array=np.array([1, 2]),
        tensor=torch.tensor([1, 2]),
    )


@pytest.mark.parametrize("args,config", BAD_INPUTS)
def test_bad_inputs(args, config):
    with pytest.raises(ValidationError):
        config(**args)


@pytest.mark.parametrize("args,config", GOOD_INPUTS)
def test_good_inputs(args: dict, config):
    c = config(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


@pytest.mark.parametrize(
    "args,config,transform",
    [
        ({"include": ["array"], "dtype": "float16"}, FormatConfig, Format),
        (
            {"keys": ["array", "tensor"], "output_key": "merge"},
            MergeFieldsConfig,
            MergeFields,
        ),
    ],
)
def test_get_object(datapoint, args, config, transform):
    c = config(**args, copy=True)
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config, transform)
    output = transform_from_config(datapoint)
    assert isinstance(output, DataPoint)
    assert output is not datapoint
