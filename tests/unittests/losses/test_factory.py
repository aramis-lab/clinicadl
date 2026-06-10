import pytest

from clinicadl.losses.config import *
from clinicadl.losses.factory import get_loss_function_from_dict
from clinicadl.utils.json import read_json


@pytest.mark.parametrize(
    "config",
    [
        BCELossConfig,
        BCEWithLogitsLossConfig,
        CrossEntropyLossConfig,
        HuberLossConfig,
        KLDivLossConfig,
        L1LossConfig,
        MSELossConfig,
        MultiMarginLossConfig,
        NLLLossConfig,
        SmoothL1LossConfig,
    ],
)
def test_get_loss_function_from_dict(config, tmp_path):
    c = config()
    c.to_json(tmp_path / "config.json")
    dict_ = read_json(tmp_path / "config.json")
    c = get_loss_function_from_dict(dict_)
    assert isinstance(c, config)

    if config is NLLLossConfig:
        c = NLLLossConfig(weight=[1, 2])
        assert get_loss_function_from_dict(c.to_dict()).weight == [1, 2]
