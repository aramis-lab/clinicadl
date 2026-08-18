import pytest

from clinicadl.losses.config import *
from clinicadl.losses.factory import get_loss_function_from_dict
from clinicadl.utils.json import read_json


@pytest.mark.parametrize(
    "config,mandatory_args",
    [
        (BCELossConfig, {}),
        (BCEWithLogitsLossConfig, {}),
        (CrossEntropyLossConfig, {}),
        (HuberLossConfig, {}),
        (KLDivLossConfig, {}),
        (L1LossConfig, {}),
        (MSELossConfig, {}),
        (MultiMarginLossConfig, {}),
        (NLLLossConfig, {}),
        (SmoothL1LossConfig, {}),
        (DiceLossConfig, {}),
        (DiceCELossConfig, {}),
        (DiceFocalLossConfig, {}),
        (GeneralizedDiceLossConfig, {}),
        (GeneralizedDiceFocalLossConfig, {}),
        (FocalLossConfig, {}),
        (TverskyLossConfig, {}),
        (SoftclDiceLossConfig, {}),
        (SSIMLossConfig, {"spatial_dims": 3}),
    ],
)
def test_get_loss_function_from_dict(config, mandatory_args, tmp_path):
    c = config(**mandatory_args)
    c.to_json(tmp_path / "config.json")
    dict_ = read_json(tmp_path / "config.json")
    c = get_loss_function_from_dict(dict_)
    assert isinstance(c, config)

    if config is NLLLossConfig:
        c = NLLLossConfig(weight=[1, 2])
        assert get_loss_function_from_dict(c.to_dict()).weight == [1, 2]


@pytest.mark.parametrize(
    "config",
    [
        DiceLossConfig(sigmoid=True, weight=[1, 2]),
        GeneralizedDiceLossConfig(w_type="simple", soft_label=True),
        FocalLossConfig(gamma=3, alpha=0.25, use_softmax=True),
    ],
)
def test_monai_loss_non_default_round_trip(config):
    assert get_loss_function_from_dict(config.to_dict()) == config
