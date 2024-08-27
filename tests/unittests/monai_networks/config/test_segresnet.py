import pytest
from pydantic import ValidationError

from clinicadl.monai_networks.config.resnet import SegResNetConfig


def test_fails_validations():
    with pytest.raises(ValidationError):
        SegResNetConfig(dropout_prob=1.1)


def test_passes_validations():
    SegResNetConfig(dropout_prob=0.5)


def test_SegResNetConfig():
    config = SegResNetConfig(
        spatial_dims=2,
        init_filters=3,
        in_channels=1,
        out_channels=1,
        dropout_prob=0.1,
        act=("ELU", {"inplace": False}),
        norm=("group", {"num_groups": 4}),
        use_conv_final=False,
        blocks_down=[1, 2, 3],
        blocks_up=[3, 2, 1],
        upsample_mode="pixelshuffle",
    )
    assert config.spatial_dims == 2
    assert config.init_filters == 3
    assert config.in_channels == 1
    assert config.out_channels == 1
    assert config.dropout_prob == 0.1
    assert config.act == ("elu", {"inplace": False})
    assert config.norm == ("group", {"num_groups": 4})
    assert config.use_conv_final == False
    assert config.blocks_down == (1, 2, 3)
    assert config.blocks_up == (3, 2, 1)
    assert config.upsample_mode == "pixelshuffle"
