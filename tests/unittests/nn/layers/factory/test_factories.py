import pytest
import torch.nn as nn


def test_get_conv_layer():
    from clinicadl.nn.layers.factory import get_conv_layer

    assert get_conv_layer(2) == nn.Conv2d
    assert get_conv_layer(3) == nn.Conv3d
    with pytest.raises(AssertionError):
        get_conv_layer(1)


def test_get_norm_layer():
    from clinicadl.nn.layers.factory import get_norm_layer

    assert get_norm_layer("InstanceNorm", 2) == nn.InstanceNorm2d
    assert get_norm_layer("BatchNorm", 3) == nn.BatchNorm3d
    assert get_norm_layer("GroupNorm", 3) == nn.GroupNorm


def test_get_pool_layer():
    from clinicadl.nn.layers import PadMaxPool3d
    from clinicadl.nn.layers.factory import get_pool_layer

    assert get_pool_layer("MaxPool", 2) == nn.MaxPool2d
    assert get_pool_layer("PadMaxPool", 3) == PadMaxPool3d
