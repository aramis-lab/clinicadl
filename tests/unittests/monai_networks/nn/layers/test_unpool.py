import pytest
from torch.nn import ConvTranspose2d, ConvTranspose3d, Upsample

from clinicadl.monai_networks.nn.layers import get_unpool_layer


def test_get_unpool_layer():
    upsample = get_unpool_layer("upsample", spatial_dims=2)
    assert isinstance(upsample, Upsample)
    with pytest.raises(TypeError):
        get_unpool_layer("convtranspose", spatial_dims=2)

    upsample = get_unpool_layer(
        ("upsample", {"size": (5, 5), "mode": "bilinear"}),
        spatial_dims=2,
        in_channels=1,
    )
    assert isinstance(upsample, Upsample)
    assert upsample.mode == "bilinear"
    assert upsample.size == (5, 5)

    upconv = get_unpool_layer(
        ("convtranspose", {"kernel_size": 2, "stride": 2}),
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
    )
    assert isinstance(upconv, ConvTranspose2d)
    assert upconv.kernel_size == (2, 2)
    assert upconv.stride == (2, 2)

    upconv = get_unpool_layer(
        ("convtranspose", {"kernel_size": 2, "stride": 2}),
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
    )
    assert isinstance(upconv, ConvTranspose3d)
    assert upconv.kernel_size == (2, 2, 2)
    assert upconv.stride == (2, 2, 2)
