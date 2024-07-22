import pytest
import torch


@pytest.fixture
def input_3d():
    return torch.randn(2, 6, 10, 10, 10)


def test_SE_Block(input_3d):
    from clinicadl.nn.blocks import SE_Block

    layer = SE_Block(num_channels=input_3d.shape[1], ratio_channel=4)
    out = layer(input_3d)
    assert out.shape == input_3d.shape


def test_ResBlock_SE(input_3d):
    from clinicadl.nn.blocks import ResBlock_SE

    layer = ResBlock_SE(
        num_channels=input_3d.shape[1],
        block_number=1,
        input_size=input_3d.shape[1],
        ratio_channel=4,
    )
    out = layer(input_3d)
    assert out.shape[:2] == torch.Size(
        (
            input_3d.shape[0],
            2**3,
        )
    )
