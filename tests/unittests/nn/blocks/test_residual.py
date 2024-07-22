import torch

from clinicadl.nn.blocks import ResBlock


def test_resblock():
    input_ = torch.randn((2, 4, 5, 5, 5))
    resblock = ResBlock(block_number=1, input_size=4)
    assert resblock(input_).shape == torch.Size((2, 8, 5, 5, 5))
