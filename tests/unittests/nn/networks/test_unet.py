import torch

from clinicadl.nn.networks.unet import UNet


def test_UNet():
    input_ = torch.randn(2, 1, 64, 64, 64)  # TODO : specify the size that works
    network = UNet()
    assert network(input_).shape == input_.shape
