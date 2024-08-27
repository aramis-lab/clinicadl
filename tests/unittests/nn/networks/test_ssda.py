import torch

from clinicadl.nn.networks.ssda import Conv5_FC3_SSDA


def test_UNet():
    input_ = torch.randn(2, 1, 64, 63, 62)
    network = Conv5_FC3_SSDA(input_size=(1, 64, 63, 62), output_size=3)
    output = network(input_)
    for out in output:
        assert out.shape == torch.Size((2, 3))
