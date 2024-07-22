import pytest
import torch

import clinicadl.nn.networks.ae as ae


@pytest.mark.parametrize("network", [net.value for net in ae.AE2d])
def test_2d_ae(network):
    input_ = torch.randn(2, 3, 100, 100)
    network = getattr(ae, network)(input_size=input_.shape[1:], dropout=0.5)
    output = network(input_)
    assert output.shape == input_.shape


@pytest.mark.parametrize("network", [net.value for net in ae.AE3d])
def test_3d_ae(network):
    input_ = torch.randn(2, 1, 49, 49, 49)
    if network == "CAE_half":
        network = getattr(ae, network)(
            input_size=input_.shape[1:], latent_space_size=10
        )
    else:
        network = getattr(ae, network)(input_size=input_.shape[1:], dropout=0.5)
    output = network(input_)
    assert output.shape == input_.shape
