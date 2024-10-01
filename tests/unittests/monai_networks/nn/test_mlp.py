import pytest
import torch
from torch.nn import ELU, Dropout, InstanceNorm1d, Linear

from clinicadl.monai_networks.nn import MLP
from clinicadl.monai_networks.nn.layers import ActFunction


@pytest.fixture
def input_tensor():
    return torch.randn(8, 10)


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_activations(input_tensor, act):
    net = MLP(
        in_channels=10, out_channels=2, hidden_channels=[6, 4], act=act, output_act=act
    )
    assert net(input_tensor).shape == (8, 2)


@pytest.mark.parametrize(
    "dropout,norm,bias,adn_ordering",
    [
        (None, "batch", True, "ADN"),
        (0.5, "layer", False, "DAN"),
        (0.5, "syncbatch", True, "NA"),
        (0.0, "instance", True, "DN"),
        (None, ("group", {"num_groups": 2}), True, "N"),
        (0.5, None, True, "ADN"),
        (0.5, "batch", True, ""),
    ],
)
def test_params(input_tensor, dropout, norm, bias, adn_ordering):
    net = MLP(
        in_channels=10,
        out_channels=2,
        hidden_channels=[6, 4],
        dropout=dropout,
        norm=norm,
        act=None,
        bias=bias,
        adn_ordering=adn_ordering,
    )
    assert net(input_tensor).shape == (8, 2)
    assert isinstance(net.output, Linear)

    if bias:
        assert len(net.hidden_0.linear.bias) > 0
        assert len(net.hidden_1.linear.bias) > 0
        assert len(net.output.bias) > 0
    else:
        assert net.hidden_0.linear.bias is None
        assert net.hidden_1.linear.bias is None
        assert net.output.bias is None
    if isinstance(dropout, float) and "D" in adn_ordering:
        assert net.hidden_0.adn.D.p == dropout
        assert net.hidden_1.adn.D.p == dropout


def test_activation_parameters():
    act = ("ELU", {"alpha": 0.1})
    output_act = ("ELU", {"alpha": 0.2})
    net = MLP(
        in_channels=10,
        out_channels=2,
        hidden_channels=[6, 4],
        act=act,
        output_act=output_act,
    )
    assert isinstance(net.hidden_0.adn.A, ELU)
    assert net.hidden_0.adn.A.alpha == 0.1
    assert isinstance(net.hidden_1.adn.A, ELU)
    assert net.hidden_1.adn.A.alpha == 0.1
    assert isinstance(net.output.output_act, ELU)
    assert net.output.output_act.alpha == 0.2

    net = MLP(in_channels=10, out_channels=2, hidden_channels=[6, 4], act=None)
    with pytest.raises(AttributeError):
        net.hidden_0.adn.A
    with pytest.raises(AttributeError):
        net.hidden_1.adn.A
    assert isinstance(net.output, Linear)


def test_norm_parameters():
    norm = ("instance", {"momentum": 1.0})
    net = MLP(in_channels=10, out_channels=2, hidden_channels=[6, 4], norm=norm)
    assert isinstance(net.hidden_0.adn.N, InstanceNorm1d)
    assert net.hidden_0.adn.N.momentum == 1.0
    assert isinstance(net.hidden_1.adn.N, InstanceNorm1d)
    assert net.hidden_1.adn.N.momentum == 1.0

    net = MLP(in_channels=10, out_channels=2, hidden_channels=[6, 4], act=None)
    with pytest.raises(AttributeError):
        net.layer_0[1].N
    with pytest.raises(AttributeError):
        net.layer_1[1].N


@pytest.mark.parametrize("adn_ordering", ["DAN", "NA", "A"])
def test_adn_ordering(adn_ordering):
    net = MLP(
        in_channels=10,
        out_channels=2,
        hidden_channels=[6, 4],
        dropout=0.1,
        adn_ordering=adn_ordering,
        act="elu",
        norm="instance",
    )
    objects = {"D": Dropout, "N": InstanceNorm1d, "A": ELU}
    for i, letter in enumerate(adn_ordering):
        assert isinstance(net.hidden_0.adn[i], objects[letter])
        assert isinstance(net.hidden_1.adn[i], objects[letter])
    for letter in set(["A", "D", "N"]) - set(adn_ordering):
        with pytest.raises(AttributeError):
            getattr(net.hidden_0.adn, letter)
        with pytest.raises(AttributeError):
            getattr(net.hidden_1.adn, letter)


def test_checks():
    with pytest.raises(ValueError):
        MLP(in_channels=10, out_channels=2, hidden_channels=[6, 4], norm="group")
