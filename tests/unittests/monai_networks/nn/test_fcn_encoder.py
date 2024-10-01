import pytest
import torch
from torch.nn import ELU, AvgPool2d, Conv2d, Dropout, InstanceNorm2d, MaxPool2d

from clinicadl.monai_networks.nn import FCNEncoder
from clinicadl.monai_networks.nn.layers import ActFunction


@pytest.fixture
def input_tensor():
    return torch.randn(2, 1, 55, 54)


@pytest.mark.skip()
def _check_output(output, expected_out_channels):
    output_shape = output.shape
    return len(output_shape) == 4 and output_shape[1] == expected_out_channels


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_activations(input_tensor, act):
    net = FCNEncoder(
        in_shape=input_tensor.shape[1:], channels=[2, 4, 1], act=act, output_act=act
    )
    output = net(input_tensor)
    assert _check_output(output, expected_out_channels=1)


@pytest.mark.parametrize(
    "kernel_size,stride,padding,dilation,pooling,pooling_indices,norm,dropout,bias,adn_ordering",
    [
        (3, 1, 0, 1, ("max", {"kernel_size": 3}), [1], "batch", None, True, "ADN"),
        (
            (4, 4),
            (2, 1),
            2,
            2,
            ("max", {"kernel_size": 2}),
            [0, 1],
            "layer",
            0.5,
            False,
            "DAN",
        ),
        (
            5,
            1,
            (2, 1),
            1,
            [("avg", {"kernel_size": 2}), ("max", {"kernel_size": 2})],
            [0, 1],
            "syncbatch",
            0.5,
            True,
            "NA",
        ),
        (5, 1, 0, (1, 2), None, [0, 1], "instance", 0.0, True, "DN"),
        (
            5,
            1,
            2,
            1,
            ("avg", {"kernel_size": 2}),
            None,
            ("group", {"num_groups": 2}),
            None,
            True,
            "N",
        ),
        (5, 1, 2, 1, ("avg", {"kernel_size": 2}), None, None, None, True, ""),
    ],
)
def test_params(
    input_tensor,
    kernel_size,
    stride,
    padding,
    dilation,
    pooling,
    pooling_indices,
    norm,
    dropout,
    bias,
    adn_ordering,
):
    net = FCNEncoder(
        in_shape=input_tensor.shape[1:],
        channels=[2, 4, 1],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        pooling=pooling,
        pooling_indices=pooling_indices,
        dropout=dropout,
        act=None,
        norm=norm,
        bias=bias,
        adn_ordering=adn_ordering,
    )
    output = net(input_tensor)
    assert _check_output(output, expected_out_channels=1)
    assert output.shape[2:] == net.final_size
    assert isinstance(net.layer_2.conv, Conv2d)
    with pytest.raises(IndexError):
        net.layer_2[1]  # no adn at the end

    named_layers = list(net.named_children())
    if pooling and pooling_indices and pooling_indices != []:
        for i, idx in enumerate(pooling_indices):
            name, layer = named_layers[idx + 1 + i]
            assert name == f"pool_{i}"
            if net.pooling[i][0] == "max":
                assert isinstance(layer, MaxPool2d)
            else:
                assert isinstance(layer, AvgPool2d)
    else:
        for name, layer in named_layers:
            assert not isinstance(layer, AvgPool2d) or isinstance(layer, MaxPool2d)
            assert "pool" not in name

    assert (
        net.layer_0.conv.kernel_size == kernel_size
        if isinstance(kernel_size, tuple)
        else (kernel_size, kernel_size)
    )
    assert (
        net.layer_0.conv.stride == stride
        if isinstance(stride, tuple)
        else (stride, stride)
    )
    assert (
        net.layer_0.conv.padding == padding
        if isinstance(padding, tuple)
        else (padding, padding)
    )
    assert (
        net.layer_0.conv.dilation == dilation
        if isinstance(dilation, tuple)
        else (dilation, dilation)
    )

    if bias:
        assert len(net.layer_0.conv.bias) > 0
        assert len(net.layer_1.conv.bias) > 0
        assert len(net.layer_2.conv.bias) > 0
    else:
        assert net.layer_0.conv.bias is None
        assert net.layer_1.conv.bias is None
        assert net.layer_2.conv.bias is None
    if isinstance(dropout, float) and "D" in adn_ordering:
        assert net.layer_0.adn.D.p == dropout
        assert net.layer_1.adn.D.p == dropout


def test_activation_parameters(input_tensor):
    act = ("ELU", {"alpha": 0.1})
    output_act = ("ELU", {"alpha": 0.2})
    net = FCNEncoder(
        in_shape=input_tensor.shape[1:],
        channels=[2, 4, 1],
        act=act,
        output_act=output_act,
    )
    assert isinstance(net.layer_0.adn.A, ELU)
    assert net.layer_0.adn.A.alpha == 0.1
    assert isinstance(net.layer_1.adn.A, ELU)
    assert net.layer_1.adn.A.alpha == 0.1
    assert isinstance(net.output_act, ELU)
    assert net.output_act.alpha == 0.2

    net = FCNEncoder(in_shape=input_tensor.shape[1:], channels=[2, 4, 1], act=None)
    with pytest.raises(AttributeError):
        net.layer_0.adn.A
    with pytest.raises(AttributeError):
        net.layer_1.adn.A
    with pytest.raises(AttributeError):
        net.output_act


def test_norm_parameters(input_tensor):
    norm = ("instance", {"momentum": 1.0})
    net = FCNEncoder(in_shape=input_tensor.shape[1:], channels=[2, 4, 1], norm=norm)
    assert isinstance(net.layer_0.adn.N, InstanceNorm2d)
    assert net.layer_0.adn.N.momentum == 1.0
    assert isinstance(net.layer_1.adn.N, InstanceNorm2d)
    assert net.layer_1.adn.N.momentum == 1.0

    net = FCNEncoder(in_shape=input_tensor.shape[1:], channels=[2, 4, 1], norm=None)
    with pytest.raises(AttributeError):
        net.layer_0.adn.N
    with pytest.raises(AttributeError):
        net.layer_1.adn.N


def test_pool_parameters(input_tensor):
    pooling = ("avg", {"kernel_size": 3, "stride": 2})
    net = FCNEncoder(
        in_shape=input_tensor.shape[1:],
        channels=[2, 4, 1],
        pooling=pooling,
        pooling_indices=[1],
    )
    assert isinstance(net.pool_0, AvgPool2d)
    assert net.pool_0.stride == 2
    assert net.pool_0.kernel_size == 3


@pytest.mark.parametrize("adn_ordering", ["DAN", "NA", "A"])
def test_adn_ordering(input_tensor, adn_ordering):
    net = FCNEncoder(
        in_shape=input_tensor.shape[1:],
        channels=[2, 4, 1],
        dropout=0.1,
        adn_ordering=adn_ordering,
        act="elu",
        norm="instance",
    )
    objects = {"D": Dropout, "N": InstanceNorm2d, "A": ELU}
    for i, letter in enumerate(adn_ordering):
        assert isinstance(net.layer_0.adn[i], objects[letter])
        assert isinstance(net.layer_1.adn[i], objects[letter])
    for letter in set(["A", "D", "N"]) - set(adn_ordering):
        with pytest.raises(AttributeError):
            getattr(net.layer_0.adn, letter)
        with pytest.raises(AttributeError):
            getattr(net.layer_1.adn, letter)


@pytest.mark.parametrize(
    "input_tensor", [torch.randn(2, 1, 16), torch.randn(2, 3, 20, 21, 22)]
)
def test_other_dimensions(input_tensor):
    net = FCNEncoder(
        in_shape=input_tensor.shape[1:],
        channels=[2, 4, 1],
    )
    output_shape = net(input_tensor).shape
    assert len(output_shape) == len(input_tensor.shape) and output_shape[1] == 1
    assert output_shape[2:] == net.final_size


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_size": (3, 3, 3)},
        {"stride": [1, 1]},
        {"padding": [1, 1]},
        {"dilation": (1,)},
        {"pooling_indices": [0, 1, 2]},
        {"pooling": "avg", "pooling_indices": [0]},
        {"norm": "group"},
    ],
)
def test_checks(input_tensor, kwargs):
    with pytest.raises(ValueError):
        FCNEncoder(in_shape=input_tensor.shape[1:], channels=[2, 4, 1], **kwargs)


@pytest.mark.parametrize(
    "pooling,error",
    [
        (None, False),
        ("abc", True),
        ("max", True),
        (("max",), True),
        (("max", 3), True),
        (("avg", {"stride": 1}), True),
        (("avg", {"kernel_size": 1}), False),
        (("avg", {"kernel_size": 1, "stride": 1}), False),
        (("abc", {"kernel_size": 1, "stride": 1}), True),
        ([("avg", {"kernel_size": 1}), ("max", {"kernel_size": 1})], False),
        ([("avg", {"kernel_size": 1}), None], True),
        ([("avg", {"kernel_size": 1}), "max"], True),
        ([("avg", {"kernel_size": 1}), ("max", 3)], True),
        ([("avg", {"kernel_size": 1}), ("max", {"stride": 1})], True),
        (
            [
                ("avg", {"kernel_size": 1}),
                ("max", {"stride": 1}),
                ("max", {"stride": 1}),
            ],
            True,
        ),
    ],
)
def test_check_pool_layers(pooling, error):
    if error:
        with pytest.raises(ValueError):
            FCNEncoder(
                in_shape=(1, 10, 10),
                channels=[2, 4, 1],
                pooling=pooling,
                pooling_indices=[0, 1],
            )
    else:
        FCNEncoder(
            in_shape=(1, 10, 10),
            channels=[2, 4, 1],
            pooling=pooling,
            pooling_indices=[0, 1],
        )
