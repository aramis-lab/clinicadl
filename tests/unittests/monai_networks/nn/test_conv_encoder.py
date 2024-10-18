import pytest
import torch
from torch.nn import (
    ELU,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    AvgPool2d,
    Conv2d,
    Dropout,
    InstanceNorm2d,
    MaxPool2d,
)

from clinicadl.monai_networks.nn import ConvEncoder
from clinicadl.monai_networks.nn.layers.utils import ActFunction


@pytest.fixture
def input_tensor():
    return torch.randn(2, 1, 55, 54)


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_activations(input_tensor, act):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        act=act,
        output_act=act,
    )
    output_shape = net(input_tensor).shape
    assert len(output_shape) == 4 and output_shape[1] == 1


@pytest.mark.parametrize(
    "kernel_size,stride,padding,dilation,pooling,pooling_indices,norm,dropout,bias,adn_ordering",
    [
        (
            3,
            1,
            0,
            1,
            ("adaptivemax", {"output_size": 1}),
            [2],
            "batch",
            None,
            True,
            "ADN",
        ),
        (
            (4, 4),
            (2, 1),
            2,
            2,
            ("max", {"kernel_size": 2}),
            [0, 1],
            "instance",
            0.5,
            False,
            "DAN",
        ),
        (
            5,
            1,
            (2, 1),
            1,
            [
                ("avg", {"kernel_size": 2}),
                ("max", {"kernel_size": 2}),
                ("adaptiveavg", {"output_size": (2, 3)}),
            ],
            [-1, 1, 2],
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
    batch_size, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    # test output size
    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
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
        _input_size=input_size,
    )
    output = net(input_tensor)
    assert output.shape == (batch_size, 1, *net.final_size)

    # other checks
    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
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
    assert isinstance(net.layer2.conv, Conv2d)
    with pytest.raises(IndexError):
        net.layer2[1]  # no adn at the end

    named_layers = list(net.named_children())
    if pooling and pooling_indices and pooling_indices != []:
        for i, idx in enumerate(pooling_indices):
            name, layer = named_layers[idx + 1 + i]
            if idx == -1:
                assert name == "init_pool"
            else:
                assert name == f"pool{idx}"
            pooling_mode = net.pooling[i][0]
            if pooling_mode == "max":
                assert isinstance(layer, MaxPool2d)
            elif pooling_mode == "avg":
                assert isinstance(layer, AvgPool2d)
            elif pooling_mode == "adaptivemax":
                assert isinstance(layer, AdaptiveMaxPool2d)
            else:
                assert isinstance(layer, AdaptiveAvgPool2d)
    else:
        for name, layer in named_layers:
            assert not isinstance(layer, AvgPool2d) or isinstance(layer, MaxPool2d)
            assert "pool" not in name

    assert (
        net.layer0.conv.kernel_size == kernel_size
        if isinstance(kernel_size, tuple)
        else (kernel_size, kernel_size)
    )
    assert (
        net.layer0.conv.stride == stride
        if isinstance(stride, tuple)
        else (stride, stride)
    )
    assert (
        net.layer0.conv.padding == padding
        if isinstance(padding, tuple)
        else (padding, padding)
    )
    assert (
        net.layer0.conv.dilation == dilation
        if isinstance(dilation, tuple)
        else (dilation, dilation)
    )

    if bias:
        assert len(net.layer0.conv.bias) > 0
        assert len(net.layer1.conv.bias) > 0
        assert len(net.layer2.conv.bias) > 0
    else:
        assert net.layer0.conv.bias is None
        assert net.layer1.conv.bias is None
        assert net.layer2.conv.bias is None
    if isinstance(dropout, float) and "D" in adn_ordering:
        assert net.layer0.adn.D.p == dropout
        assert net.layer1.adn.D.p == dropout


def test_activation_parameters(input_tensor):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    act = ("ELU", {"alpha": 0.1})
    output_act = ("ELU", {"alpha": 0.2})
    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        act=act,
        output_act=output_act,
    )
    assert isinstance(net.layer0.adn.A, ELU)
    assert net.layer0.adn.A.alpha == 0.1
    assert isinstance(net.layer1.adn.A, ELU)
    assert net.layer1.adn.A.alpha == 0.1
    assert isinstance(net.output_act, ELU)
    assert net.output_act.alpha == 0.2

    net = ConvEncoder(
        spatial_dims=spatial_dims, in_channels=in_channels, channels=[2, 4, 1], act=None
    )
    with pytest.raises(AttributeError):
        net.layer0.adn.A
    with pytest.raises(AttributeError):
        net.layer1.adn.A
    assert net.output_act is None


def test_norm_parameters(input_tensor):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    norm = ("instance", {"momentum": 1.0})
    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        norm=norm,
    )
    assert isinstance(net.layer0.adn.N, InstanceNorm2d)
    assert net.layer0.adn.N.momentum == 1.0
    assert isinstance(net.layer1.adn.N, InstanceNorm2d)
    assert net.layer1.adn.N.momentum == 1.0

    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        norm=None,
    )
    with pytest.raises(AttributeError):
        net.layer0.adn.N
    with pytest.raises(AttributeError):
        net.layer1.adn.N


def test_pool_parameters(input_tensor):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    pooling = ("avg", {"kernel_size": 3, "stride": 2})
    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        pooling=pooling,
        pooling_indices=[1],
    )
    assert isinstance(net.pool1, AvgPool2d)
    assert net.pool1.stride == 2
    assert net.pool1.kernel_size == 3


@pytest.mark.parametrize("adn_ordering", ["DAN", "NA", "A"])
def test_adn_ordering(input_tensor, adn_ordering):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        dropout=0.1,
        adn_ordering=adn_ordering,
        act="elu",
        norm="instance",
    )
    objects = {"D": Dropout, "N": InstanceNorm2d, "A": ELU}
    for i, letter in enumerate(adn_ordering):
        assert isinstance(net.layer0.adn[i], objects[letter])
        assert isinstance(net.layer1.adn[i], objects[letter])
    for letter in set(["A", "D", "N"]) - set(adn_ordering):
        with pytest.raises(AttributeError):
            getattr(net.layer0.adn, letter)
        with pytest.raises(AttributeError):
            getattr(net.layer1.adn, letter)


@pytest.mark.parametrize(
    "input_tensor", [torch.randn(2, 1, 16), torch.randn(2, 3, 20, 21, 22)]
)
def test_other_dimensions(input_tensor):
    batch_size, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    net = ConvEncoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        _input_size=input_size,
    )
    output = net(input_tensor)
    assert output.shape == (batch_size, 1, *net.final_size)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kernel_size": (3, 3, 3)},
        {"stride": [1, 1]},
        {"padding": [1, 1]},
        {"dilation": (1,)},
        {"pooling_indices": [0, 1, 2, 3]},
        {"pooling": "avg", "pooling_indices": [0]},
        {"norm": "group"},
        {"_input_size": (1, 10, 10), "stride": 2, "channels": [2, 4, 6, 8]},
    ],
)
def test_checks(kwargs):
    if "channels" not in kwargs:
        kwargs["channels"] = [2, 4, 1]
    if "in_channels" not in kwargs:
        kwargs["in_channels"] = 1
    if "spatial_dims" not in kwargs:
        kwargs["spatial_dims"] = 2
    with pytest.raises(ValueError):
        ConvEncoder(**kwargs)


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
def test_check_pool_layers(input_tensor, pooling, error):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    if error:
        with pytest.raises(ValueError):
            ConvEncoder(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                channels=[2, 4, 1],
                pooling=pooling,
                pooling_indices=[0, 1],
            )
    else:
        ConvEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=[2, 4, 1],
            pooling=pooling,
            pooling_indices=[0, 1],
        )
