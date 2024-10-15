import pytest
import torch
from torch.nn import ELU, ConvTranspose2d, Dropout, InstanceNorm2d, Upsample

from clinicadl.monai_networks.nn import ConvDecoder
from clinicadl.monai_networks.nn.layers.utils import ActFunction


@pytest.fixture
def input_tensor():
    return torch.randn(2, 1, 8, 8)


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_activations(input_tensor, act):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)
    net = ConvDecoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        act=act,
        output_act=act,
    )
    output_shape = net(input_tensor).shape
    return len(output_shape) == 4 and output_shape[1] == 1


@pytest.mark.parametrize(
    "kernel_size,stride,padding,output_padding,dilation,unpooling,unpooling_indices,norm,dropout,bias,adn_ordering",
    [
        (
            3,
            2,
            0,
            1,
            1,
            ("upsample", {"scale_factor": 2}),
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
            (1, 0),
            2,
            ("upsample", {"scale_factor": 2}),
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
            0,
            1,
            [("upsample", {"size": (16, 16)}), ("convtranspose", {"kernel_size": 2})],
            [0, 1],
            "syncbatch",
            0.5,
            True,
            "NA",
        ),
        (5, 1, 0, 1, (2, 3), None, [0, 1], "instance", 0.0, True, "DN"),
        (
            5,
            1,
            2,
            0,
            1,
            ("convtranspose", {"kernel_size": 2}),
            None,
            ("group", {"num_groups": 2}),
            None,
            True,
            "N",
        ),
        (
            5,
            3,
            2,
            (2, 1),
            1,
            ("convtranspose", {"kernel_size": 2}),
            [0, 1],
            None,
            None,
            True,
            "",
        ),
    ],
)
def test_params(
    input_tensor,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    unpooling,
    unpooling_indices,
    norm,
    dropout,
    bias,
    adn_ordering,
):
    batch_size, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    # test size computation
    net = ConvDecoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        unpooling=unpooling,
        unpooling_indices=unpooling_indices,
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
    net = ConvDecoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        unpooling=unpooling,
        unpooling_indices=unpooling_indices,
        dropout=dropout,
        act=None,
        norm=norm,
        bias=bias,
        adn_ordering=adn_ordering,
    )
    assert isinstance(net.layer2[0], ConvTranspose2d)
    with pytest.raises(IndexError):
        net.layer2[1]  # no adn at the end

    named_layers = list(net.named_children())
    if unpooling and unpooling_indices and unpooling_indices != []:
        for i, idx in enumerate(unpooling_indices):
            name, layer = named_layers[idx + 1 + i]
            assert name == f"unpool{i}"
            if net.unpooling[i][0] == "upsample":
                assert isinstance(layer, Upsample)
            else:
                assert isinstance(layer, ConvTranspose2d)
    else:
        for name, layer in named_layers:
            assert not isinstance(layer, Upsample)
            assert "unpool" not in name

    assert (
        net.layer0[0].kernel_size == kernel_size
        if isinstance(kernel_size, tuple)
        else (kernel_size, kernel_size)
    )
    assert (
        net.layer0[0].stride == stride
        if isinstance(stride, tuple)
        else (stride, stride)
    )
    assert (
        net.layer0[0].padding == padding
        if isinstance(padding, tuple)
        else (padding, padding)
    )
    assert (
        net.layer0[0].output_padding == output_padding
        if isinstance(output_padding, tuple)
        else (output_padding, output_padding)
    )
    assert (
        net.layer0[0].dilation == dilation
        if isinstance(dilation, tuple)
        else (dilation, dilation)
    )

    if bias:
        assert len(net.layer0[0].bias) > 0
        assert len(net.layer1[0].bias) > 0
        assert len(net.layer2[0].bias) > 0
    else:
        assert net.layer0[0].bias is None
        assert net.layer1[0].bias is None
        assert net.layer2[0].bias is None
    if isinstance(dropout, float) and "D" in adn_ordering:
        assert net.layer0[1].D.p == dropout
        assert net.layer1[1].D.p == dropout


def test_activation_parameters(input_tensor):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    act = ("ELU", {"alpha": 0.1})
    output_act = ("ELU", {"alpha": 0.2})
    net = ConvDecoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        act=act,
        output_act=output_act,
    )
    assert isinstance(net.layer0[1].A, ELU)
    assert net.layer0[1].A.alpha == 0.1
    assert isinstance(net.layer1[1].A, ELU)
    assert net.layer1[1].A.alpha == 0.1
    assert isinstance(net.output_act, ELU)
    assert net.output_act.alpha == 0.2

    net = ConvDecoder(
        spatial_dims=spatial_dims, in_channels=in_channels, channels=[2, 4, 1], act=None
    )
    with pytest.raises(AttributeError):
        net.layer0[1].A
    with pytest.raises(AttributeError):
        net.layer1[1].A
    assert net.output_act is None


def test_norm_parameters(input_tensor):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    norm = ("instance", {"momentum": 1.0})
    net = ConvDecoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        norm=norm,
    )
    assert isinstance(net.layer0[1].N, InstanceNorm2d)
    assert net.layer0[1].N.momentum == 1.0
    assert isinstance(net.layer1[1].N, InstanceNorm2d)
    assert net.layer1[1].N.momentum == 1.0

    net = ConvDecoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        norm=None,
    )
    with pytest.raises(AttributeError):
        net.layer0[1].N
    with pytest.raises(AttributeError):
        net.layer1[1].N


def test_unpool_parameters(input_tensor):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    unpooling = ("convtranspose", {"kernel_size": 3, "stride": 2})
    net = ConvDecoder(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        channels=[2, 4, 1],
        unpooling=unpooling,
        unpooling_indices=[1],
    )
    assert isinstance(net.unpool0, ConvTranspose2d)
    assert net.unpool0.stride == (2, 2)
    assert net.unpool0.kernel_size == (3, 3)


@pytest.mark.parametrize("adn_ordering", ["DAN", "NA", "A"])
def test_adn_ordering(input_tensor, adn_ordering):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    net = ConvDecoder(
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
        assert isinstance(net.layer0[1][i], objects[letter])
        assert isinstance(net.layer1[1][i], objects[letter])
    for letter in set(["A", "D", "N"]) - set(adn_ordering):
        with pytest.raises(AttributeError):
            getattr(net.layer0[1], letter)
        with pytest.raises(AttributeError):
            getattr(net.layer1[1], letter)


@pytest.mark.parametrize(
    "input_tensor", [torch.randn(2, 1, 16), torch.randn(2, 3, 20, 21, 22)]
)
def test_other_dimensions(input_tensor):
    batch_size, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    net = ConvDecoder(
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
        {"unpooling_indices": [0, 1, 2, 3]},
        {"unpooling": "upsample", "unpooling_indices": [0]},
        {"norm": "group"},
        {"norm": "layer"},
    ],
)
def test_checks(input_tensor, kwargs):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    with pytest.raises(ValueError):
        ConvDecoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=[2, 4, 1],
            **kwargs,
        )


@pytest.mark.parametrize(
    "unpooling,error",
    [
        (None, False),
        ("abc", True),
        ("upsample", True),
        (("upsample",), True),
        (("upsample", 2), True),
        (("convtranspose", {"kernel_size": 2}), False),
        (("upsample", {"scale_factor": 2}), False),
        (
            [("upsample", {"scale_factor": 2}), ("convtranspose", {"kernel_size": 2})],
            False,
        ),
        ([("upsample", {"scale_factor": 2}), None], True),
        ([("upsample", {"scale_factor": 2}), "convtranspose"], True),
        ([("upsample", {"scale_factor": 2}), ("convtranspose", 2)], True),
        (
            [
                ("upsample", {"scale_factor": 2}),
                ("convtranspose", {"kernel_size": 2}),
                ("convtranspose", {"kernel_size": 2}),
            ],
            True,
        ),
    ],
)
def test_check_unpool_layer(input_tensor, unpooling, error):
    _, in_channels, *input_size = input_tensor.shape
    spatial_dims = len(input_size)

    if error:
        with pytest.raises(ValueError):
            ConvDecoder(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                channels=[2, 4, 1],
                unpooling=unpooling,
                unpooling_indices=[0, 1],
            )
    else:
        ConvDecoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=[2, 4, 1],
            unpooling=unpooling,
            unpooling_indices=[0, 1],
        )
