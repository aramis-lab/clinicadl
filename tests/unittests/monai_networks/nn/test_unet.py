import pytest
import torch

from clinicadl.monai_networks.nn import UNet
from clinicadl.monai_networks.nn.layers.utils import ActFunction

INPUT_1D = torch.randn(2, 1, 16)
INPUT_2D = torch.randn(2, 2, 32, 64)
INPUT_3D = torch.randn(2, 3, 16, 32, 8)


@pytest.mark.parametrize(
    "input_tensor,out_channels,channels,act,output_act,dropout,error",
    [
        (INPUT_1D, 1, (2, 3, 4), "relu", "sigmoid", None, False),
        (INPUT_2D, 1, (2, 4, 5), "relu", None, 0.0, False),
        (INPUT_3D, 2, (2, 3), None, ("softmax", {"dim": 1}), 0.1, False),
        (
            INPUT_3D,
            2,
            (2,),
            None,
            ("softmax", {"dim": 1}),
            0.1,
            True,
        ),  # channels length is less than 2
    ],
)
def test_unet(input_tensor, out_channels, channels, act, output_act, dropout, error):
    batch_size, in_channels, *img_size = input_tensor.shape
    spatial_dims = len(img_size)
    if error:
        with pytest.raises(ValueError):
            UNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                act=act,
                output_act=output_act,
                dropout=dropout,
            )
    else:
        net = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            act=act,
            output_act=output_act,
            dropout=dropout,
        )

        out = net(input_tensor)
        assert out.shape == (batch_size, out_channels, *img_size)

        if output_act:
            assert net.output_act is not None
        else:
            assert net.output_act is None

        assert net.doubleconv[1].conv.out_channels == channels[0]
        if dropout:
            assert net.doubleconv[1].adn.D.p == dropout
        else:
            with pytest.raises(AttributeError):
                net.doubleconv[1].conv.adn.D

        for i in range(1, len(channels)):
            down = getattr(net, f"down{i}").doubleconv
            up = getattr(net, f"doubleconv{i}")
            assert down[0].conv.in_channels == channels[i - 1]
            assert down[1].conv.out_channels == channels[i]
            assert up[0].conv.in_channels == channels[i - 1] * 2
            assert up[1].conv.out_channels == channels[i - 1]
            for m in (down, up):
                if dropout is not None:
                    assert m[1].adn.D.p == dropout
                else:
                    with pytest.raises(AttributeError):
                        m[1].adn.D
        with pytest.raises(AttributeError):
            down = getattr(net, f"down{i+1}")
        with pytest.raises(AttributeError):
            getattr(net, f"doubleconv{i+1}")


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_activations(act):
    batch_size, in_channels, *img_size = INPUT_2D.shape
    net = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=2,
        channels=(2, 4),
        act=act,
        output_act=act,
    )
    assert net(INPUT_2D).shape == (batch_size, 2, *img_size)


def test_activation_parameters():
    in_channels = INPUT_2D.shape[1]
    act = ("ELU", {"alpha": 0.1})
    output_act = ("ELU", {"alpha": 0.2})
    net = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=2,
        channels=(2, 4),
        act=act,
        output_act=output_act,
    )
    assert isinstance(net.doubleconv[0].adn.A, torch.nn.ELU)
    assert net.doubleconv[0].adn.A.alpha == 0.1

    assert isinstance(net.down1.doubleconv[0].adn.A, torch.nn.ELU)
    assert net.down1.doubleconv[0].adn.A.alpha == 0.1

    assert isinstance(net.upsample1[1].adn.A, torch.nn.ELU)
    assert net.upsample1[1].adn.A.alpha == 0.1

    assert isinstance(net.doubleconv1[1].adn.A, torch.nn.ELU)
    assert net.doubleconv1[1].adn.A.alpha == 0.1

    assert isinstance(net.output_act, torch.nn.ELU)
    assert net.output_act.alpha == 0.2
