import pytest
import torch

from clinicadl.networks.nn import AttentionUNet

INPUT_1D = torch.randn(2, 1, 16)
INPUT_2D = torch.randn(2, 2, 32, 64)
INPUT_3D = torch.randn(2, 3, 16, 32, 8)


@pytest.mark.parametrize(
    "input_tensor,out_channels,channels,act,output_act,dropout,error",
    [
        (INPUT_2D, 1, (2, 3, 4), "relu", "sigmoid", None, False),
        (INPUT_3D, 1, (2, 4, 5), ("softmax", {"dim": 1}), None, 0.0, False),
        (INPUT_3D, 2, (2, 3), None, ("softmax", {"dim": 1}), 0.1, True),
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
def test_attentionunet(
    input_tensor, out_channels, channels, act, output_act, dropout, error
):
    batch_size, in_channels, *img_size = input_tensor.shape
    spatial_dims = len(img_size)
    if error:
        with pytest.raises(ValueError):
            AttentionUNet(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                act=act,
                output_act=output_act,
                dropout=dropout,
            )
    else:
        net = AttentionUNet(
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
            up = getattr(net, f"up{i}").doubleconv
            att = getattr(net, f"up{i}").attention
            assert down[0].conv.in_channels == channels[i - 1]
            assert down[1].conv.out_channels == channels[i]
            assert att.W_g[0].out_channels == channels[i - 1] // 2
            assert att.W_x[0].out_channels == channels[i - 1] // 2
            assert up[0].conv.in_channels == channels[i - 1] * 2
            assert up[1].conv.out_channels == channels[i - 1]
            for m in (down, up):
                if dropout is not None:
                    assert m[1].adn.D.p == dropout
                else:
                    with pytest.raises(AttributeError):
                        m[1].adn.D
        with pytest.raises(AttributeError):
            down = getattr(net, f"down{i + 1}")
        with pytest.raises(AttributeError):
            getattr(net, f"up{i + 1}")
