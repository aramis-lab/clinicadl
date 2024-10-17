import pytest
import torch

from clinicadl.monai_networks.nn import ResNet, get_resnet
from clinicadl.monai_networks.nn.layers.resnet import ResNetBlock, ResNetBottleneck
from clinicadl.monai_networks.nn.layers.utils import ActFunction
from clinicadl.monai_networks.nn.resnet import CommonResNet

INPUT_1D = torch.randn(3, 1, 16)
INPUT_2D = torch.randn(3, 2, 15, 16)
INPUT_3D = torch.randn(3, 3, 20, 21, 22)


@pytest.mark.parametrize(
    "input_tensor,num_outputs,block_type,n_res_blocks,n_features,init_conv_size,init_conv_stride,bottleneck_reduction,act,output_act",
    [
        (INPUT_1D, 2, "basic", (2, 3), (4, 8), 7, 1, 2, "relu", None),
        (
            INPUT_2D,
            None,
            "bottleneck",
            (3, 2, 2),
            (8, 12, 16),
            5,
            (2, 1),
            4,
            "elu",
            "sigmoid",
        ),
        (INPUT_3D, 1, "bottleneck", (2,), (3,), (4, 3, 4), 2, 1, "tanh", "sigmoid"),
    ],
)
def test_resnet(
    input_tensor,
    num_outputs,
    block_type,
    n_res_blocks,
    n_features,
    init_conv_size,
    init_conv_stride,
    bottleneck_reduction,
    act,
    output_act,
):
    batch_size = input_tensor.shape[0]
    spatial_dims = len(input_tensor.shape[2:])
    net = ResNet(
        spatial_dims=spatial_dims,
        in_channels=input_tensor.shape[1],
        num_outputs=num_outputs,
        block_type=block_type,
        n_res_blocks=n_res_blocks,
        n_features=n_features,
        init_conv_size=init_conv_size,
        init_conv_stride=init_conv_stride,
        bottleneck_reduction=bottleneck_reduction,
        act=act,
        output_act=output_act,
    )
    output = net(input_tensor)

    if num_outputs:
        assert output.shape == (batch_size, num_outputs)
    else:
        assert len(output.shape) == len(input_tensor.shape)

    if output_act and num_outputs:
        assert net.fc.output_act is not None
    elif output_act and num_outputs is None:
        with pytest.raises(AttributeError):
            net.fc.output_act

    for i, (n_blocks, n_feats) in enumerate(zip(n_res_blocks, n_features), start=1):
        layer = getattr(net, f"layer{i}")
        for k in range(n_blocks):
            res_block = layer[k]
            if block_type == "basic":
                assert isinstance(res_block, ResNetBlock)
            else:
                assert isinstance(res_block, ResNetBottleneck)
        if block_type == "basic":
            assert res_block.conv2.out_channels == n_feats
        else:
            assert res_block.conv1.out_channels == n_feats // bottleneck_reduction
            assert res_block.conv3.out_channels == n_feats
        with pytest.raises(IndexError):
            layer[k + 1]
    with pytest.raises(AttributeError):
        getattr(net, f"layer{i+1}")

    assert (
        net.conv0.kernel_size == init_conv_size
        if isinstance(init_conv_size, tuple)
        else (init_conv_size,) * spatial_dims
    )
    assert (
        net.conv0.stride == init_conv_stride
        if isinstance(init_conv_stride, tuple)
        else (init_conv_stride,) * spatial_dims
    )


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_activations(act):
    batch_size = INPUT_2D.shape[0]
    net = ResNet(
        spatial_dims=len(INPUT_2D.shape[2:]),
        in_channels=INPUT_2D.shape[1],
        num_outputs=2,
        n_features=(8, 16),
        n_res_blocks=(2, 2),
        act=act,
    )
    assert net(INPUT_2D).shape == (batch_size, 2)


def test_activation_parameters():
    act = ("ELU", {"alpha": 0.1})
    output_act = ("ELU", {"alpha": 0.2})
    net = ResNet(
        spatial_dims=len(INPUT_2D.shape[2:]),
        in_channels=INPUT_2D.shape[1],
        num_outputs=2,
        n_features=(8, 16),
        n_res_blocks=(2, 2),
        act=act,
        output_act=output_act,
    )
    assert isinstance(net.layer1[0].act1, torch.nn.ELU)
    assert net.layer1[0].act1.alpha == 0.1
    assert isinstance(net.layer2[1].act2, torch.nn.ELU)
    assert net.layer2[1].act2.alpha == 0.1
    assert isinstance(net.act0, torch.nn.ELU)
    assert net.act0.alpha == 0.1
    assert isinstance(net.fc.output_act, torch.nn.ELU)
    assert net.fc.output_act.alpha == 0.2


@pytest.mark.parametrize(
    "name,num_outputs,output_act",
    [
        (CommonResNet.RESNET_18, 1, "sigmoid"),
        (CommonResNet.RESNET_34, 2, None),
        (CommonResNet.RESNET_50, None, "sigmoid"),
        (CommonResNet.RESNET_101, None, None),
        (CommonResNet.RESNET_152, None, None),
    ],
)
def test_get_resnet(name, num_outputs, output_act):
    resnet = get_resnet(
        name, num_outputs=num_outputs, output_act=output_act, pretrained=True
    )
    if num_outputs:
        assert resnet.fc.out.out_features == num_outputs
    else:
        assert resnet.fc is None

    if output_act and num_outputs:
        assert resnet.fc.output_act is not None
    elif output_act and num_outputs is None:
        with pytest.raises(AttributeError):
            resnet.fc.output_act


def test_get_resnet_output():
    from torchvision.models import resnet18

    resnet = get_resnet(CommonResNet.RESNET_18, num_outputs=None, pretrained=True)
    gt = resnet18(weights="DEFAULT")
    gt.avgpool = torch.nn.Identity()
    gt.fc = torch.nn.Identity()
    x = torch.randn(1, 3, 128, 128)
    assert (torch.flatten(resnet(x), start_dim=1) == gt(x)).all()
