import pytest
import torch
from pydantic import ValidationError

from clinicadl.networks.nn import SEResNet, SEResNet50, SEResNet101, SEResNet152
from clinicadl.networks.nn.layers.senet import SEResNetBlock, SEResNetBottleneck

INPUT_1D = torch.randn(3, 1, 16)
INPUT_2D = torch.randn(3, 2, 15, 16)
INPUT_3D = torch.randn(3, 3, 20, 21, 22)


@pytest.mark.parametrize(
    "input_tensor,num_outputs,block_type,n_res_blocks,n_features,init_conv_size,init_conv_stride,bottleneck_reduction,act,output_act,se_reduction",
    [
        (INPUT_1D, 2, "basic", (2, 3), (4, 8), 7, 1, 2, "relu", None, 4),
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
            2,
        ),
        (INPUT_3D, 1, "bottleneck", (2,), (3,), (4, 3, 4), 2, 1, "tanh", "sigmoid", 2),
    ],
)
def test_seresnet(
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
    se_reduction,
):
    batch_size = input_tensor.shape[0]
    spatial_dims = len(input_tensor.shape[2:])
    net = SEResNet(
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
        se_reduction=se_reduction,
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
                assert isinstance(res_block, SEResNetBlock)
            else:
                assert isinstance(res_block, SEResNetBottleneck)
        if block_type == "basic":
            assert res_block.conv2.out_channels == n_feats
        else:
            assert res_block.conv1.out_channels == n_feats // bottleneck_reduction
            assert res_block.conv3.out_channels == n_feats
        with pytest.raises(IndexError):
            layer[k + 1]
    with pytest.raises(AttributeError):
        getattr(net, f"layer{i + 1}")

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


@pytest.mark.parametrize(
    "net,num_outputs,output_act",
    [
        (SEResNet50, 1, "sigmoid"),
        (SEResNet101, 2, None),
        (SEResNet152, None, "sigmoid"),
    ],
)
def test_literature(net, num_outputs, output_act):
    seresnet = net(
        num_outputs=num_outputs,
        output_act=output_act,
    )
    if num_outputs:
        assert seresnet.fc.out.out_features == num_outputs
    else:
        assert seresnet.fc is None

    if output_act and num_outputs:
        assert seresnet.fc.output_act is not None
    elif output_act and num_outputs is None:
        with pytest.raises(AttributeError):
            seresnet.fc.output_act


@pytest.mark.parametrize(
    "args,error",
    [
        (
            {
                "bottleneck_reduction": 2,
                "n_features": [3, 4],
                "se_reduction": 2,
                "block_type": "bottleneck",
            },
            True,
        ),
        ({"n_features": [2], "n_res_blocks": [2, 4], "se_reduction": 2}, True),
        ({"n_features": [2, 4], "n_res_blocks": [2, 4], "se_reduction": 3}, True),
        ({"n_features": [2, 4], "n_res_blocks": [2, 4], "se_reduction": 2}, False),
        ({"se_reduction": 0}, True),
    ],
)
def test_checks(args, error):
    args.update({"spatial_dims": 2, "in_channels": 2, "num_outputs": 1})
    if error:
        with pytest.raises(ValidationError):
            SEResNet(**args)
    else:
        _ = SEResNet(**args)
