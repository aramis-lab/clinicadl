import pytest
import torch

from clinicadl.monai_networks.nn import DenseNet, get_densenet
from clinicadl.monai_networks.nn.densenet import SOTADenseNet
from clinicadl.monai_networks.nn.layers.utils import ActFunction

INPUT_1D = torch.randn(3, 1, 16)
INPUT_2D = torch.randn(3, 2, 15, 16)
INPUT_3D = torch.randn(3, 3, 20, 21, 22)


@pytest.mark.parametrize(
    "input_tensor,num_outputs,n_dense_layers,init_features,growth_rate,bottleneck_factor,act,output_act,dropout",
    [
        (INPUT_1D, 2, (3, 4), 16, 8, 2, "relu", None, 0.1),
        (INPUT_2D, None, (3, 4, 2), 9, 5, 3, "elu", "sigmoid", 0.0),
        (INPUT_3D, 1, (2,), 4, 4, 2, "tanh", "sigmoid", 0.1),
    ],
)
def test_densenet(
    input_tensor,
    num_outputs,
    n_dense_layers,
    init_features,
    growth_rate,
    bottleneck_factor,
    act,
    output_act,
    dropout,
):
    batch_size = input_tensor.shape[0]
    net = DenseNet(
        spatial_dims=len(input_tensor.shape[2:]),
        in_channels=input_tensor.shape[1],
        num_outputs=num_outputs,
        n_dense_layers=n_dense_layers,
        init_features=init_features,
        growth_rate=growth_rate,
        bottleneck_factor=bottleneck_factor,
        act=act,
        output_act=output_act,
        dropout=dropout,
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

    features = net.features
    for i, n in enumerate(n_dense_layers, start=1):
        dense_block = getattr(features, f"denseblock{i}")
        for k in range(1, n + 1):
            dense_layer = getattr(dense_block, f"denselayer{k}").layers
            assert dense_layer.conv1.out_channels == growth_rate * bottleneck_factor
            assert dense_layer.conv2.out_channels == growth_rate
            if dropout:
                assert dense_layer.dropout.p == dropout
        with pytest.raises(AttributeError):
            getattr(dense_block, f"denseblock{n+1}")
    with pytest.raises(AttributeError):
        getattr(dense_block, f"denseblock{i+1}")

    assert features.conv0.out_channels == init_features


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_activations(act):
    batch_size = INPUT_2D.shape[0]
    net = DenseNet(
        spatial_dims=len(INPUT_2D.shape[2:]),
        in_channels=INPUT_2D.shape[1],
        n_dense_layers=(2, 2),
        num_outputs=2,
        act=act,
    )
    assert net(INPUT_2D).shape == (batch_size, 2)


def test_activation_parameters():
    act = ("ELU", {"alpha": 0.1})
    output_act = ("ELU", {"alpha": 0.2})
    net = DenseNet(
        spatial_dims=len(INPUT_2D.shape[2:]),
        in_channels=INPUT_2D.shape[1],
        num_outputs=2,
        n_dense_layers=(2, 2),
        act=act,
        output_act=output_act,
    )
    assert isinstance(net.features.denseblock1.denselayer1.layers.act1, torch.nn.ELU)
    assert net.features.denseblock1.denselayer1.layers.act1.alpha == 0.1
    assert isinstance(net.fc.output_act, torch.nn.ELU)
    assert net.fc.output_act.alpha == 0.2


@pytest.mark.parametrize(
    "name,num_outputs,output_act",
    [
        (SOTADenseNet.DENSENET_121, 1, "sigmoid"),
        (SOTADenseNet.DENSENET_161, 2, None),
        (SOTADenseNet.DENSENET_169, None, "sigmoid"),
        (SOTADenseNet.DENSENET_201, None, None),
    ],
)
def test_get_densenet(name, num_outputs, output_act):
    densenet = get_densenet(
        name, num_outputs=num_outputs, output_act=output_act, pretrained=True
    )
    if num_outputs:
        assert densenet.fc.out.out_features == num_outputs
    else:
        assert densenet.fc is None

    if output_act and num_outputs:
        assert densenet.fc.output_act is not None
    elif output_act and num_outputs is None:
        with pytest.raises(AttributeError):
            densenet.fc.output_act


def test_get_densenet_output():
    from torchvision.models import densenet121

    densenet = get_densenet(
        SOTADenseNet.DENSENET_121, num_outputs=None, pretrained=True
    ).features
    gt = densenet121(weights="DEFAULT").features
    x = torch.randn(1, 3, 128, 128)
    assert (densenet(x) == gt(x)).all()
