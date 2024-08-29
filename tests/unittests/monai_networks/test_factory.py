import pytest
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBottleneck
from torch.nn import Conv2d

from clinicadl.monai_networks import get_network
from clinicadl.monai_networks.config import create_network_config


@pytest.mark.parametrize(
    "network_name,params",
    [
        (
            "AutoEncoder",
            {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "channels": [2, 2],
                "strides": [1, 1],
            },
        ),
        (
            "VarAutoEncoder",
            {
                "spatial_dims": 3,
                "in_shape": (1, 16, 16, 16),
                "out_channels": 1,
                "latent_size": 16,
                "channels": [2, 2],
                "strides": [1, 1],
            },
        ),
        (
            "Regressor",
            {
                "in_shape": (1, 16, 16, 16),
                "out_shape": (1, 16, 16, 16),
                "channels": [2, 2],
                "strides": [1, 1],
            },
        ),
        (
            "Classifier",
            {
                "in_shape": (1, 16, 16, 16),
                "classes": 2,
                "channels": [2, 2],
                "strides": [1, 1],
            },
        ),
        (
            "Discriminator",
            {"in_shape": (1, 16, 16, 16), "channels": [2, 2], "strides": [1, 1]},
        ),
        (
            "Critic",
            {"in_shape": (1, 16, 16, 16), "channels": [2, 2], "strides": [1, 1]},
        ),
        ("DenseNet", {"spatial_dims": 3, "in_channels": 1, "out_channels": 1}),
        (
            "FullyConnectedNet",
            {"in_channels": 3, "out_channels": 1, "hidden_channels": [2, 3]},
        ),
        (
            "VarFullyConnectedNet",
            {
                "in_channels": 1,
                "out_channels": 1,
                "latent_size": 16,
                "encode_channels": [2, 2],
                "decode_channels": [2, 2],
            },
        ),
        (
            "Generator",
            {
                "latent_shape": (3,),
                "start_shape": (1, 16, 16, 16),
                "channels": [2, 2],
                "strides": [1, 1],
            },
        ),
        (
            "ResNet",
            {
                "block": "bottleneck",
                "layers": (4, 4, 4, 4),
                "block_inplanes": (5, 5, 5, 5),
                "spatial_dims": 2,
            },
        ),
        ("ResNetFeatures", {"model_name": "resnet10"}),
        ("SegResNet", {}),
        (
            "UNet",
            {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "channels": [2, 2, 2],
                "strides": [1, 1],
            },
        ),
        (
            "AttentionUnet",
            {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 1,
                "channels": [2, 2, 2],
                "strides": [1, 1],
            },
        ),
        ("ViT", {"in_channels": 3, "img_size": 16, "patch_size": 4}),
        ("ViTAutoEnc", {"in_channels": 3, "img_size": 16, "patch_size": 4}),
    ],
)
def test_get_network(network_name, params):
    config = create_network_config(network_name)(**params)
    network, updated_config = get_network(config)

    if network_name == "ResNet":
        assert isinstance(network, ResNet)
        assert isinstance(network.layer1[0], ResNetBottleneck)
        assert len(network.layer1) == 4
        assert network.layer1[0].conv1.in_channels == 5
        assert isinstance(network.layer1[0].conv1, Conv2d)

        assert updated_config.network == "ResNet"
        assert updated_config.block == "bottleneck"
        assert updated_config.layers == (4, 4, 4, 4)
        assert updated_config.block_inplanes == (5, 5, 5, 5)
        assert updated_config.spatial_dims == 2
        assert updated_config.conv1_t_size == 7
        assert updated_config.act == ("relu", {"inplace": True})
