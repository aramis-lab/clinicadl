import pytest

from clinicadl.monai_networks import (
    ImplementedNetworks,
    get_network,
    get_network_from_config,
)
from clinicadl.monai_networks.config.autoencoder import AutoEncoderConfig
from clinicadl.monai_networks.factory import _update_config_with_defaults
from clinicadl.monai_networks.nn import AutoEncoder

tested = []


@pytest.mark.parametrize(
    "network_name,params",
    [
        (
            "AutoEncoder",
            {
                "in_shape": (1, 64, 65),
                "latent_size": 1,
                "conv_args": {"channels": [2, 4]},
            },
        ),
        (
            "VAE",
            {
                "in_shape": (1, 64, 65),
                "latent_size": 1,
                "conv_args": {"channels": [2, 4]},
            },
        ),
        (
            "CNN",
            {
                "in_shape": (1, 64, 65),
                "num_outputs": 1,
                "conv_args": {"channels": [2, 4]},
            },
        ),
        (
            "Generator",
            {
                "latent_size": 1,
                "start_shape": (1, 5, 5),
                "conv_args": {"channels": [2, 4]},
            },
        ),
        (
            "ConvDecoder",
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "channels": [2, 4],
            },
        ),
        (
            "ConvEncoder",
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "channels": [2, 4],
            },
        ),
        (
            "MLP",
            {
                "in_channels": 1,
                "out_channels": 2,
                "hidden_channels": [2, 4],
            },
        ),
        (
            "AttentionUNet",
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 2,
            },
        ),
        (
            "UNet",
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "out_channels": 2,
            },
        ),
        (
            "ResNet",
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "num_outputs": 1,
            },
        ),
        (
            "DenseNet",
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "num_outputs": 1,
            },
        ),
        (
            "SEResNet",
            {
                "spatial_dims": 2,
                "in_channels": 1,
                "num_outputs": 1,
            },
        ),
        (
            "ViT",
            {
                "in_shape": (1, 64, 65),
                "patch_size": (4, 5),
                "num_outputs": 1,
            },
        ),
        (
            "ResNet-18",
            {
                "num_outputs": 1,
            },
        ),
        (
            "ResNet-34",
            {
                "num_outputs": 1,
            },
        ),
        (
            "ResNet-50",
            {
                "num_outputs": 1,
            },
        ),
        (
            "ResNet-101",
            {
                "num_outputs": 1,
            },
        ),
        (
            "ResNet-152",
            {
                "num_outputs": 1,
                "pretrained": True,
            },
        ),
        (
            "DenseNet-121",
            {
                "num_outputs": 1,
            },
        ),
        (
            "DenseNet-161",
            {
                "num_outputs": 1,
            },
        ),
        (
            "DenseNet-169",
            {
                "num_outputs": 1,
            },
        ),
        (
            "DenseNet-201",
            {
                "num_outputs": 1,
                "pretrained": True,
            },
        ),
        (
            "SEResNet-50",
            {
                "num_outputs": 1,
            },
        ),
        (
            "SEResNet-101",
            {
                "num_outputs": 1,
            },
        ),
        (
            "SEResNet-152",
            {
                "num_outputs": 1,
            },
        ),
        (
            "ViT-B/16",
            {
                "num_outputs": 1,
                "pretrained": True,
            },
        ),
        (
            "ViT-B/32",
            {
                "num_outputs": 1,
            },
        ),
        (
            "ViT-L/16",
            {
                "num_outputs": 1,
            },
        ),
        (
            "ViT-L/32",
            {
                "num_outputs": 1,
            },
        ),
    ],
)
def test_get_network(network_name, params):
    tested.append(network_name)
    _ = get_network(name=network_name, **params)
    if network_name == "ViT-L/32":  # the last one
        assert set(tested) == set(
            net.value for net in ImplementedNetworks
        )  # check we haven't miss a network


def test_update_config_with_defaults():
    config = AutoEncoderConfig(
        latent_size=1,
        in_shape=(1, 10, 10),
        conv_args={"channels": [1, 2], "dropout": 0.2},
        mlp_args={"hidden_channels": [5], "act": "relu"},
    )
    _update_config_with_defaults(config, AutoEncoder.__init__)
    assert config.in_shape == (1, 10, 10)
    assert config.latent_size == 1
    assert config.conv_args.channels == [1, 2]
    assert config.conv_args.dropout == 0.2
    assert config.conv_args.act == "prelu"
    assert config.mlp_args.hidden_channels == [5]
    assert config.mlp_args.act == "relu"
    assert config.mlp_args.norm == "batch"
    assert config.out_channels is None


def test_parameters():
    net, updated_config = get_network(
        "AutoEncoder",
        return_config=True,
        latent_size=1,
        in_shape=(1, 10, 10),
        conv_args={"channels": [1, 2], "dropout": 0.2},
        mlp_args={"hidden_channels": [5], "act": "relu"},
    )
    assert isinstance(net, AutoEncoder)
    assert net.encoder.mlp.out_channels == 1
    assert net.encoder.mlp.hidden_channels == [5]
    assert net.encoder.mlp.act == "relu"
    assert net.encoder.mlp.norm == "batch"
    assert net.in_shape == (1, 10, 10)
    assert net.encoder.convolutions.channels == (1, 2)
    assert net.encoder.convolutions.dropout == 0.2
    assert net.encoder.convolutions.act == "prelu"

    assert updated_config.in_shape == (1, 10, 10)
    assert updated_config.latent_size == 1
    assert updated_config.conv_args.channels == [1, 2]
    assert updated_config.conv_args.dropout == 0.2
    assert updated_config.conv_args.act == "prelu"
    assert updated_config.mlp_args.hidden_channels == [5]
    assert updated_config.mlp_args.act == "relu"
    assert updated_config.mlp_args.norm == "batch"
    assert updated_config.out_channels is None


def test_without_return():
    net = get_network(
        "AutoEncoder",
        return_config=False,
        latent_size=1,
        in_shape=(1, 10, 10),
        conv_args={"channels": [1, 2]},
    )
    assert isinstance(net, AutoEncoder)


def test_get_network_from_config():
    config = AutoEncoderConfig(
        latent_size=1,
        in_shape=(1, 10, 10),
        conv_args={"channels": [1, 2], "dropout": 0.2},
        mlp_args={"hidden_channels": [5], "act": "relu"},
    )
    net, updated_config = get_network_from_config(config)
    assert isinstance(net, AutoEncoder)
    assert updated_config.conv_args.act == "prelu"
    assert config.conv_args.act == "DefaultFromLibrary"
