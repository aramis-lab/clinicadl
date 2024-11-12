import pytest
from pydantic import ValidationError

from clinicadl.networks import (
    ImplementedNetwork,
    get_network_config,
    get_network_from_config,
)
from clinicadl.networks.config import ImplementedNetwork, create_network_config
from clinicadl.networks.config.cnns import AutoEncoderConfig
from clinicadl.networks.config.mlp_conv import ConvEncoderOptions, MLPOptions
from clinicadl.networks.factory import _update_config_with_defaults
from clinicadl.networks.nn import AutoEncoder

MANDATORY_ARGS = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "in_shape": (1, 6, 6),
    "latent_size": 1,
    "conv_args": {"channels": [1]},
    "num_outputs": 1,
    "channels": [1, 1],
    "start_shape": (1, 4, 4),
    "num_inputs": 1,
    "hidden_dims": [1],
    "patch_size": 3,
}


def test_get_network_from_config():
    # test all networks
    for network in ImplementedNetwork:
        config = create_network_config(network=network)(**MANDATORY_ARGS)
        _ = get_network_from_config(config=config)

    # test arguments
    config = create_network_config("AutoEncoder")(
        latent_size=1,
        in_shape=(1, 10, 10),
        conv_args={"channels": [1, 2], "dropout": 0.2},
        mlp_args={"hidden_dims": [5], "act": "relu"},
    )
    net, updated_config = get_network_from_config(config=config)
    assert isinstance(net, AutoEncoder)
    assert net.encoder.mlp.out_channels == 1
    assert net.encoder.mlp.hidden_dims == [5]
    assert net.encoder.mlp.act == "relu"
    assert net.encoder.mlp.norm == "batch"
    assert net.in_shape == (1, 10, 10)
    assert net.encoder.convolutions.channels == [1, 2]
    assert net.encoder.convolutions.dropout == 0.2
    assert net.encoder.convolutions.act == "prelu"

    assert updated_config.in_shape == (1, 10, 10)
    assert updated_config.latent_size == 1
    assert updated_config.conv_args.channels == [1, 2]
    assert updated_config.conv_args.dropout == 0.2
    assert updated_config.conv_args.act == "prelu"
    assert updated_config.mlp_args.hidden_dims == [5]
    assert updated_config.mlp_args.act == "relu"
    assert updated_config.mlp_args.norm == "batch"
    assert updated_config.out_channels is None

    # test that checks are performed when getting defaults
    config = create_network_config("ViT")(
        in_shape=(1, 16, 16),
        patch_size=4,
        num_outputs=1,
        embedding_dim=767,  # not divisible by 12 (the default value for num_heads)
    )
    with pytest.raises(ValidationError):
        get_network_from_config(config=config)

    config.embedding_dim = 768
    _ = get_network_from_config(config)


def test_get_network_config():
    config = get_network_config(
        "AutoEncoder",
        latent_size=1,
        in_shape=(1, 10, 10),
        conv_args={"channels": [1, 2], "dropout": 0.2},
        mlp_args={"hidden_dims": [5], "act": "relu"},
    )
    assert config.name == "AutoEncoder"
    assert config.conv_args.channels == [1, 2]
    assert config.conv_args.dropout == 0.2
    assert config.conv_args.pooling == ("max", {"kernel_size": 2})
    assert config.mlp_args.hidden_dims == [5]
    assert config.mlp_args.act == "relu"
    assert config.mlp_args.norm == "batch"
    assert config.unpooling_mode == "nearest"

    with pytest.raises(ValueError):
        get_network_config("abc", **MANDATORY_ARGS)

    # test that checks are performed when getting defaults
    with pytest.raises(ValidationError):
        get_network_config(
            "ViT",
            in_shape=(1, 16, 16),
            patch_size=4,
            num_outputs=1,
            embedding_dim=767,  # not divisible by 12 (the default value for num_heads)
        )

    get_network_config(
        "ViT",
        in_shape=(1, 16, 16),
        patch_size=4,
        num_outputs=1,
        embedding_dim=768,
    )
