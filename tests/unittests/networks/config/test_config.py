from copy import deepcopy

import pytest
from pydantic import ValidationError

from clinicadl.networks.config import ImplementedNetwork, create_network_config
from clinicadl.networks.config.cnns import (
    AutoEncoderConfig,
    CNNConfig,
    GeneratorConfig,
    VAEConfig,
)
from clinicadl.networks.config.mlp_conv import (
    ConvDecoderConfig,
    ConvDecoderOptions,
    ConvEncoderConfig,
    ConvEncoderOptions,
    MLPConfig,
    MLPOptions,
)
from clinicadl.networks.config.resnet import ResNetConfig
from clinicadl.networks.config.senet import SEResNetConfig
from clinicadl.networks.config.unet import AttentionUNetConfig, UNetConfig
from clinicadl.networks.config.vit import ViTConfig
from clinicadl.networks.nn.layers.utils import ActFunction

BAD_INPUTS = {
    "num_inputs": 0,
    "num_outputs": 0,
    "hidden_dims": [0, 2],
    "dropout": 1.1,
    "bias": None,
    "spatial_dims": 0,
    "in_channels": 0,
    "channels": [0, 1],
    "in_shape": 6,
    "start_shape": 5,
    "latent_size": 0,
    "out_channels": 0,
    "n_dense_layers": (0, 2),
    "init_features": 0,
    "growth_rate": 0,
    "bottleneck_factor": 0,
    "block_type": "abc",
    "bottleneck_reduction": 0,
    "se_reduction": 0,
    "n_res_blocks": (2, 0, 2, 2),
    "n_features": (2, 0, 2, 2),
    "patch_size": 0,
    "embedding_dim": 0,
    "num_layers": 0,
    "num_heads": 0,
    "mlp_dim": 0,
    "pos_embed_type": "abc",
}
GOOD_INPUTS_1 = {
    "num_inputs": 1,
    "num_outputs": 1,
    "hidden_dims": [1, 2],
    "dropout": 0.5,
    "bias": True,
    "spatial_dims": 1,
    "in_channels": 1,
    "channels": [1, 1],
    "in_shape": (1, 6, 6),
    "start_shape": (5, 5),
    "latent_size": 1,
    "out_channels": 1,
    "n_dense_layers": (1, 2),
    "init_features": 1,
    "growth_rate": 1,
    "bottleneck_factor": 1,
    "block_type": "basic",
    "bottleneck_reduction": 1,
    "se_reduction": 1,
    "pos_embed_type": None,
    "patch_size": 3,
    "embedding_dim": 1,
    "num_layers": 1,
    "num_heads": 1,
    "mlp_dim": 1,
    "conv_args": {"channels": [1, 1]},
}
GOOD_INPUTS_2 = {
    "dropout": None,
    "bias": False,
    "in_shape": (5,),
    "start_shape": (5,),
    "out_channels": None,
    "block_type": "bottleneck",
    "pos_embed_type": "sincos",
}
GOOD_INPUTS_3 = {"pos_embed_type": "learnable"}


def test_validation_fail():
    configs = [create_network_config(network) for network in ImplementedNetwork] + [
        MLPOptions,
        ConvEncoderOptions,
        ConvDecoderOptions,
    ]
    for config in configs:
        fields = config.model_fields
        inputs = {key: value for key, value in BAD_INPUTS.items() if key in fields}

        for input, value in inputs.items():
            mandatory_inputs = deepcopy(GOOD_INPUTS_1)
            if input in mandatory_inputs:
                del mandatory_inputs[input]
            with pytest.raises(ValidationError):
                config(**{input: value}, **mandatory_inputs)


@pytest.mark.parametrize(
    "good_inputs",
    [
        GOOD_INPUTS_1,
        GOOD_INPUTS_2,
    ],
)
def test_validation_pass(good_inputs):
    configs = [create_network_config(network) for network in ImplementedNetwork] + [
        MLPOptions,
        ConvEncoderOptions,
        ConvDecoderOptions,
    ]
    for config in configs:
        fields = config.model_fields
        inputs = {key: value for key, value in good_inputs.items() if key in fields}

        if config.__name__.replace("Config", "") in ["UNet", "AttentionUNet"]:
            inputs["out_channels"] = 1

        mandatory_inputs = deepcopy(GOOD_INPUTS_1)
        for input in inputs:
            if input in mandatory_inputs:
                del mandatory_inputs[input]

        c = config(**inputs, **mandatory_inputs)
        for arg, value in inputs.items():
            if arg == "conv_args" and c.name in ["CNN", "AutoEncoder", "VAE"]:
                assert getattr(c, arg) == ConvEncoderOptions(**value)
            elif arg == "conv_args" and c.name == "Generator":
                assert getattr(c, arg) == ConvDecoderOptions(**value)
            else:
                assert getattr(c, arg) == value


def test_act():
    configs = [create_network_config(network) for network in ImplementedNetwork] + [
        MLPOptions,
        ConvEncoderOptions,
        ConvDecoderOptions,
    ]
    for config in configs:
        fields = config.model_fields
        if "act" in fields:
            inputs = {
                key: value for key, value in GOOD_INPUTS_1.items() if key in fields
            }

            for act in ActFunction:
                inputs["act"] = act.value
                c = config(**inputs)
                assert c.act == act.value

            c.act = ("elu", {"alpha": 1.0})
            assert c.act == ("elu", {"alpha": 1.0})

            if (
                c.name in ["MLP", "ConvEncoder", "ConvDecoder"] or c.name is None
            ):  # None for MLPOptions, ConvEncoderOptions and ConvDecoderOptions
                c.act = None
                assert c.act is None
            else:
                with pytest.raises(ValidationError):
                    c.act = None

        if "output_act" in fields:
            inputs = {
                key: value for key, value in GOOD_INPUTS_1.items() if key in fields
            }

            for act in ActFunction:
                inputs["output_act"] = act.value
                c = config(**inputs)
                assert c.output_act == act

            c.output_act = ("elu", {"alpha": 1.0})
            assert c.output_act == ("elu", {"alpha": 1.0})

            c.output_act = None
            assert c.output_act is None


@pytest.mark.parametrize("config", [MLPConfig, ConvEncoderConfig, ConvDecoderConfig])
def test_norm(config):
    fields = config.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}

    for norm in ["batch", "syncbatch", "instance"]:
        inputs["norm"] = norm
        c = config(**inputs)
        assert c.norm == norm

    c.norm = ("group", {"num_groups": 1})
    assert c.norm == ("group", {"num_groups": 1})

    if config == MLPConfig:
        c.norm = "layer"
        assert c.norm == "layer"


@pytest.mark.parametrize("config", [MLPConfig, ConvEncoderConfig, ConvDecoderConfig])
def test_adn_ordering(config):
    fields = config.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}

    for adn_ordering in ["ADN", "ND", "A", ""]:
        inputs["adn_ordering"] = adn_ordering
        c = config(**inputs)
        assert c.adn_ordering == adn_ordering

    for adn_ordering in ["AAD", "ADM"]:
        with pytest.raises(ValidationError):
            c.adn_ordering = adn_ordering


@pytest.mark.parametrize(
    "config,fields_to_test",
    [
        (ConvEncoderConfig, ["kernel_size", "stride", "padding", "dilation"]),
        (
            ConvDecoderConfig,
            ["kernel_size", "stride", "padding", "dilation", "output_padding"],
        ),
    ],
)
def test_ensure_list_of_tuples(config, fields_to_test):
    fields = config.model_fields
    for field in fields_to_test:
        inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}
        inputs["spatial_dims"] = 2
        inputs["channels"] = [1, 1]

        for value in [5, (5, 5), [5, (5, 5)], [(5, 5), (5, 5)]]:
            inputs[field] = value
            c = config(**inputs)
            assert getattr(c, field) == value

        for value in [None, (5, 5, 5), [5, (5, 5, 5)], [(5, 5), (5, 5), (5, 5)]]:
            inputs[field] = value
            with pytest.raises(ValidationError):
                config(**inputs)


@pytest.mark.parametrize(
    "config,fields_to_test",
    [
        (ResNetConfig, ["init_conv_size", "init_conv_stride"]),
        (ViTConfig, ["patch_size"]),
    ],
)
def test_ensure_tuple(config, fields_to_test):
    fields = config.model_fields
    for field in fields_to_test:
        inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}
        inputs["spatial_dims"] = 2

        for value in [3, (3, 3)]:
            inputs[field] = value
            c = config(**inputs)
            assert getattr(c, field) == value

        for value in [None, (3, 3, 3)]:
            inputs[field] = value
            with pytest.raises(ValidationError):
                config(**inputs)


def test_check_pooling():
    fields = ConvEncoderConfig.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}
    inputs["channels"] == [1, 1]

    for value in [(0, 1), (-1, 0, 1), None]:
        inputs["pooling_indices"] = value
        c = ConvEncoderConfig(**inputs)
        assert c.pooling_indices == value

    for value in [(0, 2), (-2, 0), 0, [0, 1, 1]]:
        inputs["pooling_indices"] = value
        with pytest.raises(ValidationError):
            ConvEncoderConfig(**inputs)

    inputs["pooling_indices"] = [0, 1]
    for value in [
        None,
        ("max", {"kernel_size": 2}),
        [("max", {"kernel_size": 2}), ("avg", {"kernel_size": 2})],
    ]:
        inputs["pooling"] = value
        c = ConvEncoderConfig(**inputs)
        assert c.pooling == value

    for value in [
        "max",
        ("max",),
        ("abc", {"kernel_size": 2}),
        [("max", {"kernel_size": 2})],
    ]:
        inputs["pooling"] = value
        with pytest.raises(ValidationError):
            ConvEncoderConfig(**inputs)


def test_check_unpooling():
    fields = ConvDecoderConfig.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}
    inputs["channels"] == [1, 1]

    for value in [(0, 1), (-1, 0, 1), None]:
        inputs["unpooling_indices"] = value
        c = ConvDecoderConfig(**inputs)
        assert c.unpooling_indices == value

    for value in [(0, 2), (-2, 0), 0, [0, 1, 1]]:
        inputs["unpooling_indices"] = value
        with pytest.raises(ValidationError):
            ConvDecoderConfig(**inputs)

    inputs["unpooling_indices"] = [0, 1]
    for value in [
        None,
        ("upsample", {"size": 2}),
        [("upsample", {"size": 2}), ("convtranspose", {"kernel_size": 2})],
    ]:
        inputs["unpooling"] = value
        c = ConvDecoderConfig(**inputs)
        assert c.unpooling == value

    for value in [
        "upsample",
        ("upsample",),
        ("abc", {"size": 2}),
        [("upsample", {"size": 2})],
    ]:
        inputs["unpooling"] = value
        with pytest.raises(ValidationError):
            ConvDecoderConfig(**inputs)


@pytest.mark.parametrize(
    "config", [CNNConfig, GeneratorConfig, AutoEncoderConfig, VAEConfig]
)
def test_mlp_args(config):
    fields = config.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}

    c = config(
        **inputs,
        mlp_args={
            "hidden_dims": [1, 1],
            "adn_ordering": "ADN",
        },
    )
    assert isinstance(c.mlp_args, MLPOptions)
    assert c.mlp_args.adn_ordering == "ADN"

    with pytest.raises(ValidationError):
        config(
            **inputs,
            mlp_args={
                "hidden_dims": [1, 1],
                "adn_ordering": "ADD",
            },
        )

    with pytest.raises(ValidationError):
        config(**inputs, mlp_args={})


@pytest.mark.parametrize("config", [CNNConfig, AutoEncoderConfig, VAEConfig])
def test_conv_args(config):
    fields = config.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}

    inputs["conv_args"] = {
        "channels": [1, 1],
        "pooling": [("max", {"kernel_size": 2}), ("avg", {"kernel_size": 2})],
        "pooling_indices": [0, 1],
    }
    c = config(**inputs)
    assert isinstance(c.conv_args, ConvEncoderOptions)
    assert c.conv_args.pooling_indices == [0, 1]

    with pytest.raises(ValidationError):
        inputs["conv_args"] = {
            "channels": [1, 1],
            "pooling": [("max", {"kernel_size": 2})],
            "pooling_indices": [0, 1],
        }
        config(**inputs)

    inputs["conv_args"] = {}
    with pytest.raises(ValidationError):
        config(**inputs)


def test_unconv_args():
    fields = GeneratorConfig.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}

    inputs["conv_args"] = {
        "channels": [1, 1],
        "unpooling": [
            ("upsample", {"scale_factor": 2}),
            ("convtranspose", {"kernel_size": 2}),
        ],
        "unpooling_indices": [0, 1],
    }
    c = GeneratorConfig(**inputs)
    assert isinstance(c.conv_args, ConvDecoderOptions)
    assert c.conv_args.unpooling_indices == [0, 1]

    with pytest.raises(ValidationError):
        inputs["conv_args"] = {
            "channels": [1, 1],
            "unpooling": [("upsample", {"scale_factor": 2})],
            "unpooling_indices": [0, 1],
        }
        GeneratorConfig(**inputs)

    inputs["conv_args"] = {}
    with pytest.raises(ValidationError):
        GeneratorConfig(**inputs)


@pytest.mark.parametrize("config", [ResNetConfig, SEResNetConfig])
def test_res_blocks(config):
    fields = ResNetConfig.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}
    inputs["bottleneck_reduction"] = 2
    inputs["n_res_blocks"] = (3, 3)

    inputs["n_features"] = (4, 4)
    c = config(**inputs)
    assert c.n_features == (4, 4)

    inputs["n_features"] = (4, 4, 4)
    with pytest.raises(ValidationError):
        config(**inputs)

    inputs["n_features"] = (4, 5)
    with pytest.raises(ValidationError):
        config(**inputs)

    if "se_reduction" in fields:
        inputs["se_reduction"] = 5

        inputs["n_features"] = (4, 4)
        with pytest.raises(ValidationError):
            config(**inputs)

    inputs["n_features"] = (6, 6)
    c = config(**inputs)
    assert c.n_features == (6, 6)


@pytest.mark.parametrize("config", [UNetConfig, AttentionUNetConfig])
def test_unet_channels(config):
    fields = config.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}
    inputs["channels"] = [4]

    with pytest.raises(ValidationError):
        config(**inputs)


def test_vit_checks():
    fields = ViTConfig.model_fields
    inputs = {key: value for key, value in GOOD_INPUTS_1.items() if key in fields}
    inputs["in_shape"] = (1, 6, 6)

    inputs["patch_size"] = 4
    with pytest.raises(ValidationError):
        ViTConfig(**inputs)

    inputs["embedding_dim"] = 4
    inputs["num_heads"] = 3
    with pytest.raises(ValidationError):
        ViTConfig(**inputs)

    inputs["patch_size"] = 3
    inputs["num_heads"] = 2
    c = ViTConfig(**inputs)
    assert c.patch_size == 3
    assert c.num_heads == 2


def test_pretrained():
    for network in ImplementedNetwork:
        config = create_network_config(network)
        fields = ViTConfig.model_fields
        if "pretrained" in fields:
            inputs = {
                key: value for key, value in GOOD_INPUTS_1.items() if key in fields
            }

            inputs["pretrained"] = None
            with pytest.raises(ValidationError):
                config(**inputs)

            inputs["pretrained"] = False
            c = config(**inputs)
            assert c.pretrained

            inputs["pretrained"] = True
            if "SENet" in config.name:
                with pytest.raises(ValidationError):
                    config(**inputs)
            else:
                c = config(**inputs)
                assert c.pretrained
