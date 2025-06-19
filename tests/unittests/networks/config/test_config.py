from copy import deepcopy

import pytest
from pydantic import ValidationError

from clinicadl.networks.config import *
from clinicadl.networks.config import ImplementedNetwork
from clinicadl.networks.config.mlp_conv import (
    ConvDecoderOptions,
    ConvEncoderOptions,
    MLPOptions,
)
from clinicadl.networks.nn.layers.utils import ActFunction

MANDATORY_ARGS = {
    "MLP": {"num_inputs": 1, "num_outputs": 1, "hidden_dims": [1]},
    "ConvEncoder": {"spatial_dims": 2, "in_channels": 1, "channels": [1, 2]},
    "ConvDecoder": {"spatial_dims": 2, "in_channels": 1, "channels": [1, 2]},
    "CNN": {"in_shape": (1, 3, 3), "num_outputs": 1, "conv_args": {"channels": [1]}},
    "Generator": {
        "latent_size": 1,
        "start_shape": (1, 3, 3),
        "conv_args": {"channels": [1]},
    },
    "AutoEncoder": {
        "latent_size": 1,
        "in_shape": (1, 3, 3),
        "conv_args": {"channels": [1]},
    },
    "VAE": {"latent_size": 1, "in_shape": (1, 3, 3), "conv_args": {"channels": [1]}},
    "UNet": {"spatial_dims": 2, "in_channels": 1, "out_channels": 1},
    "AttentionUNet": {"spatial_dims": 2, "in_channels": 1, "out_channels": 1},
    "DenseNet": {"spatial_dims": 2, "in_channels": 1, "num_outputs": 1},
    "ResNet": {"spatial_dims": 2, "in_channels": 1, "num_outputs": 1},
    "SEResNet": {"spatial_dims": 2, "in_channels": 1, "num_outputs": 1},
    "ViT": {"in_shape": (1, 3, 3), "patch_size": 1, "num_outputs": 1},
    "DenseNet-121": {"num_outputs": None},
    "DenseNet-161": {"num_outputs": None},
    "DenseNet-169": {"num_outputs": None},
    "DenseNet-201": {"num_outputs": None},
    "ResNet-18": {"num_outputs": None},
    "ResNet-34": {"num_outputs": None},
    "ResNet-50": {"num_outputs": None},
    "ResNet-101": {"num_outputs": None},
    "ResNet-152": {"num_outputs": None},
    "SEResNet-50": {"num_outputs": None},
    "SEResNet-101": {"num_outputs": None},
    "SEResNet-152": {"num_outputs": None},
    "ViT-B/16": {"num_outputs": None},
    "ViT-B/32": {"num_outputs": None},
    "ViT-L/16": {"num_outputs": None},
    "ViT-L/32": {"num_outputs": None},
}
BAD_INPUTS = [
    ({"num_inputs": 0}, MLPConfig),
    ({"hidden_dims": [0, 1]}, MLPConfig),
    (
        {"spatial_dims": 4},
        [
            ConvEncoderConfig,
            ConvDecoderConfig,
            UNetConfig,
            AttentionUNetConfig,
            DenseNetConfig,
            ResNetConfig,
            SEResNetConfig,
        ],
    ),
    (
        {"in_channels": 0},
        [
            ConvEncoderConfig,
            ConvDecoderConfig,
            UNetConfig,
            AttentionUNetConfig,
            DenseNetConfig,
            ResNetConfig,
            SEResNetConfig,
        ],
    ),
    (
        {"channels": [0, 1]},
        [ConvEncoderConfig, ConvDecoderConfig],
    ),
    (
        {"in_shape": (1, 3, 0)},
        [CNNConfig, AutoEncoderConfig, VAEConfig, ViTConfig],
    ),
    (
        {"conv_args": {"channels": [0, 1]}},
        [CNNConfig, GeneratorConfig, AutoEncoderConfig, VAEConfig],
    ),
    (
        {"latent_size": 0},
        [GeneratorConfig, AutoEncoderConfig, VAEConfig],
    ),
    (
        {"start_shape": (1, 3, 0)},
        GeneratorConfig,
    ),
    (
        {"out_channels": 0},
        [UNetConfig, AttentionUNetConfig, AutoEncoderConfig, VAEConfig],
    ),
    ({"out_channels": None}, [UNetConfig, AttentionUNetConfig]),
    (
        {"patch_size": 0},
        ViTConfig,
    ),
    (
        {"num_outputs": 0},
        [
            MLPConfig,
            CNNConfig,
            DenseNetConfig,
            ResNetConfig,
            SEResNetConfig,
            ViTConfig,
            DenseNet121Config,
            DenseNet161Config,
            DenseNet169Config,
            DenseNet201Config,
            ResNet18Config,
            ResNet34Config,
            ResNet50Config,
            ResNet101Config,
            ResNet152Config,
            SEResNet50Config,
            SEResNet101Config,
            SEResNet152Config,
            ViTB16Config,
            ViTB32Config,
            ViTL16Config,
            ViTL32Config,
        ],
    ),
    (
        {"dropout": 1.1},
        [
            MLPConfig,
            ConvEncoderConfig,
            ConvDecoderConfig,
            DenseNetConfig,
            UNetConfig,
            AttentionUNetConfig,
            ViTConfig,
        ],
    ),
    ({"n_dense_layers": (0, 2)}, DenseNetConfig),
    ({"init_features": 0}, DenseNetConfig),
    ({"growth_rate": 0}, DenseNetConfig),
    ({"bottleneck_factor": 0}, DenseNetConfig),
    ({"block_type": "abc"}, [ResNetConfig, SEResNetConfig]),
    ({"n_res_blocks": (2, 0, 2, 2)}, [ResNetConfig, SEResNetConfig]),
    ({"n_features": (2, 0, 2, 2)}, [ResNetConfig, SEResNetConfig]),
    ({"bottleneck_reduction": 0}, [ResNetConfig, SEResNetConfig]),
    ({"se_reduction": 0}, SEResNetConfig),
    ({"embedding_dim": 0}, ViTConfig),
    ({"num_layers": 0}, ViTConfig),
    ({"num_heads": 0}, ViTConfig),
    ({"mlp_dim": 0}, ViTConfig),
    ({"pos_embed_type": "abc"}, ViTConfig),
]
GOOD_INPUTS = [
    (
        {"dropout": 0.5},
        [
            MLPConfig,
            ConvEncoderConfig,
            ConvDecoderConfig,
            DenseNetConfig,
            UNetConfig,
            AttentionUNetConfig,
            ViTConfig,
        ],
    ),
    (
        {"dropout": None},
        [
            MLPConfig,
            ConvEncoderConfig,
            ConvDecoderConfig,
            DenseNetConfig,
            UNetConfig,
            AttentionUNetConfig,
            ViTConfig,
        ],
    ),
    ({"bias": True}, [MLPConfig, ConvEncoderConfig, ConvDecoderConfig]),
    (
        {
            "n_dense_layers": (1, 2),
            "init_features": 1,
            "growth_rate": 1,
            "bottleneck_factor": 1,
        },
        DenseNetConfig,
    ),
    (
        {"block_type": "basic", "bottleneck_reduction": 1},
        [ResNetConfig, SEResNetConfig],
    ),
    ({"block_type": "bottleneck"}, [ResNetConfig, SEResNetConfig]),
    ({"se_reduction": 1}, SEResNetConfig),
    (
        {
            "pos_embed_type": None,
            "embedding_dim": 1,
            "num_layers": 1,
            "num_heads": 1,
            "mlp_dim": 1,
        },
        ViTConfig,
    ),
    ({"pos_embed_type": "sincos"}, ViTConfig),
    ({"pos_embed_type": "learnable"}, ViTConfig),
    ({"out_channels": None}, [AutoEncoderConfig, VAEConfig]),
]


@pytest.mark.parametrize("args,configs", BAD_INPUTS)
def test_bad_inputs(args, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        args_ = deepcopy(MANDATORY_ARGS[config._get_name()])
        args_.update(args)
        with pytest.raises(ValidationError):
            config(**args_)


@pytest.mark.parametrize("args,configs", GOOD_INPUTS)
def test_good_inputs(args: dict, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        args_ = deepcopy(MANDATORY_ARGS[config._get_name()])
        args_.update(args)
        c = config(**args_)
        for arg, value in args_.items():
            if arg == "conv_args":
                for sub_arg in value:
                    assert getattr(getattr(c, "conv_args"), sub_arg) == value[sub_arg]
            else:
                assert getattr(c, arg) == value


@pytest.mark.parametrize(
    "field,configs",
    [
        (
            "act",
            [
                MLPConfig,
                ConvEncoderConfig,
                ConvDecoderConfig,
            ],
        ),
        (
            "mandatory act",
            [
                DenseNetConfig,
                ResNetConfig,
                SEResNetConfig,
                UNetConfig,
                AttentionUNetConfig,
            ],
        ),
        (
            "output_act",
            [
                AutoEncoderConfig,
                VAEConfig,
                ViTConfig,
                DenseNet121Config,
                DenseNet161Config,
                DenseNet169Config,
                DenseNet201Config,
                ResNet18Config,
                ResNet34Config,
                ResNet50Config,
                ResNet101Config,
                ResNet152Config,
                SEResNet50Config,
                SEResNet101Config,
                SEResNet152Config,
                ViTB16Config,
                ViTB32Config,
                ViTL16Config,
                ViTL32Config,
            ],
        ),
    ],
)
def test_act(field, configs):
    for config in configs:
        name = config._get_name()
        args = deepcopy(MANDATORY_ARGS[name])

        for act in ActFunction:
            args["output_act"] = act.value
            c = config(**args)
            assert c.output_act == act

        c.output_act = ("elu", {"alpha": 1.0})
        assert c.output_act == ("elu", {"alpha": 1.0})

        c.output_act = None
        assert c.output_act is None

        if field == "act" or field == "mandatory act":
            for act in ActFunction:
                args["act"] = act.value
                c = config(**args)
                assert c.act == act.value

                c.act = ("elu", {"alpha": 1.0})
                assert c.act == ("elu", {"alpha": 1.0})

        if field == "act":
            c.act = None
            assert c.act is None

        # if field == "mandatory act":
        #     with pytest.raises(ValidationError):
        #         c.act = None
        # TODO : check what's the pb here ?


@pytest.mark.parametrize(
    "config",
    [
        MLPConfig,
        ConvEncoderConfig,
        ConvDecoderConfig,
    ],
)
def test_norm(config):
    name = config._get_name()
    args = deepcopy(MANDATORY_ARGS[name])

    for norm in ["batch", "syncbatch", "instance"]:
        args["norm"] = norm
        c = config(**args)
        assert c.norm == norm

    c.norm = ("group", {"num_groups": 1})
    assert c.norm == ("group", {"num_groups": 1})

    if config == MLPConfig:
        c.norm = "layer"
        assert c.norm == "layer"


@pytest.mark.parametrize(
    "config",
    [
        MLPConfig,
        ConvEncoderConfig,
        ConvDecoderConfig,
    ],
)
def test_adn_ordering(config):
    name = config._get_name()
    args = deepcopy(MANDATORY_ARGS[name])

    for adn_ordering in ["ADN", "ND", "A", ""]:
        args["adn_ordering"] = adn_ordering
        c = config(**args)
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
    name = config._get_name()
    for field in fields_to_test:
        args = deepcopy(MANDATORY_ARGS[name])

        for value in [5, (5, 5), [5, (5, 5)], [(5, 5), (5, 5)]]:
            args[field] = value
            c = config(**args)
            assert getattr(c, field) == value

        for value in [None, (5, 5, 5), [5, (5, 5, 5)], [(5, 5), (5, 5), (5, 5)]]:
            args[field] = value
            print(args)
            with pytest.raises(ValidationError):
                config(**args)


@pytest.mark.parametrize(
    "config,fields_to_test",
    [
        (ResNetConfig, ["init_conv_size", "init_conv_stride"]),
        (ViTConfig, ["patch_size"]),
    ],
)
def test_ensure_tuple(config, fields_to_test):
    name = config._get_name()
    for field in fields_to_test:
        args = deepcopy(MANDATORY_ARGS[name])

        for value in [3, (3, 3)]:
            args[field] = value
            c = config(**args)
            assert getattr(c, field) == value

        for value in [None, (3, 3, 3)]:
            args[field] = value
            with pytest.raises(ValidationError):
                config(**args)


def test_check_pooling():
    args = deepcopy(MANDATORY_ARGS["ConvEncoder"])

    for value in [(0, 1), (-1, 0, 1), None]:
        args["pooling_indices"] = value
        c = ConvEncoderConfig(**args)
        assert c.pooling_indices == value

    for value in [(0, 2), (-2, 0), 0, [0, 1, 1]]:
        args["pooling_indices"] = value
        with pytest.raises(ValidationError):
            ConvEncoderConfig(**args)

    args["pooling_indices"] = [0, 1]
    for value in [
        None,
        ("max", {"kernel_size": 2}),
        [("max", {"kernel_size": 2}), ("avg", {"kernel_size": 2})],
    ]:
        args["pooling"] = value
        c = ConvEncoderConfig(**args)
        assert c.pooling == value

    for value in [
        "max",
        ("max",),
        ("abc", {"kernel_size": 2}),
        [("max", {"kernel_size": 2})],
    ]:
        args["pooling"] = value
        with pytest.raises(ValidationError):
            ConvEncoderConfig(**args)


def test_check_unpooling():
    args = deepcopy(MANDATORY_ARGS["ConvDecoder"])

    for value in [(0, 1), (-1, 0, 1), None]:
        args["unpooling_indices"] = value
        c = ConvDecoderConfig(**args)
        assert c.unpooling_indices == value

    for value in [(0, 2), (-2, 0), 0, [0, 1, 1]]:
        args["unpooling_indices"] = value
        with pytest.raises(ValidationError):
            ConvDecoderConfig(**args)

    args["unpooling_indices"] = [0, 1]
    for value in [
        None,
        ("upsample", {"size": 2}),
        [("upsample", {"size": 2}), ("convtranspose", {"kernel_size": 2})],
    ]:
        args["unpooling"] = value
        c = ConvDecoderConfig(**args)
        assert c.unpooling == value

    for value in [
        "upsample",
        ("upsample",),
        ("abc", {"size": 2}),
        [("upsample", {"size": 2})],
    ]:
        args["unpooling"] = value
        with pytest.raises(ValidationError):
            ConvDecoderConfig(**args)


@pytest.mark.parametrize(
    "config", [CNNConfig, GeneratorConfig, AutoEncoderConfig, VAEConfig]
)
def test_mlp_args(config):
    args = deepcopy(MANDATORY_ARGS[config._get_name()])

    c = config(
        **args,
        mlp_args={
            "hidden_dims": [1, 1],
            "adn_ordering": "ADN",
        },
    )
    assert isinstance(c.mlp_args, MLPOptions)
    assert c.mlp_args.adn_ordering == "ADN"
    assert c.mlp_args.bias

    with pytest.raises(ValidationError):
        config(
            **args,
            mlp_args={
                "hidden_dims": [1, 1],
                "adn_ordering": "ADD",
            },
        )

    with pytest.raises(ValidationError):
        config(**args, mlp_args={})


@pytest.mark.parametrize("config", [CNNConfig, AutoEncoderConfig, VAEConfig])
def test_conv_args(config):
    args = deepcopy(MANDATORY_ARGS[config._get_name()])

    args["conv_args"] = {
        "channels": [1, 1],
        "pooling": [("max", {"kernel_size": 2}), ("avg", {"kernel_size": 2})],
        "pooling_indices": [0, 1],
    }
    c = config(**args)
    assert isinstance(c.conv_args, ConvEncoderOptions)
    assert c.conv_args.pooling_indices == [0, 1]
    assert c.conv_args.adn_ordering == "NDA"

    with pytest.raises(ValidationError):
        args["conv_args"] = {
            "channels": [1, 1],
            "pooling": [("max", {"kernel_size": 2})],
            "pooling_indices": [0, 1],
        }
        config(**args)

    args["conv_args"] = {}
    with pytest.raises(ValidationError):
        config(**args)


def test_unconv_args():
    args = deepcopy(MANDATORY_ARGS["Generator"])

    args["conv_args"] = {
        "channels": [1, 1],
        "unpooling": [
            ("upsample", {"scale_factor": 2}),
            ("convtranspose", {"kernel_size": 2}),
        ],
        "unpooling_indices": [0, 1],
    }
    c = GeneratorConfig(**args)
    assert isinstance(c.conv_args, ConvDecoderOptions)
    assert c.conv_args.unpooling_indices == [0, 1]
    assert c.conv_args.adn_ordering == "NDA"

    with pytest.raises(ValidationError):
        args["conv_args"] = {
            "channels": [1, 1],
            "unpooling": [("upsample", {"scale_factor": 2})],
            "unpooling_indices": [0, 1],
        }
        GeneratorConfig(**args)

    args["conv_args"] = {}
    with pytest.raises(ValidationError):
        GeneratorConfig(**args)


@pytest.mark.parametrize("config", [AutoEncoderConfig, VAEConfig])
def test_unpooling_mode(config):
    args = deepcopy(MANDATORY_ARGS[config._get_name()])

    args["unpooling_mode"] = "linear"
    with pytest.raises(ValidationError):
        config(**args)

    args["unpooling_mode"] = "nearest"
    c = config(**args)
    assert c.unpooling_mode == "nearest"


@pytest.mark.parametrize("config", [ResNetConfig, SEResNetConfig])
def test_res_blocks(config):
    args = deepcopy(MANDATORY_ARGS[config._get_name()])
    args["bottleneck_reduction"] = 2
    args["n_res_blocks"] = (3, 3)
    args["n_features"] = (4, 4)

    if config == SEResNetConfig:
        args["se_reduction"] = 5

        args["n_features"] = (4, 4)
        with pytest.raises(ValidationError):
            config(**args)

        args["se_reduction"] = 4

    c = config(**args)
    assert c.n_features == (4, 4)

    args["n_features"] = (4, 4, 4)
    with pytest.raises(ValidationError):
        config(**args)

    args["n_features"] = (4, 5)
    with pytest.raises(ValidationError):
        config(**args)

    args["n_features"] = (6, 6)
    c = config(**args)
    assert c.n_features == (6, 6)


@pytest.mark.parametrize("config", [UNetConfig, AttentionUNetConfig])
def test_unet_channels(config):
    args = deepcopy(MANDATORY_ARGS[config._get_name()])
    args["channels"] = [4]

    with pytest.raises(ValidationError):
        config(**args)


def test_vit_checks():
    args = deepcopy(MANDATORY_ARGS["ViT"])

    args["patch_size"] = 4
    with pytest.raises(ValidationError):
        ViTConfig(**args)

    args["embedding_dim"] = 4
    args["num_heads"] = 3
    with pytest.raises(ValidationError):
        ViTConfig(**args)

    args["patch_size"] = 3
    args["num_heads"] = 2
    c = ViTConfig(**args)
    assert c.patch_size == 3
    assert c.num_heads == 2


@pytest.mark.parametrize(
    "config",
    [
        DenseNet121Config,
        DenseNet161Config,
        DenseNet169Config,
        DenseNet201Config,
        ResNet18Config,
        ResNet34Config,
        ResNet50Config,
        ResNet101Config,
        ResNet152Config,
        SEResNet50Config,
        SEResNet101Config,
        SEResNet152Config,
        ViTB16Config,
        ViTB32Config,
        ViTL16Config,
        ViTL32Config,
    ],
)
def test_pretrained(config):
    name = config._get_name()
    args = deepcopy(MANDATORY_ARGS[name])

    # args["pretrained"] = None
    # with pytest.raises(ValidationError):
    #     config(**args)

    args["pretrained"] = False
    c = config(**args)
    assert not c.pretrained and c.pretrained is not None

    args["pretrained"] = True
    if "SEResNet" in name:
        with pytest.raises(ValidationError):
            config(**args)
    else:
        c = config(**args)
        assert c.pretrained


@pytest.mark.parametrize(
    "config,network",
    [
        (MLPConfig, nets.MLP),
        (ConvEncoderConfig, nets.ConvEncoder),
        (ConvDecoderConfig, nets.ConvDecoder),
        (CNNConfig, nets.CNN),
        (GeneratorConfig, nets.Generator),
        (AutoEncoderConfig, nets.AutoEncoder),
        (VAEConfig, nets.VAE),
        (DenseNetConfig, nets.DenseNet),
        (ResNetConfig, nets.ResNet),
        (SEResNetConfig, nets.SEResNet),
        (UNetConfig, nets.UNet),
        (AttentionUNetConfig, nets.AttentionUNet),
        (ViTConfig, nets.ViT),
        (DenseNet121Config, nets.DenseNet),
        (DenseNet161Config, nets.DenseNet),
        (DenseNet169Config, nets.DenseNet),
        (DenseNet201Config, nets.DenseNet),
        (ResNet18Config, nets.ResNet),
        (ResNet34Config, nets.ResNet),
        (ResNet50Config, nets.ResNet),
        (ResNet101Config, nets.ResNet),
        (ResNet152Config, nets.ResNet),
        (SEResNet50Config, nets.SEResNet),
        (SEResNet101Config, nets.SEResNet),
        (SEResNet152Config, nets.SEResNet),
        (ViTB16Config, nets.ViT),
        (ViTB32Config, nets.ViT),
        (ViTL16Config, nets.ViT),
        (ViTL32Config, nets.ViT),
    ],
)
def test_get_object(config, network):
    name = config._get_name()
    args = deepcopy(MANDATORY_ARGS[name])
    c = config(**args)
    network_from_config = c.get_object()
    assert isinstance(network_from_config, network)

    if name == "AutoEncoder":
        config = AutoEncoderConfig(
            latent_size=1,
            in_shape=(1, 10, 10),
            conv_args={"channels": [1, 2], "dropout": 0.2},
            mlp_args={"hidden_dims": [5], "act": "relu"},
        )
        net = config.get_object()
        assert isinstance(net, nets.AutoEncoder)
        assert net.encoder.mlp.out_channels == 1
        assert net.encoder.mlp.hidden_dims == [5]
        assert net.encoder.mlp.act == "relu"
        assert net.encoder.mlp.norm == "batch"
        assert net.in_shape == (1, 10, 10)
        assert net.encoder.convolutions.channels == [1, 2]
        assert net.encoder.convolutions.dropout == 0.2
        assert net.encoder.convolutions.act == "prelu"
