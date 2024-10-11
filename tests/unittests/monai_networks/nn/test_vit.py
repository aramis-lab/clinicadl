import numpy as np
import pytest
import torch

from clinicadl.monai_networks.nn import ViT, get_vit
from clinicadl.monai_networks.nn.layers import ActFunction
from clinicadl.monai_networks.nn.vit import CommonViT

INPUT_1D = torch.randn(2, 1, 16)
INPUT_2D = torch.randn(2, 2, 15, 16)
INPUT_3D = torch.randn(2, 3, 24, 24, 24)


@pytest.mark.parametrize(
    "input_tensor,patch_size,num_outputs,embedding_dim,num_layers,num_heads,mlp_dim,pos_embed_type,output_act,dropout,error",
    [
        (INPUT_1D, 4, 1, 25, 3, 5, 26, None, "softmax", None, False),
        (
            INPUT_1D,
            5,
            1,
            25,
            3,
            5,
            26,
            None,
            "softmax",
            None,
            True,
        ),  # img not divisible by patch
        (
            INPUT_1D,
            4,
            1,
            25,
            3,
            4,
            26,
            None,
            "softmax",
            None,
            True,
        ),  # embedding not divisible by num heads
        (INPUT_1D, 4, 1, 24, 5, 4, 26, "sincos", "softmax", None, True),  # sincos
        (INPUT_2D, (3, 4), None, 24, 2, 4, 42, "learnable", "tanh", 0.1, False),
        (
            INPUT_2D,
            4,
            None,
            24,
            2,
            6,
            42,
            "learnable",
            "tanh",
            0.1,
            True,
        ),  # img not divisible by patch
        (
            INPUT_2D,
            (3, 4),
            None,
            24,
            2,
            5,
            42,
            "learnable",
            "tanh",
            0.1,
            True,
        ),  # embedding not divisible by num heads
        (
            INPUT_2D,
            (3, 4),
            None,
            18,
            2,
            6,
            42,
            "sincos",
            "tanh",
            0.1,
            True,
        ),  # sincos : embedding not divisible by 4
        (INPUT_2D, (3, 4), None, 24, 2, 6, 42, "sincos", "tanh", 0.1, False),
        (
            INPUT_3D,
            6,
            2,
            15,
            2,
            3,
            42,
            "sincos",
            None,
            0.0,
            True,
        ),  # sincos : embedding not divisible by 6
        (INPUT_3D, 6, 2, 18, 2, 3, 42, "sincos", None, 0.0, False),
    ],
)
def test_vit(
    input_tensor,
    patch_size,
    num_outputs,
    embedding_dim,
    num_layers,
    num_heads,
    mlp_dim,
    pos_embed_type,
    output_act,
    dropout,
    error,
):
    batch_size = input_tensor.shape[0]
    img_size = input_tensor.shape[2:]
    spatial_dims = len(img_size)
    if error:
        with pytest.raises(ValueError):
            ViT(
                in_shape=input_tensor.shape[1:],
                patch_size=patch_size,
                num_outputs=num_outputs,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                pos_embed_type=pos_embed_type,
                output_act=output_act,
                dropout=dropout,
            )
    else:
        net = ViT(
            in_shape=input_tensor.shape[1:],
            patch_size=patch_size,
            num_outputs=num_outputs,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            pos_embed_type=pos_embed_type,
            output_act=output_act,
            dropout=dropout,
        )
        output = net(input_tensor)

        if num_outputs:
            assert output.shape == (batch_size, num_outputs)
        else:
            n_patches = int(
                np.prod(
                    np.array(img_size)
                    // np.array(
                        patch_size
                        if isinstance(patch_size, tuple)
                        else (patch_size,) * spatial_dims
                    )
                )
            )
            assert output.shape == (batch_size, n_patches, embedding_dim)

        if output_act and num_outputs:
            assert net.fc.output_act is not None
        elif output_act and num_outputs is None:
            with pytest.raises(AttributeError):
                net.fc.output_act

        assert net.conv_proj.out_channels == embedding_dim
        encoder = net.encoder.layers
        for transformer_block in encoder:
            assert isinstance(transformer_block.norm1, torch.nn.LayerNorm)
            assert isinstance(transformer_block.norm2, torch.nn.LayerNorm)
            assert transformer_block.self_attention.num_heads == num_heads
            assert transformer_block.self_attention.dropout == (
                dropout if dropout is not None else 0.0
            )
            assert transformer_block.self_attention.embed_dim == embedding_dim
            assert transformer_block.mlp[0].out_features == mlp_dim
            assert transformer_block.mlp[2].p == (
                dropout if dropout is not None else 0.0
            )
            assert transformer_block.mlp[4].p == (
                dropout if dropout is not None else 0.0
            )
        assert net.encoder.dropout.p == (dropout if dropout is not None else 0.0)
        assert isinstance(net.encoder.norm, torch.nn.LayerNorm)

        pos_embedding = net.encoder.pos_embedding
        if pos_embed_type is None:
            assert not pos_embedding.requires_grad
            assert (pos_embedding == torch.zeros_like(pos_embedding)).all()
        elif pos_embed_type == "sincos":
            assert not pos_embedding.requires_grad
            if num_outputs:
                assert (
                    pos_embedding[0, 1, 0] == 0.0
                )  # first element of of sincos embedding of first patch is zero
            else:
                assert pos_embedding[0, 0, 0] == 0.0
        else:
            assert pos_embedding.requires_grad
            if num_outputs:
                assert pos_embedding[0, 1, 0] != 0.0
            else:
                assert pos_embedding[0, 0, 0] != 0.0

        with pytest.raises(IndexError):
            encoder[num_layers]


@pytest.mark.parametrize("act", [act for act in ActFunction])
def test_activations(act):
    batch_size = INPUT_2D.shape[0]
    net = ViT(
        in_shape=INPUT_2D.shape[1:],
        patch_size=(3, 4),
        num_outputs=1,
        embedding_dim=12,
        num_layers=2,
        num_heads=3,
        mlp_dim=24,
        output_act=act,
    )
    assert net(INPUT_2D).shape == (batch_size, 1)


def test_activation_parameters():
    output_act = ("ELU", {"alpha": 0.2})
    net = ViT(
        in_shape=(1, 12, 12),
        patch_size=3,
        num_outputs=1,
        embedding_dim=12,
        num_layers=2,
        num_heads=3,
        mlp_dim=24,
        output_act=output_act,
    )
    assert isinstance(net.fc.output_act, torch.nn.ELU)
    assert net.fc.output_act.alpha == 0.2


@pytest.mark.parametrize(
    "name,num_outputs,output_act,img_size",
    [
        (CommonViT.B_16, 1, "sigmoid", (224, 224)),
        (CommonViT.B_32, 2, None, (224, 224)),
        (CommonViT.L_16, None, "sigmoid", (224, 224)),
        (CommonViT.L_32, None, None, (224, 224)),
    ],
)
def test_get_vit(name, num_outputs, output_act, img_size):
    input_tensor = torch.randn(1, 3, *img_size)

    vit = get_vit(name, num_outputs=num_outputs, output_act=output_act, pretrained=True)
    if num_outputs:
        assert vit.fc.out.out_features == num_outputs
    else:
        assert vit.fc is None

    if output_act and num_outputs:
        assert vit.fc.output_act is not None
    elif output_act and num_outputs is None:
        assert vit.fc is None

    vit(input_tensor)


def test_get_vit_output():
    from torchvision.models import vit_b_16

    gt = vit_b_16(weights="DEFAULT")
    gt.heads = torch.nn.Identity()
    x = torch.randn(1, 3, 224, 224)

    vit = get_vit(CommonViT.B_16, num_outputs=1, pretrained=True)
    vit.fc = torch.nn.Identity()
    with torch.no_grad():
        assert (vit(x) == gt(x)).all()
