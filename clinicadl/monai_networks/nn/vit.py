import re
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets import ViT as BaseViT
from torch.hub import load_state_dict_from_url
from torchvision.models.vision_transformer import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)

from .layers import ActFunction
from .layers.vit import TransformerBlock
from .utils import ActivationParameters


class PosEmbedType(str, Enum):
    """Available position embedding types for ViT."""

    LEARN = "learnable"
    SINCOS = "sincos"


class ViT(BaseViT):
    """
    Vision Transformer based on the [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale]
    (https://arxiv.org/pdf/2010.11929) paper.
    Adapted from [MONAI's implementation](https://docs.monai.io/en/stable/networks.html#vit).

    The user can customize the patch size, the embedding dimension, the number of transformer blocks, the number of
    attention heads, as well as other parameters like the type of positional embedding.

    Note: the network returns the output of the network as well as patch embeddings after each transformer block.

    Parameters
    ----------
    in_shape : Sequence[int]
        sequence of integers stating the dimension of the input tensor (minus batch dimension).
    patch_size : Union[Sequence[int], int]
        sequence of integers stating the patch size (minus batch and channel dimensions).
    num_outputs : Optional[int]
        number of output variables after the last linear layer.\n
        If None, the features before the last fully connected layer will be returned.
    embedding_dim : int (optional, default=768)
        size of the embedding vectors. Must be divisible by `num_heads` as each head will be responsible for
        a part of the embedding vectors. Default to 768, as for 'ViT-Base' in the original paper.
    num_layers : int (optional, default=12)
        number of consecutive transformer blocks. Default to 12, as for 'ViT-Base' in the original paper.
    num_heads : int (optional, default=12)
        number of heads in the self-attention block. Must divide `embedding_size`.
        Default to 12, as for 'ViT-Base' in the original paper.
    mlp_dim : int (optional, default=3072)
        size of the hidden layer in the MLP part of the transformer block. Default to 3072, as for 'ViT-Base'
        in the original paper.
    pos_embed_type : Optional[Union[str, PosEmbedType]] (optional, default="learnable")
        type of positional embedding. Can be either `"learnable"`, `"sincos"` or `None`.\n
        - `learnable`: the positional embeddings are parameters that will be learned during the training
        process.
        - `sincos`: the positional embeddings are fixed and determined with sinus and cosinus formulas (see Dosovitskiy et al.,
        'Attention Is All You Need, https://arxiv.org/pdf/1706.03762).
        - `None`: no positional embeddings are used.\n
        Default to `"learnable"`, as in the original paper.
    output_act : ActivationParameters (optional, default=ActFunction.TANH)
        if `num_outputs` is not None, a potential activation layer applied to the outputs of the network,
        and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.\n
    dropout : Optional[float] (optional, default=None)
        dropout ratio. If None, no dropout.

    Examples
    --------
    >>> ViT(
            in_shape=(3, 60, 64),
            patch_size=4,
            num_outputs=2,
            embedding_dim=32,
            num_layers=2,
            num_heads=4,
            mlp_dim=128,
            output_act="softmax",
        )
    ViT(
        (patch_embedding): PatchEmbeddingBlock(
            (patch_embeddings): Conv2d(3, 32, kernel_size=(4, 4), stride=(4, 4))
            (dropout): Dropout(p=0.0, inplace=False)
        )
        (blocks): ModuleList(
            (0-1): 2 x TransformerBlock(
                (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (attn): SABlock(
                    (qkv): Linear(in_features=32, out_features=96, bias=True)
                    (input_rearrange): Rearrange('b h (qkv l d) -> qkv b l h d', qkv=3, l=4)
                    (drop_weights): Dropout(p=0.0, inplace=False)
                    (out_rearrange): Rearrange('b h l d -> b l (h d)')
                    (drop_output): Dropout(p=0.0, inplace=False)
                )
                (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (mlp): MLPBlock(
                    (linear1): Linear(in_features=32, out_features=128, bias=True)
                    (fn): GELU(approximate='none')
                    (drop1): Dropout(p=0.0, inplace=False)
                    (linear2): Linear(in_features=128, out_features=32, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                )
            )
        )
        (norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
        (classification_head): Sequential(
            (linear): Linear(in_features=32, out_features=2, bias=True)
            (output_act): Softmax(dim=None)
        )
    )
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        patch_size: Union[Sequence[int], int],
        num_outputs: Optional[int],
        embedding_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        pos_embed_type: Optional[Union[str, PosEmbedType]] = PosEmbedType.LEARN,
        output_act: ActivationParameters = ActFunction.TANH,
        dropout: Optional[float] = None,
    ) -> None:
        super(BaseViT, self).__init__()
        pos_embed_type = self._check_pos_embedding(pos_embed_type)
        self._check_embedding_dim(embedding_dim, num_heads)
        in_channels, *img_size = in_shape
        spatial_dims = len(img_size)
        dropout_rate = dropout if dropout else 0.0

        self.classification = True if num_outputs else False
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=embedding_dim,
            num_heads=num_heads,
            proj_type="conv",
            pos_embed_type=pos_embed_type,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=embedding_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=True,
                    save_attn=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            self.classification_head = nn.Sequential(
                OrderedDict([("out", nn.Linear(embedding_dim, num_outputs))])
            )
            self.classification_head.output_act = (
                get_act_layer(output_act) if output_act else None
            )

    @classmethod
    def _check_pos_embedding(
        cls, pos_embed_type: Optional[Union[str, PosEmbedType]]
    ) -> Union[str, PosEmbedType]:
        """
        Checks positional embedding and converts None to a string.
        """
        if pos_embed_type:
            pos_embed_type = PosEmbedType(pos_embed_type)
        else:
            pos_embed_type = "none"
        return pos_embed_type

    @classmethod
    def _check_embedding_dim(cls, embedding_dim: int, num_heads: int) -> None:
        """
        Checks consistency between embedding dimension and number of heads.
        """
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim should be divisible by num_heads. Got embedding_dim={embedding_dim} "
                f" and num_heads={num_heads}"
            )


class CommonViT(str, Enum):
    """Supported ViT networks."""

    B_16 = "ViT-B/16"
    B_32 = "ViT-B/32"
    L_16 = "ViT-L/16"
    L_32 = "ViT-L/32"


def get_vit(
    model: Union[str, CommonViT],
    num_outputs: Optional[int],
    output_act: ActivationParameters = None,
    pretrained: bool = False,
) -> ViT:
    """
    To get a Vision Transformer implemented in the [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale]
    (https://arxiv.org/pdf/2010.11929) paper.

    Only the last fully connected layer will be changed to match `num_outputs`.

    The user can also use the pretrained models from `torchvision`. Note that the last fully connected layer will not
    used pretrained weights, as it is task specific.

    Note: the networks return the output of the network as well as patch embeddings after each transformer block.

    .. warning:: `ViT-B/16`, `ViT-B/32`, `ViT-L/16` and `ViT-L/32` work with 2D images of size (224, 224), with 3 channels.

    Parameters
    ----------
    model : Union[str, CommonViT]
        The name of the Vision Transformer. Available networks are `ViT-B/16`, `ViT-B/32`, `ViT-L/16` and `ViT-L/32`.
    num_outputs : Optional[int]
        number of output variables after the last linear layer.\n
        If None, the features before the last fully connected layer will be returned.
    output_act : ActivationParameters (optional, default=None)
        if `num_outputs` is not None, a potential activation layer applied to the outputs of the network,
        and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.
    pretrained : bool (optional, default=False)
        whether to use pretrained weights. The pretrained weights used are the default ones from [torchvision](https://
        pytorch.org/vision/main/models/vision_transformer.html).

    Returns
    -------
    ViT
        The network, with potentially pretrained weights.
    """
    model = CommonViT(model)
    if model == CommonViT.B_16:
        in_shape = (3, 224, 224)
        patch_size = 16
        embedding_dim = 768
        mlp_dim = 3072
        num_layers = 12
        num_heads = 12
        model_url = ViT_B_16_Weights.DEFAULT.url
    elif model == CommonViT.B_32:
        in_shape = (3, 224, 224)
        patch_size = 32
        embedding_dim = 768
        mlp_dim = 3072
        num_layers = 12
        num_heads = 12
        model_url = ViT_B_32_Weights.DEFAULT.url
    elif model == CommonViT.L_16:
        in_shape = (3, 224, 224)
        patch_size = 16
        embedding_dim = 1024
        mlp_dim = 4096
        num_layers = 24
        num_heads = 16
        model_url = ViT_L_16_Weights.DEFAULT.url
    elif model == CommonViT.L_32:
        in_shape = (3, 224, 224)
        patch_size = 32
        embedding_dim = 1024
        mlp_dim = 4096
        num_layers = 24
        num_heads = 16
        model_url = ViT_L_32_Weights.DEFAULT.url

    vit = ViT(
        in_shape=in_shape,
        patch_size=patch_size,
        num_outputs=num_outputs,
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_act=output_act,
    )

    if pretrained:
        pretrained_dict = load_state_dict_from_url(model_url, progress=True)
        if num_outputs:
            fc_layers = deepcopy(vit.classification_head)
            vit.classification_head = None
            vit.load_state_dict(_state_dict_adapter(pretrained_dict))
            vit.classification_head = fc_layers
        else:
            del pretrained_dict["class_token"]
            vit.load_state_dict(_state_dict_adapter(pretrained_dict))

    return vit


def _state_dict_adapter(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    A mapping between torchvision's layer names and ours.
    Also remove the position embedding of the classification token, as it does not
    exist in our implementation.
    """
    state_dict = {k: v for k, v in state_dict.items() if "heads" not in k}

    mappings = [
        ("class_token", "cls_token"),
        ("conv_proj", "patch_embedding.patch_embeddings"),
        ("encoder.pos_embedding", "patch_embedding.position_embeddings"),
        ("encoder.ln", "norm"),
        (r"encoder\.layers\.encoder_layer_(\d+)", r"blocks.\1"),
        ("self_attention", "attn"),
        ("in_proj_weight", "qkv.weight"),
        ("in_proj_bias", "qkv.bias"),
        ("ln", "norm"),
        (r"_(\d+)", r"\1"),
    ]

    for key in list(state_dict.keys()):
        new_key = key
        for transform in mappings:
            new_key = re.sub(transform[0], transform[1], new_key)
        state_dict[new_key] = state_dict.pop(key)

    state_dict["patch_embedding.position_embeddings"] = state_dict[
        "patch_embedding.position_embeddings"
    ][:, 1:]  # pos embedding for the classification token is always 0 here

    return state_dict
