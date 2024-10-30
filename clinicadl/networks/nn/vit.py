import math
import re
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks.pos_embed_utils import build_sincos_position_embedding
from monai.networks.layers import Conv
from monai.networks.layers.utils import get_act_layer
from monai.utils import ensure_tuple_rep
from torch.hub import load_state_dict_from_url
from torchvision.models.vision_transformer import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)

from .layers.utils import ActFunction, ActivationParameters
from .layers.vit import Encoder


class PosEmbedType(str, Enum):
    """Available position embedding types for ViT."""

    LEARN = "learnable"
    SINCOS = "sincos"


class ViT(nn.Module):
    """
    Vision Transformer based on the [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale]
    (https://arxiv.org/pdf/2010.11929) paper.
    Adapted from [torchvision's implementation](https://pytorch.org/vision/main/models/vision_transformer.html).

    The user can customize the patch size, the embedding dimension, the number of transformer blocks, the number of
    attention heads, as well as other parameters like the type of position embedding.

    Parameters
    ----------
    in_shape : Sequence[int]
        sequence of integers stating the dimension of the input tensor (minus batch dimension).
    patch_size : Union[Sequence[int], int]
        sequence of integers stating the patch size (minus batch and channel dimensions). If int, the same
        patch size will be used for all dimensions.
        Patch size must divide image size in all dimensions.
    num_outputs : Optional[int]
        number of output variables after the last linear layer.\n
        If None, the patch embeddings after the last transformer block will be returned.
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
        type of position embedding. Can be either `"learnable"`, `"sincos"` or `None`.\n
        - `learnable`: the position embeddings are parameters that will be learned during the training
        process.
        - `sincos`: the position embeddings are fixed and determined with sinus and cosinus formulas (based on Dosovitskiy et al.,
        'Attention Is All You Need, https://arxiv.org/pdf/1706.03762). Only implemented for 2D and 3D images. With `sincos`
        position embedding, `embedding_dim` must be divisible by 4 for 2D images and by 6 for 3D images.
        - `None`: no position embeddings are used.\n
        Default to `"learnable"`, as in the original paper.
    output_act : Optional[ActivationParameters] (optional, default=ActFunction.TANH)
        if `num_outputs` is not None, a potential activation layer applied to the outputs of the network,
        and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.\n
        Default to `"tanh"`, as in the original paper.
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
        (conv_proj): Conv2d(3, 32, kernel_size=(4, 4), stride=(4, 4))
        (encoder): Encoder(
            (dropout): Dropout(p=0.0, inplace=False)
            (layers): ModuleList(
                (0-1): 2 x EncoderBlock(
                    (norm1): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
                    (self_attention): MultiheadAttention(
                        (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
                    )
                    (dropout): Dropout(p=0.0, inplace=False)
                    (norm2): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
                    (mlp): MLPBlock(
                        (0): Linear(in_features=32, out_features=128, bias=True)
                        (1): GELU(approximate='none')
                        (2): Dropout(p=0.0, inplace=False)
                        (3): Linear(in_features=128, out_features=32, bias=True)
                        (4): Dropout(p=0.0, inplace=False)
                    )
                )
            )
            (norm): LayerNorm((32,), eps=1e-06, elementwise_affine=True)
        )
        (fc): Sequential(
            (out): Linear(in_features=32, out_features=2, bias=True)
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
        output_act: Optional[ActivationParameters] = ActFunction.TANH,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.in_channels, *self.img_size = in_shape
        self.spatial_dims = len(self.img_size)
        self.patch_size = ensure_tuple_rep(patch_size, self.spatial_dims)

        self._check_embedding_dim(embedding_dim, num_heads)
        self._check_patch_size(self.img_size, self.patch_size)
        self.embedding_dim = embedding_dim
        self.classification = True if num_outputs else False
        dropout = dropout if dropout else 0.0

        self.conv_proj = Conv[Conv.CONV, self.spatial_dims](  # pylint: disable=not-callable
            in_channels=self.in_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.seq_length = int(
            np.prod(np.array(self.img_size) // np.array(self.patch_size))
        )

        # Add a class token
        if self.classification:
            self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
            self.seq_length += 1

        pos_embedding = self._get_pos_embedding(pos_embed_type)
        self.encoder = Encoder(
            self.seq_length,
            num_layers,
            num_heads,
            self.embedding_dim,
            mlp_dim,
            dropout=dropout,
            attention_dropout=dropout,
            pos_embedding=pos_embedding,
        )

        if self.classification:
            self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
            self.fc = nn.Sequential(
                OrderedDict([("out", nn.Linear(embedding_dim, num_outputs))])
            )
            self.fc.output_act = get_act_layer(output_act) if output_act else None
        else:
            self.fc = None

        self._init_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, (h * w * d), hidden_dim)
        x = x.flatten(2).transpose(-1, -2)
        n = x.shape[0]

        # Expand the class token to the full batch
        if self.fc:
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        if self.fc:
            x = x[:, 0]
            x = self.fc(x)

        return x

    def _get_pos_embedding(
        self, pos_embed_type: Optional[Union[str, PosEmbedType]]
    ) -> Optional[nn.Parameter]:
        """
        Gets position embeddings. If `pos_embed_type` is "learnable", will return None as it will be handled
        by the encoder module.
        """
        if pos_embed_type is None:
            pos_embed = nn.Parameter(
                torch.zeros(1, self.seq_length, self.embedding_dim)
            )
            pos_embed.requires_grad = False
            return pos_embed

        pos_embed_type = PosEmbedType(pos_embed_type)

        if pos_embed_type == PosEmbedType.LEARN:
            return None  # will be initialized inside the Encoder

        elif pos_embed_type == PosEmbedType.SINCOS:
            if self.spatial_dims != 2 and self.spatial_dims != 3:
                raise ValueError(
                    f"{self.spatial_dims}D sincos position embedding not implemented"
                )
            elif self.spatial_dims == 2 and self.embedding_dim % 4:
                raise ValueError(
                    f"embedding_dim must be divisible by 4 for 2D sincos position embedding. Got embedding_dim={self.embedding_dim}"
                )
            elif self.spatial_dims == 3 and self.embedding_dim % 6:
                raise ValueError(
                    f"embedding_dim must be divisible by 6 for 3D sincos position embedding. Got embedding_dim={self.embedding_dim}"
                )
            grid_size = []
            for in_size, pa_size in zip(self.img_size, self.patch_size):
                grid_size.append(in_size // pa_size)
            pos_embed = build_sincos_position_embedding(
                grid_size, self.embedding_dim, self.spatial_dims
            )
            if self.classification:
                pos_embed = torch.nn.Parameter(
                    torch.cat([torch.zeros(1, 1, self.embedding_dim), pos_embed], dim=1)
                )  # add 0 for class token pos embedding
                pos_embed.requires_grad = False
            return pos_embed

    def _init_layers(self):
        """
        Initializes some layers, based on torchvision's implementation: https://pytorch.org/vision/main/
        _modules/torchvision/models/vision_transformer.html
        """
        fan_in = self.conv_proj.in_channels * np.prod(self.conv_proj.kernel_size)
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(self.conv_proj.bias)

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

    @classmethod
    def _check_patch_size(
        cls, img_size: Tuple[int, ...], patch_size: Tuple[int, ...]
    ) -> None:
        """
        Checks consistency between image size and patch size.
        """
        for i, p in zip(img_size, patch_size):
            if i % p != 0:
                raise ValueError(
                    f"img_size should be divisible by patch_size. Got img_size={img_size} "
                    f" and patch_size={patch_size}"
                )


class SOTAViT(str, Enum):
    """Supported ViT networks."""

    B_16 = "ViT-B/16"
    B_32 = "ViT-B/32"
    L_16 = "ViT-L/16"
    L_32 = "ViT-L/32"


def get_vit(
    name: Union[str, SOTAViT],
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

    .. warning:: `ViT-B/16`, `ViT-B/32`, `ViT-L/16` and `ViT-L/32` work with 2D images of size (224, 224), with 3 channels.

    Parameters
    ----------
    model : Union[str, SOTAViT]
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
    name = SOTAViT(name)
    if name == SOTAViT.B_16:
        in_shape = (3, 224, 224)
        patch_size = 16
        embedding_dim = 768
        mlp_dim = 3072
        num_layers = 12
        num_heads = 12
        model_url = ViT_B_16_Weights.DEFAULT.url
    elif name == SOTAViT.B_32:
        in_shape = (3, 224, 224)
        patch_size = 32
        embedding_dim = 768
        mlp_dim = 3072
        num_layers = 12
        num_heads = 12
        model_url = ViT_B_32_Weights.DEFAULT.url
    elif name == SOTAViT.L_16:
        in_shape = (3, 224, 224)
        patch_size = 16
        embedding_dim = 1024
        mlp_dim = 4096
        num_layers = 24
        num_heads = 16
        model_url = ViT_L_16_Weights.DEFAULT.url
    elif name == SOTAViT.L_32:
        in_shape = (3, 224, 224)
        patch_size = 32
        embedding_dim = 1024
        mlp_dim = 4096
        num_layers = 24
        num_heads = 16
        model_url = ViT_L_32_Weights.DEFAULT.url

    # pylint: disable=possibly-used-before-assignment
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
        if num_outputs is None:
            del pretrained_dict["class_token"]
            pretrained_dict["encoder.pos_embedding"] = pretrained_dict[
                "encoder.pos_embedding"
            ][:, 1:]  # remove class token position embedding
        fc_layers = deepcopy(vit.fc)
        vit.fc = None
        vit.load_state_dict(_state_dict_adapter(pretrained_dict))
        vit.fc = fc_layers

    return vit


def _state_dict_adapter(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    A mapping between torchvision's layer names and ours.
    """
    state_dict = {k: v for k, v in state_dict.items() if "heads" not in k}

    mappings = [
        ("ln_", "norm"),
        ("ln", "norm"),
        (r"encoder_layer_(\d+)", r"\1"),
    ]

    for key in list(state_dict.keys()):
        new_key = key
        for transform in mappings:
            new_key = re.sub(transform[0], transform[1], new_key)
        state_dict[new_key] = state_dict.pop(key)

    return state_dict
