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
from torch.hub import load_state_dict_from_url
from torchvision.models.vision_transformer import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
)

from .layers.utils import ActFunction, ActivationParameters
from .layers.vit import Encoder
from .utils import ensure_tuple

__all__ = [
    "ViT",
    "ViTB16",
    "ViTB32",
    "ViTL16",
    "ViTL32",
    "check_embedding_dim",
    "check_patch_size",
]


class PosEmbedType(str, Enum):
    """Available position embedding types for ViT."""

    LEARN = "learnable"
    SINCOS = "sincos"


class ViT(nn.Module):
    """
    Vision Transformer, based on `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Adapted from :torchvision:`torchvision's implementation <models/vision_transformer.html>`.

    The user can customize the patch size, the embedding dimension, the number of transformer blocks, the number of
    attention heads, as well as other parameters like the type of position embedding.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    Parameters
    ----------
    in_shape : Sequence[int]
        Dimensions of the input tensor (without batch dimension).
    patch_size : Union[Sequence[int], int]
        Patch size (without batch and channel dimensions). If ``int``, the same
        patch size will be used for all dimensions.
        ``patch_size`` must divide ``in_shape`` in all spatial dimensions.
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.\n
        If ``None``, the patch embeddings after the last transformer block will be returned.
    embedding_dim : int, default=768
        Size of the embedding vectors. Must be divisible by ``num_heads`` as each head will be responsible for
        a part of the embedding vectors. Default to ``768``, as ``ViT-Base`` in the original paper.
    num_layers : int, default=12
        Number of consecutive transformer blocks. Default to ``12``, as ``ViT-Base`` in the original paper.
    num_heads : int, default=12
        Number of heads in the self-attention blocks. Must divide ``embedding_dim``.
        Default to ``12``, as ``ViT-Base`` in the original paper.
    mlp_dim : int, default=3072
        Size of the hidden layer in the MLP part of the transformer block. Default to ``3072``, as ``ViT-Base``
        in the original paper.
    pos_embed_type : Optional[Union[str, PosEmbedType]], default="learnable"
        Type of position embedding. Can be either ``learnable``, ``sincos`` or ``None``:

        - ``learnable``: the position embeddings are parameters that will be learned during the training
          process.
        - ``sincos``: the position embeddings are fixed and determined with sinus and cosinus formulas described in
          :footcite:t:`Vaswani2023`. Only implemented for 2D and 3D images. With ``sincos``
          position embedding, ``embedding_dim`` must be divisible by ``4`` for 2D images, and by ``6`` for 3D images.
        - ``None``: no position embeddings are used.

        Default to ``learnable``, as in the original paper.
    output_act : Optional[ActivationParameters], default="tanh"
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.\n
        Default is ``tanh``, as in the original paper.
    dropout : Optional[float], default=None
        Dropout ratio. If ``None``, no dropout.

    Examples
    --------

    .. code-block:: python

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

    References
    ----------
    .. footbibliography::

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
        self.patch_size = ensure_tuple(patch_size, self.spatial_dims, "patch_size")

        check_embedding_dim(embedding_dim, num_heads)
        check_patch_size(self.patch_size, self.img_size)
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

    def _load_weights(self, url: str) -> None:
        """To load weights from torchvision."""
        pretrained_dict = load_state_dict_from_url(url, progress=True)

        if not self.classification:
            del pretrained_dict["class_token"]
            pretrained_dict["encoder.pos_embedding"] = pretrained_dict[
                "encoder.pos_embedding"
            ][:, 1:]  # remove class token position embedding

        fc_layers = deepcopy(self.fc)
        self.fc = None
        self.load_state_dict(_state_dict_adapter(pretrained_dict))
        self.fc = fc_layers


class ViTB16(ViT):
    """
    ViT-B/16, from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

    .. warning:: Only works with **2D images of size (224, 224), with 3 channels**.

    Parameters
    ----------
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.
        If ``None``, the feature map before the last fully connected layer will be returned.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.vit_b_16`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ViT`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__(
            in_shape=(3, 224, 224),
            patch_size=16,
            num_outputs=num_outputs,
            embedding_dim=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            output_act=output_act,
        )
        if pretrained:
            self._load_weights(ViT_B_16_Weights.DEFAULT.url)


class ViTB32(ViT):
    """
    ViT-B/32, from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

    .. warning:: Only works with **2D images of size (224, 224), with 3 channels**.

    Parameters
    ----------
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.
        If ``None``, the feature map before the last fully connected layer will be returned.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.vit_b_32`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ViT`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__(
            in_shape=(3, 224, 224),
            patch_size=32,
            num_outputs=num_outputs,
            embedding_dim=768,
            mlp_dim=3072,
            num_heads=12,
            num_layers=12,
            output_act=output_act,
        )
        if pretrained:
            self._load_weights(ViT_B_32_Weights.DEFAULT.url)


class ViTL16(ViT):
    """
    ViT-L/16, from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

    .. warning:: Only works with **2D images of size (224, 224), with 3 channels**.

    Parameters
    ----------
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.
        If ``None``, the feature map before the last fully connected layer will be returned.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.vit_l_16`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ViT`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__(
            in_shape=(3, 224, 224),
            patch_size=16,
            num_outputs=num_outputs,
            embedding_dim=1024,
            mlp_dim=4096,
            num_heads=16,
            num_layers=24,
            output_act=output_act,
        )
        if pretrained:
            self._load_weights(ViT_L_16_Weights.DEFAULT.url)


class ViTL32(ViT):
    """
    ViT-L/32, from `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Only the last fully connected layer will be changed to match ``num_outputs``.

    The user can use the pretrained models from ``torchvision``. Note that the last fully connected layer will not
    use pretrained weights, as it is task specific.

    .. warning:: Only works with **2D images of size (224, 224), with 3 channels**.

    Parameters
    ----------
    num_outputs : Optional[int]
        Number of output variables after the last linear layer.
        If ``None``, the feature map before the last fully connected layer will be returned.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.
    pretrained : bool, default=False
        Whether to use pretrained weights. The pretrained weights used are the default ones
        from :py:func:`torchvision.models.vit_l_32`.

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.ViT`

    """

    def __init__(
        self,
        num_outputs: Optional[int],
        output_act: Optional[ActivationParameters] = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__(
            in_shape=(3, 224, 224),
            patch_size=32,
            num_outputs=num_outputs,
            embedding_dim=1024,
            mlp_dim=4096,
            num_heads=16,
            num_layers=24,
            output_act=output_act,
        )
        if pretrained:
            self._load_weights(ViT_L_32_Weights.DEFAULT.url)


def check_embedding_dim(embedding_dim: int, num_heads: int) -> None:
    """
    Checks consistency between embedding dimension and number of heads.
    """
    if embedding_dim % num_heads != 0:
        raise ValueError(
            f"embedding_dim should be divisible by num_heads. Got embedding_dim={embedding_dim} "
            f" and num_heads={num_heads}"
        )


def check_patch_size(patch_size: Tuple[int, ...], img_size: Tuple[int, ...]) -> None:
    """
    Checks consistency between image size and patch size.
    """
    for i, p in zip(img_size, patch_size):
        if i % p != 0:
            raise ValueError(
                f"img_size should be divisible by patch_size. Got img_size={img_size} "
                f" and patch_size={patch_size}"
            )


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
