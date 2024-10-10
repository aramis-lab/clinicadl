"""This module only aims to override MONAI's module representations, in order to have submodules in the right order."""

from monai.networks.blocks.mlp import MLPBlock
from monai.networks.blocks.selfattention import SABlock
from monai.networks.blocks.transformerblock import (
    TransformerBlock as BaseTransformerBlock,
)
from torch.nn.modules.module import _addindent


class TransformerBlock(BaseTransformerBlock):
    """
    A transformer block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __repr__(self):
        lines = [
            "(norm1): " + _addindent(repr(self.norm1), 2),
            "(attn): " + _addindent(_attn_repr(self.attn), 2),
            "(norm2): " + _addindent(repr(self.norm2), 2),
            "(mlp): " + _addindent(_mlp_repr(self.mlp), 2),
        ]

        main_str = self._get_name() + "("
        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


def _mlp_repr(mlp: MLPBlock) -> str:
    """Representation of the mlp block."""
    lines = [
        "(linear1): " + _addindent(repr(mlp.linear1), 2),
        "(fn): " + _addindent(repr(mlp.fn), 2),
        "(drop1): " + _addindent(repr(mlp.drop1), 2),
        "(linear2): " + _addindent(repr(mlp.linear2), 2),
        "(drop2): " + _addindent(repr(mlp.drop2), 2),
    ]

    main_str = mlp._get_name() + "("
    main_str += "\n  " + "\n  ".join(lines) + "\n"
    main_str += ")"
    return main_str


def _attn_repr(sa: SABlock) -> str:
    """Representation of the attention block."""
    lines = [
        "(qkv): " + _addindent(repr(sa.qkv), 2),
        "(input_rearrange): " + _addindent(repr(sa.input_rearrange), 2),
        "(drop_weights): " + _addindent(repr(sa.drop_weights), 2),
        "(out_rearrange): " + _addindent(repr(sa.out_rearrange), 2),
        "(drop_output): " + _addindent(repr(sa.drop_output), 2),
    ]

    main_str = sa._get_name() + "("
    main_str += "\n  " + "\n  ".join(lines) + "\n"
    main_str += ")"
    return main_str
