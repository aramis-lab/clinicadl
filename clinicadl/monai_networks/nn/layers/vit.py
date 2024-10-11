from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import MLPBlock


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.norm1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.norm2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x += residual

        y = self.norm2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Encoder with multiple transformer blocks."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        pos_embedding: Optional[nn.Parameter] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()

        if pos_embedding is not None:
            self.pos_embedding = pos_embedding
        else:
            self.pos_embedding = nn.Parameter(
                torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
            )  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(
                EncoderBlock(
                    num_heads,
                    hidden_dim,
                    mlp_dim,
                    dropout,
                    attention_dropout,
                    norm_layer,
                )
            )
        self.layers = nn.Sequential(*layers)
        self.norm = norm_layer(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding
        return self.norm(self.layers(self.dropout(x)))
