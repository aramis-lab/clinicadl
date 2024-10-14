from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from .autoencoder import AutoEncoder
from .layers.utils import ActivationParameters, UpsamplingMode


class VAE(nn.Module):
    """
    A Variational AutoEncoder with convolutional and fully connected layers.

    The user must pass the arguments to build an encoder, from its convolutional and
    fully connected parts, and the decoder will be automatically built by taking the
    symmetrical network.

    More precisely, to build the decoder, the order of the encoding layers is reverted, convolutions are
    replaced by transposed convolutions and pooling layers are replaced by upsampling layers.
    Please note that the order of `Activation`, `Dropout` and `Normalization`, defined with the
    argument `adn_ordering` in `conv_args`, is the same for the encoder and the decoder.

    Note that an `AutoEncoder` is an aggregation of a `CNN` (:py:class:`clinicadl.monai_networks.nn.
    cnn.CNN`), whose last linear layer is duplicated to infer both the mean and the log variance,
    and a `Generator` (:py:class:`clinicadl.monai_networks.nn.generator.Generator`).

    Parameters
    ----------
    in_shape : Sequence[int]
        sequence of integers stating the dimension of the input tensor (minus batch dimension).
    latent_size : int
        size of the latent vector.
    conv_args : Dict[str, Any]
        the arguments for the convolutional part of the encoder. The arguments are those accepted
        by :py:class:`clinicadl.monai_networks.nn.conv_encoder.ConvEncoder`, except `in_shape` that
        is specified here. So, the only mandatory argument is `channels`.
    mlp_args : Optional[Dict[str, Any]] (optional, default=None)
        the arguments for the MLP part of the encoder . The arguments are those accepted by
        :py:class:`clinicadl.monai_networks.nn.mlp.MLP`, except `in_channels` that is inferred
        from the output of the convolutional part, and `out_channels` that is set to `latent_size`.
        So, the only mandatory argument is `hidden_channels`.\n
        If None, the MLP part will be reduced to a single linear layer.\n
        The last linear layer will be duplicated to infer both the mean and the log variance.
    out_channels : Optional[int] (optional, default=None)
        number of output channels. If None, the output will have the same number of channels as the
        input.
    output_act : Optional[ActivationParameters] (optional, default=None)
        a potential activation layer applied to the output of the network, and optionally its arguments.
        Should be passed as `activation_name` or `(activation_name, arguments)`. If None, no activation will be used.\n
        `activation_name` can be any value in {`celu`, `elu`, `gelu`, `leakyrelu`, `logsoftmax`, `mish`, `prelu`,
        `relu`, `relu6`, `selu`, `sigmoid`, `softmax`, `tanh`}. Please refer to PyTorch's [activationfunctions]
        (https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) to know the optional
        arguments for each of them.
    upsampling_mode : Union[str, UpsamplingMode] (optional, default=UpsamplingMode.NEAREST)
        interpolation mode for upsampling (see: https://pytorch.org/docs/stable/generated/
        torch.nn.Upsample.html).

    Examples
    --------
    >>> VAE(
            in_shape=(1, 16, 16),
            latent_size=4,
            conv_args={"channels": [2]},
            mlp_args={"hidden_channels": [16], "output_act": "relu"},
            out_channels=2,
            output_act="sigmoid",
            upsampling_mode="bilinear",
        )
    VAE(
        (encoder): CNN(
            (convolutions): ConvEncoder(
                (layer_0): Convolution(
                    (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
                )
            )
            (mlp): MLP(
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (hidden_0): Sequential(
                    (linear): Linear(in_features=392, out_features=16, bias=True)
                    (adn): ADN(
                        (N): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (A): PReLU(num_parameters=1)
                    )
                )
                (output): Identity()
            )
        )
        (mu): Sequential(
            (linear): Linear(in_features=16, out_features=4, bias=True)
            (output_act): ReLU()
        )
        (log_var): Sequential(
            (linear): Linear(in_features=16, out_features=4, bias=True)
            (output_act): ReLU()
        )
        (decoder): Generator(
            (mlp): MLP(
                (flatten): Flatten(start_dim=1, end_dim=-1)
                (hidden_0): Sequential(
                    (linear): Linear(in_features=4, out_features=16, bias=True)
                    (adn): ADN(
                        (N): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                        (A): PReLU(num_parameters=1)
                    )
                )
                (output): Sequential(
                    (linear): Linear(in_features=16, out_features=392, bias=True)
                    (output_act): ReLU()
                )
            )
            (reshape): Reshape()
            (convolutions): ConvDecoder(
                (layer_0): Convolution(
                    (conv): ConvTranspose2d(2, 2, kernel_size=(3, 3), stride=(1, 1))
                )
                (output_act): Sigmoid()
            )
        )
    )
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        latent_size: int,
        conv_args: Dict[str, Any],
        mlp_args: Optional[Dict[str, Any]] = None,
        out_channels: Optional[int] = None,
        output_act: Optional[ActivationParameters] = None,
        upsampling_mode: Union[str, UpsamplingMode] = UpsamplingMode.NEAREST,
    ) -> None:
        super().__init__()
        ae = AutoEncoder(
            in_shape,
            latent_size,
            conv_args,
            mlp_args,
            out_channels,
            output_act,
            upsampling_mode,
        )

        # replace last mlp layer by two parallel layers
        mu_layers = deepcopy(ae.encoder.mlp.output)
        log_var_layers = deepcopy(ae.encoder.mlp.output)
        self._reset_weights(
            log_var_layers
        )  # to have different initialization for the two layers
        ae.encoder.mlp.output = nn.Identity()

        self.encoder = ae.encoder
        self.mu = mu_layers
        self.log_var = log_var_layers
        self.decoder = ae.decoder

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoding, sampling and decoding.
        """
        feature = self.encoder(x)
        mu = self.mu(feature)
        log_var = self.log_var(feature)
        z = self.reparameterize(mu, log_var)

        return self.decoder(z), mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Samples a random vector from a gaussian distribution, given the mean and log-variance
        of this distribution.
        """
        std = torch.exp(0.5 * log_var)

        if self.training:  # multiply random noise with std only during training
            std = torch.randn_like(std).mul(std)

        return std.add_(mu)

    @classmethod
    def _reset_weights(cls, layer: Union[nn.Sequential, nn.Linear]) -> None:
        """
        Resets the output layer(s) of an MLP.
        """
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()
        else:
            layer.linear.reset_parameters()
