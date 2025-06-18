from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .autoencoder import AutoEncoder
from .layers.utils import ActivationParameters, UnpoolingMode


class VAE(nn.Module):
    """
    A Variational AutoEncoder with convolutional and fully connected layers.

    The user must pass the arguments to build an encoder, from its convolutional and
    fully connected parts, and the decoder will be automatically built by taking the
    symmetrical network.

    More precisely, to build the decoder, the order of the encoding layers is reverted, convolutions are
    replaced by transposed convolutions, and pooling layers are replaced by either upsampling or transposed
    convolution layers.

    A ``VAE`` is very similar to a :py:class:`~clinicadl.networks.nn.AutoEncoder`, except that the last layer
    of the MLP part is duplicated to infer both the mean and the log variance. Besides, to sample from the
    latent distribution, the :wikipedia:`reparametrization trick <Variational_autoencoder#Reparameterization>`
    is performed with :py:func:`~VAE.reparameterize`.

    Works with 2D or 3D images (with additional batch and channel dimensions).

    .. note::
        Please note that the order of Activation, Dropout and Normalization, defined with the
        argument ``adn_ordering`` in ``conv_args``, is the same for the encoder and the decoder.

    Parameters
    ----------
    in_shape : Sequence[int]
        Dimensions of the input tensor (without batch dimension).
    latent_size : int
        Size of the latent vector.
    conv_args : Dict[str, Any]
        The arguments for the convolutional part. The arguments are those accepted by
        :py:class:`~clinicadl.networks.nn.ConvEncoder`, except ``spatial_dims`` and ``in_channels``
        that are specified here via ``in_shape``. So, the only **mandatory argument is** ``channels``.
    mlp_args : Optional[Dict[str, Any]], default=None
        The arguments for the MLP part. The arguments are those accepted by
        :py:class:`~clinicadl.networks.nn.MLP`, except ``num_inputs`` that is inferred
        from the output of the convolutional part, and ``num_outputs`` that is equal to ``latent_size`` here.
        So, the only **mandatory argument is** ``hidden_dims``.\n
        If ``None``, the MLP part will be reduced to a single linear layer.\n
        The last linear layer will be duplicated to infer both the mean and the log variance.
    out_channels : Optional[int], default=None
        Number of output channels. If ``None``, the output will have the same number of channels as the
        input.
    output_act : Optional[ActivationParameters], default=None
        A potential activation layer applied to the output of the network, and optionally its arguments.
        Must be passed as ``activation_name`` or ``(activation_name, arguments)``, where ``arguments`` is a dictionary.
        If ``None``, no activation will be used.\n
        ``activation_name`` can be any value in {``celu``, ``elu``, ``gelu``, ``leakyrelu``, ``logsoftmax``, ``mish``, ``prelu``,
        ``relu``, ``relu6``, ``selu``, ``sigmoid``, ``softmax``, ``tanh``}. Please refer to
        :torch:`PyTorch activation functions <nn.html#non-linear-activations-weighted-sum-nonlinearity>` to know the arguments
        for each of them.
    unpooling_mode : Union[str, UnpoolingMode], default=UnpoolingMode.NEAREST
        Type of unpooling. Can be any value in {``nearest``, ``linear``, ``bilinear``, ``bicubic``, ``trilinear`` or
        ``convtranspose``}:

        - ``nearest``: unpooling is performed by upsampling with the `nearest` algorithm (see
          :py:class:`torch.nn.Upsample`);
        - ``linear``: unpooling is performed by upsampling with the `linear` algorithm. Only works with 1D images (excluding the
          channel dimension);
        - ``bilinear``: unpooling is performed by upsampling with the `bilinear` algorithm. Only works with 2D images;
        - ``bicubic``: unpooling is performed by upsampling with the `bicubic` algorithm. Only works with 2D images;
        - ``trilinear``: unpooling is performed by upsampling with the `trilinear` algorithm. Only works with 3D images;
        - ``convtranspose``: unpooling is performed with a transposed convolution (see :py:class:`torch.nn.ConvTranspose3d`), whose
          parameters (kernel size, stride, etc.) are computed to reverse the pooling operation.

    Examples
    --------

    .. code-block:: python

        >>> VAE(
                in_shape=(1, 16, 16),
                latent_size=4,
                conv_args={"channels": [2]},
                mlp_args={"hidden_dims": [16], "output_act": "relu"},
                out_channels=2,
                output_act="sigmoid",
                unpooling_mode="bilinear",
            )
        VAE(
            (encoder): CNN(
                (convolutions): ConvEncoder(
                    (layer0): Convolution(
                        (conv): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
                    )
                )
                (mlp): MLP(
                    (flatten): Flatten(start_dim=1, end_dim=-1)
                    (hidden0): Sequential(
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
                    (hidden0): Sequential(
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
                    (layer0): Convolution(
                        (conv): ConvTranspose2d(2, 2, kernel_size=(3, 3), stride=(1, 1))
                    )
                    (output_act): Sigmoid()
                )
            )
        )

    See Also
    --------
    :py:class:`~clinicadl.networks.nn.AutoEncoder`
    """

    def __init__(
        self,
        in_shape: Sequence[int],
        latent_size: int,
        conv_args: Dict[str, Any],
        mlp_args: Optional[Dict[str, Any]] = None,
        out_channels: Optional[int] = None,
        output_act: Optional[ActivationParameters] = None,
        unpooling_mode: Union[str, UnpoolingMode] = UnpoolingMode.NEAREST,
    ) -> None:
        super().__init__()
        ae = AutoEncoder(
            in_shape,
            latent_size,
            conv_args,
            mlp_args,
            out_channels,
            output_act,
            unpooling_mode,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def _reset_weights(cls, layer: nn.Sequential) -> None:
        """
        Resets the output layer(s) of an MLP.
        """
        layer.linear.reset_parameters()
