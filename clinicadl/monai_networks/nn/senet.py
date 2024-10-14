from enum import Enum
from typing import Any, Optional, Sequence, Union

from clinicadl.utils.factories import get_args_and_defaults

from .resnet import GeneralResNet, ResNet, ResNetBlockType
from .utils import ActivationParameters


class SEResNet(GeneralResNet):
    """
    Squeeze-and-Excitation ResNet based on the [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507)
    paper.

    SEResNet is very similar to :py:class:`clinicadl.monai_networks.nn.resnet.ResNet`, with the only difference
    being that Squeeze-and-Excitation blocks are added before residual connections.

    Parameters
    ----------
    spatial_dims : int
        number of spatial dimensions of the input image.
    in_channels : int
        number of channels in the input image.
    num_outputs : Optional[int]
        number of output variables after the last linear layer.\n
        If None, the features before the last fully connected layer (including average pooling) will be returned.
    se_reduction : int (optional, default=16)
        reduction ratio in the bottelneck layer of the excitation modules. Default to 16, as in the original
        paper.
    kwargs : Any
        any optional argument accepted by :py:class:`clinicadl.monai_networks.nn.resnet.ResNet`.

    Examples
    --------
    >>> SEResNet(
            spatial_dims=2,
            in_channels=1,
            num_outputs=2,
            block_type="basic",
            se_reduction=2,
            n_features=(8,),
            n_res_blocks=(2,),
            output_act="softmax",
            init_conv_size=5,
        )
    SEResNet(
        (conv0): Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        (norm0): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act0): ReLU(inplace=True)
        (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (layer1): Sequential(
            (0): SEResNetBlock(
                (conv1): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (norm1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act1): ReLU(inplace=True)
                (conv2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (norm2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (se_layer): ChannelSELayer(
                    (avg_pool): AdaptiveAvgPool2d(output_size=1)
                    (fc): Sequential(
                        (0): Linear(in_features=8, out_features=4, bias=True)
                        (1): ReLU(inplace=True)
                        (2): Linear(in_features=4, out_features=8, bias=True)
                        (3): Sigmoid()
                    )
                )
                (act2): ReLU(inplace=True)
            )
            (1): SEResNetBlock(
                (conv1): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (norm1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (act1): ReLU(inplace=True)
                (conv2): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (norm2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (se_layer): ChannelSELayer(
                    (avg_pool): AdaptiveAvgPool2d(output_size=1)
                    (fc): Sequential(
                        (0): Linear(in_features=8, out_features=4, bias=True)
                        (1): ReLU(inplace=True)
                        (2): Linear(in_features=4, out_features=8, bias=True)
                        (3): Sigmoid()
                    )
                )
                (act2): ReLU(inplace=True)
            )
        )
        (fc): Sequential(
            (pool): AdaptiveAvgPool2d(output_size=(1, 1))
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (out): Linear(in_features=8, out_features=2, bias=True)
            (output_act): Softmax(dim=None)
        )
    )
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_outputs: Optional[int],
        se_reduction: int = 16,
        **kwargs: Any,
    ) -> None:
        # get defaults from resnet
        _, default_resnet_args = get_args_and_defaults(ResNet.__init__)
        for arg, value in default_resnet_args.items():
            if arg not in kwargs:
                kwargs[arg] = value

        self._check_se_channels(kwargs["n_features"], se_reduction)

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_outputs=num_outputs,
            se_reduction=se_reduction,
            **kwargs,
        )

    @classmethod
    def _check_se_channels(cls, n_features: Sequence[int], se_reduction: int) -> None:
        """
        Checks that the output of residual blocks always have a number of channels greater
        than squeeze-excitation bottleneck reduction factor.
        """
        if not isinstance(n_features, Sequence):
            raise ValueError(f"n_features must be a sequence. Got {n_features}")
        for n in n_features:
            if n < se_reduction:
                raise ValueError(
                    f"elements of n_features must be greater or equal to se_reduction. Got {n} in n_features "
                    f"and se_reduction={se_reduction}"
                )


class CommonSEResNet(str, Enum):
    """Supported SEResNet networks."""

    SE_RESNET_50 = "SE-ResNet-50"
    SE_RESNET_101 = "SE-ResNet-101"
    SE_RESNET_152 = "SE-ResNet-152"


def get_seresnet(
    model: Union[str, CommonSEResNet],
    num_outputs: Optional[int],
    output_act: ActivationParameters = None,
) -> SEResNet:
    """
    To get a Squeeze-and-Excitation ResNet implemented in the [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/
    1709.01507) paper.

    Only the last fully connected layer will be changed to match `num_outputs`.

    .. warning:: `SE-ResNet-50`, `SE-ResNet-101` and `SE-ResNet-152` only works with 2D images with 3 channels.

    Note: pretrained weights are not yet available for these networks.

    Parameters
    ----------
    model : Union[str, CommonSEResNet]
        the name of the SEResNet. Available networks are `SE-ResNet-50`, `SE-ResNet-101` and `SE-ResNet-152`.
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

    Returns
    -------
    SEResNet
        the network.
    """
    model = CommonSEResNet(model)
    if model == CommonSEResNet.SE_RESNET_50:
        block_type = ResNetBlockType.BOTTLENECK
        n_res_blocks = (3, 4, 6, 3)
        n_features = (256, 512, 1024, 2048)
    elif model == CommonSEResNet.SE_RESNET_101:
        block_type = ResNetBlockType.BOTTLENECK
        n_res_blocks = (3, 4, 23, 3)
        n_features = (256, 512, 1024, 2048)
    elif model == CommonSEResNet.SE_RESNET_152:
        block_type = ResNetBlockType.BOTTLENECK
        n_res_blocks = (3, 8, 36, 3)
        n_features = (256, 512, 1024, 2048)

    resnet = SEResNet(
        spatial_dims=2,
        in_channels=3,
        num_outputs=num_outputs,
        n_res_blocks=n_res_blocks,
        block_type=block_type,
        n_features=n_features,
        output_act=output_act,
    )

    return resnet
