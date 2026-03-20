from typing import Callable, Optional, Sequence

import torch

from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.dictionary.words import OUTPUT
from clinicadl.utils.objects import HasConfig

from .base import BaseInferer, BaseInfererConfig, OutputType


class SimpleInfererConfig(BaseInfererConfig):
    """Config class for ``SimpleInferer``."""

    @classmethod
    def _get_class(cls):
        return SimpleInferer


class SimpleInferer(BaseInferer, HasConfig[SimpleInfererConfig]):
    """
    An :py:class:`clinicadl.infer.Inferer` for classical inference, i.e. when the whole image is passed
    in the neural network and the raw output is returned (with a potential
    postprocessing).

    The inference output will be added to the input :py:class:`~clinicadl.data.structures.DataPoint`
    (or to each ``DataPoint`` of the input :py:class:`~clinicadl.data.dataloader.Batch`).

    Parameters
    ----------
    postprocessing : Optional[Sequence[TransformOrConfig]], default=None
        To apply postprocessing transformations (e.g. activations) after the pass forward
        in the neural network.

    postprocessing_on_cpu : bool, default=False
        Whether to necessarily apply postprocessing on CPU. If ``False``, postprocessing will
        be applied on the device where are the data and the neural network.

        .. important::
            ``postprocessing_on_cpu=True`` may potentially change the device on which
            are your input data.

    output_name : str, default="output"
        The name the give to the output in the ``DataPoint``.

        .. important::
            If you postprocessing transform comes from :py:class:`clinicadl.transforms.config`,
            do not forget to specify ``include=["<output_name>"]`` to apply the postprocessing to
            the output of the neural network.

    output_type : OutputType, default="tensor"
        Determines the data type of the output:

        - if ``"image"``, the output will be converted to a :py:class:`torchio.ScalarImage`;
        - if ``"mask"``, the output will be converted to a :py:class:`torchio.LabeMap`;
        - if ``"tensor"``, the output will remain a :py:class:`torch.Tensor`.

    Examples
    --------

    .. code-block::

        import torch
        from clinicadl.infer import SimpleInferer
        from clinicadl.data.structures.examples import ColinDataPoint
        from clinicadl.data.dataloader import Batch
        from clinicadl.networks.nn import ConvEncoder
        from clinicadl.transforms.config import ActivationsConfig

        net = ConvEncoder(spatial_dims=3, in_channels=1, channels=[2, 4])
        datapoint = ColinDataPoint()

    .. code-block::

        >>> datapoint.image.shape
        (1, 181, 217, 181)
        >>> inferer = SimpleInferer()
        >>> with torch.no_grad(): out = inferer(datapoint, net)
        >>> out["output"].shape
        (4, 177, 213, 177)
        >>> out["output"].tensor
        tensor([[[[ 0.5923,  0.5979,  0.5816,  ...,  0.6592,  0.6592,  0.6592],
                [ 0.5887,  0.5832,  0.5785,  ...,  0.6592,  0.6592,  0.6592],
                [ 0.5883,  0.5832,  0.5858,  ...,  0.6592,  0.6592,  0.6592],
                ...,

    Working with a specific precision:

    .. code-block::

        >>> net.to(dtype=torch.half)
        >>> with torch.no_grad(): out = inferer(datapoint, net, input_dtype=torch.half)
        >>> out["output"].tensor.dtype
        torch.float16

    With postprocessing:

        >>> net.to(dtype=torch.float)
        >>> inferer = SimpleInferer(postprocessing=[ActivationsConfig(sigmoid=True, include=["output"])])
        >>> with torch.no_grad(): out = inferer(datapoint, net)
        >>> out["output"].tensor
        tensor([[[[0.6439, 0.6452, 0.6414,  ..., 0.6591, 0.6591, 0.6591],
                [0.6430, 0.6418, 0.6407,  ..., 0.6591, 0.6591, 0.6591],
                [0.6429, 0.6418, 0.6424,  ..., 0.6591, 0.6591, 0.6591],
                ...,

    With a :py:class:`clinicadl.data.dataloader.Batch`:

        >>> from copy import deepcopy
        >>> batch = Batch([datapoint, deepcopy(datapoint)])
        >>> with torch.no_grad(): out = inferer(batch, net)
        >>> out[0]["output"].shape
        (4, 177, 213, 177)

    """

    _config_type = SimpleInfererConfig

    def __init__(
        self,
        postprocessing: Optional[Sequence[TransformOrConfig]] = None,
        postprocessing_on_cpu: bool = False,
        output_name: str = OUTPUT,
        output_type: OutputType = OutputType.TENSOR,
    ):
        super().__init__(
            postprocessing=postprocessing,
            postprocessing_on_cpu=postprocessing_on_cpu,
            output_name=output_name,
            output_type=output_type,
        )

    def _forward_pass(
        self, tensor: torch.Tensor, network: Callable[..., torch.Tensor], **kwargs
    ) -> torch.Tensor:
        return network(tensor, **kwargs)
