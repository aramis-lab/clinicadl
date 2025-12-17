from typing import Optional, Sequence

import torch
from pydantic import Field

from clinicadl.transforms.handlers import Postprocessing
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.objects import HasConfig

from .base import BaseInferer, BaseInfererConfig


class SimpleInfererConfig(BaseInfererConfig):
    """Config class for ``SimpleInferer``."""

    postprocessing: Postprocessing = Field(reader=Postprocessing.from_dict)
    postprocessing_on_cpu: bool

    @classmethod
    def _get_class(cls):
        return SimpleInferer


class SimpleInferer(BaseInferer, HasConfig[SimpleInfererConfig]):
    """
    For classical inference, i.e. when the whole images are passed
    in the neural network and the raw outputs are returned (with a potential
    postprocessing).

    Parameters
    ----------
    postprocessing : Optional[Sequence[TransformOrConfig]], default=None
        To apply postprocessing transformations (e.g. activations) after the pass forward
        in the neural network.

        .. important::
            If you postprocessing transform comes from :py:class:`clinicadl.transforms.config`,
            do not forget to specify ``include=["output"]`` to apply the postprocessing to
            the output of the neural network.

    postprocessing_on_cpu : bool, default=False
        Whether to necessarily apply postprocessing on CPU. If ``False``, postprocessing will
        be applied on the device where are the data and the neural network.

        .. important::
            ``postprocessing_on_cpu=True`` may potentially change the device on which
            are your input data.

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
    ):
        if not postprocessing:
            postprocessing = []
        postprocessing = Postprocessing(postprocessing)
        self.config = self._config_type(
            postprocessing=postprocessing, postprocessing_on_cpu=postprocessing_on_cpu
        )

    def _forward_pass(
        self, tensor: torch.Tensor, network: torch.nn.Module, **kwargs
    ) -> torch.Tensor:
        return network(tensor, **kwargs)
