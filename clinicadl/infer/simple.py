from logging import getLogger
from typing import Any, Optional, Sequence, TypeVar, Union, overload

import torch
import torch.nn as nn
import torchio as tio
from pydantic import Field

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.transforms.handlers import Postprocessing
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import OUTPUT
from clinicadl.utils.objects import HasConfig

from .base import Inferer

logger = getLogger("clinicadl.early_stopping")
T = TypeVar("T", DataPoint, Batch)
DataPointT = TypeVar("DataPointT", bound=DataPoint)


class SimpleInfererConfig(ObjectConfig["SimpleInferer"]):
    """Config class for ``SimpleInferer``."""

    postprocessing: Postprocessing = Field(reader=Postprocessing.from_dict)
    postprocessing_on_cpu: bool

    @classmethod
    def _get_class(cls):
        return SimpleInferer


class SimpleInferer(HasConfig[SimpleInfererConfig], Inferer):
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

    @overload
    def __call__(
        self,
        x: DataPointT,
        network: nn.Module,
        input_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> DataPointT:
        ...

    @overload
    def __call__(
        self,
        x: Batch[DataPointT],
        network: nn.Module,
        input_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> Batch[DataPointT]:
        ...

    def __call__(
        self,
        x: Union[DataPointT, Batch[DataPointT]],
        network: nn.Module,
        input_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> Union[DataPointT, Batch[DataPointT]]:
        tensor = self._get_input_tensor(x, input_dtype=input_dtype)

        output = network(tensor, **kwargs)

        self._add_output(x, output)

        if self.config.postprocessing_on_cpu and self.config.postprocessing.transforms:
            x.to(device="cpu")

        return self._postprocess(x)

    @classmethod
    def _add_output(cls, x: Union[DataPoint, Batch], output: torch.Tensor) -> None:
        """
        Adds the inference output in the origin data structure.
        """
        if isinstance(x, DataPoint):
            x[OUTPUT] = cls._format_output(x, output)
        elif isinstance(x, Batch):
            x.add_field(
                OUTPUT, [cls._format_output(x_, out_) for x_, out_ in zip(x, output)]
            )

    @staticmethod
    def _format_output(
        x: DataPoint, output: torch.Tensor
    ) -> Union[tio.Image, torch.Tensor]:
        """
        Formats the output, i.e. puts it in a :py:class:`torchio.Image`, or leaves it as
        a :py:class:`torch.Tensor`.
        """
        try:
            if x.label is None:
                return tio.ScalarImage(tensor=output, affine=x.image.affine)
            elif isinstance(x.label, tio.LabelMap):
                return tio.LabelMap(tensor=output, affine=x.label.affine)
        except Exception:
            logger.info(
                "The Inferer tried to wrap the neural network output in a torchio.Image, but an error occurred."
            )

        return output

    def _postprocess(self, x: DataPointT) -> DataPointT:
        """
        Applies postprocessing.
        """
        if isinstance(x, DataPoint):
            return self.config.postprocessing.apply(x)
        elif isinstance(x, Batch):
            return self.config.postprocessing.batch_apply(x)

    @classmethod
    def _from_config(cls, config):
        return cls(
            postprocessing=config.postprocessing.config.transforms.values,
            **config.to_raw_dict(exclude=["postprocessing"]),
        )  # not get_object here because we want to keep config classes as config classes
