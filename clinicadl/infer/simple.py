from logging import getLogger
from typing import Any, Optional, Sequence, TypeVar, Union, overload

import torch
import torch.nn as nn
import torchio as tio

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.transforms.handlers import Postprocessing
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.dictionary.words import OUTPUT

from .base import Inferer

logger = getLogger("clinicadl.early_stopping")
T = TypeVar("T", DataPoint, Batch)
DataPointT = TypeVar("DataPointT", bound=DataPoint)


class SimpleInferer(Inferer):
    """
    For classical inference, i.e. when the whole images are passed
    in the neural network and the raw outputs are returned (with a potential
    postprocessing).

    Parameters
    ----------

    Examples
    --------

    .. code-block::

        import torch
        from clinicadl.infer import SimpleInferer
        from clinicadl.data.structures.examples import ColinDataPoint
        from clinicadl.networks.nn import ConvEncoder
        from clinicadl.transforms.config import ActivationsConfig

        net = ConvEncoder(spatial_dims=3, in_channels=1, channels=[2, 4])
        datapoint = ColinDataPoint()

    .. code-block::

        inferer = SimpleInferer()
        with torch.no_grad():
            out = inferer(datapoint, net)

    """

    def __init__(
        self,
        postprocessing: Optional[Sequence[TransformOrConfig]] = None,
        postprocessing_on_cpu: bool = False,
    ):
        if not postprocessing:
            postprocessing = []
        self.postprocessing = Postprocessing(postprocessing)
        self.postprocessing_on_cpu = postprocessing_on_cpu

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
        tensor = self._get_input_tensor(x)

        output = network(tensor, **kwargs)

        self._add_output(x, output)

        if self.postprocessing_on_cpu and self.postprocessing:
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
            x.add_field(OUTPUT, [cls._format_output(x, out) for out in output])

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
            return self.postprocessing.apply(x)
        elif isinstance(x, Batch):
            return self.postprocessing.batch_apply(x)
