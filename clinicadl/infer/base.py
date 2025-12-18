from abc import abstractmethod
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

from .abstract import Inferer

logger = getLogger("clinicadl.infer.base")

T = TypeVar("T", DataPoint, Batch)
DataPointT = TypeVar("DataPointT", bound=DataPoint)


class BaseInfererConfig(ObjectConfig["BaseInferer"]):
    """Base config class for the inferers implemented in ``ClinicaDL``."""

    postprocessing: Postprocessing = Field(reader=Postprocessing.from_dict)
    postprocessing_on_cpu: bool


class BaseInferer(Inferer, HasConfig[BaseInfererConfig]):
    """Base class for the inferers implemented in ``ClinicaDL``."""

    def __init__(
        self,
        postprocessing: Optional[Sequence[TransformOrConfig]] = None,
        postprocessing_on_cpu: bool = False,
        **kwargs,
    ):
        if not postprocessing:
            postprocessing = []
        postprocessing = Postprocessing(postprocessing)
        self.config = self._config_type(
            postprocessing=postprocessing,
            postprocessing_on_cpu=postprocessing_on_cpu,
            **kwargs,
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
        x: Union[DataPoint, Batch],
        network: nn.Module,
        input_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> Union[DataPoint, Batch]:
        tensor = self._get_input_tensor(x, input_dtype=input_dtype)

        output = self._forward_pass(tensor, network, **kwargs)

        self._add_output(x, output)

        if self.config.postprocessing_on_cpu and self.config.postprocessing.transforms:
            x.to(device="cpu")

        return self._postprocess(x)

    @abstractmethod
    def _forward_pass(
        self, tensor: torch.Tensor, network: nn.Module, **kwargs
    ) -> torch.Tensor:
        """
        Defines how a whole image is passed in the neural network.
        """

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

    @classmethod
    def _format_output(
        cls, x: DataPoint, output: torch.Tensor
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
                f"{cls.__name__} tried to wrap the neural network output in a torchio.Image, but an error occurred."
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
