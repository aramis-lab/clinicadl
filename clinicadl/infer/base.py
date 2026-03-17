from abc import abstractmethod
from enum import Enum
from logging import getLogger
from typing import Any, Callable, Optional, Sequence, TypeVar, Union, overload

import torch
import torch.nn as nn
import torchio as tio
from pydantic import Field

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.transforms.handlers import PostprocessingHandler
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import CPU
from clinicadl.utils.objects import HasConfig

from .abstract import Inferer

logger = getLogger(__name__)

T = TypeVar("T", DataPoint, Batch)
DataPointT = TypeVar("DataPointT", bound=DataPoint)


class OutputType(str, Enum):
    """Possible types of output."""

    IMAGE = "image"
    MASK = "mask"
    TENSOR = "tensor"


class BaseInfererConfig(ObjectConfig["BaseInferer"]):
    """Base config class for the inferers implemented in ``ClinicaDL``."""

    postprocessing: PostprocessingHandler = Field(
        reader=PostprocessingHandler.from_dict
    )
    postprocessing_on_cpu: bool
    output_name: str
    output_type: Optional[OutputType]


class BaseInferer(Inferer, HasConfig[BaseInfererConfig]):
    """Base class for the inferers implemented in ``ClinicaDL``."""

    def __init__(
        self,
        postprocessing: Optional[Sequence[TransformOrConfig]] = None,
        **kwargs,
    ):
        if not postprocessing:
            postprocessing = []
        postprocessing = PostprocessingHandler(postprocessing)
        self.config = self._config_type(
            postprocessing=postprocessing,
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
        network: Callable[..., torch.Tensor],
        input_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> Union[DataPoint, Batch]:
        tensor = self._get_input_tensor(x, input_dtype=input_dtype)

        output = self._forward_pass(tensor, network, **kwargs)

        self._add_output(x, output)

        if self.config.postprocessing_on_cpu and self.config.postprocessing.transforms:
            x.to(device=CPU)

        return self._postprocess(x)

    @abstractmethod
    def _forward_pass(
        self, tensor: torch.Tensor, network: Callable[..., torch.Tensor], **kwargs
    ) -> torch.Tensor:
        """
        Defines how a whole image is passed in the neural network.
        """

    def _add_output(self, x: Union[DataPoint, Batch], output: torch.Tensor) -> None:
        """
        Adds the inference output in the origin data structure.
        """
        if isinstance(x, DataPoint):
            self._add_output_in_datapoint(output, x)
        elif isinstance(x, Batch):
            for out_, x_ in zip(output, x):
                self._add_output_in_datapoint(out_, x_)

    def _add_output_in_datapoint(self, output: torch.Tensor, x: DataPoint) -> None:
        """
        Adds the output with the right format in the input DataPoint.
        """
        if self.config.output_type == OutputType.IMAGE or (
            self.config.output_type is None and x.label is None
        ):
            x.add_image(output, self.config.output_name)
        elif self.config.output_type == OutputType.MASK or (
            self.config.output_type is None and isinstance(x.label, tio.LabelMap)
        ):
            x.add_mask(output, self.config.output_name)
        else:
            x[self.config.output_name] = output

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
