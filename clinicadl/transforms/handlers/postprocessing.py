from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, TypeVar

import torchio as tio
from pydantic import Field, ValidationInfo, field_validator

from clinicadl.transforms.config import TransformConfig
from clinicadl.utils.config import ObjectConfig, SequenceOfObjects
from clinicadl.utils.objects import HasConfig

from ..factory import get_transform_from_dict
from ..types import Transform, TransformOrConfig
from .utils import get_transform_name

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch
    from clinicadl.data.structures import DataPoint

DataPointT = TypeVar("DataPointT", bound="DataPoint")


class PostprocessingHandlerConfig(ObjectConfig["PostprocessingHandler"]):
    """Config class for ``PostprocessingHandler``."""

    transforms: SequenceOfObjects[Transform, TransformConfig] = Field(
        json_schema_extra={
            "reader": SequenceOfObjects.build_reader(get_transform_from_dict)
        }
    )

    @field_validator("transforms", mode="before")
    @classmethod
    def _handle_sequence(cls, v: Any, info: ValidationInfo) -> SequenceOfObjects:
        return SequenceOfObjects.from_sequence(v, field_name=info.field_name)

    @classmethod
    def _get_class(cls) -> type[PostprocessingHandler]:
        """Returns the class associated to this config class."""
        return PostprocessingHandler


class PostprocessingHandler(HasConfig[PostprocessingHandlerConfig]):
    """
    A configuration class for applying transformations on the outputs of a network.

    Parameters
    ----------
    transforms : list[TransformOrConfig], default=()
        A list of transformations to apply on the outputs.
    """

    _config_type = PostprocessingHandlerConfig

    def __init__(
        self,
        transforms: Sequence[TransformOrConfig] = (),
    ):
        self.config = PostprocessingHandlerConfig(
            transforms=transforms,
        )
        self.transforms = tio.Compose(
            self.config.transforms.get_object(), copy=False
        )  # copy is specified in the transforms

    def __str__(self) -> str:
        """
        Returns a detailed string representation of the ``PostprocessingHandler`` object.
        """
        str_ = "PostprocessingHandler:\n"

        if self.transforms.transforms:
            for transform in self.transforms.transforms:
                str_ += f"  - {get_transform_name(transform)}\n"
        else:
            str_ += "No transform applied.\n"

        return str_

    def apply(self, datapoint: DataPointT) -> DataPointT:
        """
        Applies the transforms and returns the output.

        Parameters
        ----------
        datapoint : DataPoint
            A :py:class:`~clinicadl.data.structures.DataPoint`.

        Returns
        -------
        DataPoint
            The transformed ``DataPoint``.
        """
        return self.transforms(datapoint)

    def batch_apply(self, batch: Batch[DataPointT]) -> Batch[DataPointT]:
        """
        Applies the transformations to a batch of
        :py:class:`~clinicadl.data.structures.DataPoint`.

        Parameters
        ----------
        batch : Batch
            A batch of :py:class:`~clinicadl.data.structures.DataPoint`,
            passed via a :py:class:`~clinicadl.data.dataloader.Batch`.

        Returns
        -------
        Batch
            The transformed batch.
        """
        for i, datapoint in enumerate(batch):
            batch[i] = self.apply(datapoint)

        return batch
