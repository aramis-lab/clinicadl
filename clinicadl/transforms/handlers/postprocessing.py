from typing import Union

from pydantic import field_serializer, model_validator

from clinicadl.data.structures import DataPoint

from ..types import Transform, TransformOrConfig
from .base import TransformsHandler


class Postprocessing(TransformsHandler):
    """
    A configuration class for applying transformations on the outputs of a network.

    Parameters
    ----------
    transforms : list[TransformOrConfig], default=[]
        A list of transformations to apply on the outputs.
    """

    transforms: list[TransformOrConfig]
    _transforms_processed: Transform

    @model_validator(mode="after")
    def _convert_transforms(self):
        """
        Converts the transform configs to actual transform objects.
        """
        super()._convert_transforms()
        return self

    @field_serializer("transforms")
    @classmethod
    def _serialize_transforms(
        cls, transforms: list[TransformOrConfig]
    ) -> list[Union[str, dict]]:
        """
        Handles serialization of transforms that are not passed via
        TransformConfigs.
        """
        return super()._serialize_transforms(transforms)

    def __str__(self) -> str:
        """
        Returns a detailed string representation of the ``Postprocessing`` object.
        """
        str_ = "Postprocessing configuration:\n"

        if self._transforms_processed:
            for transform in self._transforms_processed:
                str_ += f"  - {type(transform).__name__}\n"
        else:
            str_ += "No transform applied.\n"

        return str_

    def apply(self, datapoint: DataPoint) -> DataPoint:
        """
        Applies the transforms and returns the output.
        """
        return self._transforms_processed(datapoint)

    def batch_apply(self, batch: list[DataPoint]) -> list[DataPoint]:
        """
        Applies the transformations to a batch of
        :py:class:`~clinicadl.data.structures.DataPoint`.

        Parameters
        ----------
        batch : list[DataPoint]
            A batch of :py:class:`~clinicadl.data.structures.DataPoint`.

        Returns
        -------
        list[DataPoint]
            The transformed batch.
        """
        return [self.apply(datapoint) for datapoint in batch]
