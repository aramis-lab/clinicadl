from copy import deepcopy
from logging import getLogger
from typing import Union

import torch
import torchio as tio
from pydantic import field_serializer, model_validator

from clinicadl.data.dataloader import Batch
from clinicadl.data.structures import DataPoint
from clinicadl.utils.config import ClinicaDLConfig

from .config import TransformConfig
from .transforms import CUSTOM_TRANSFORM
from .types import Transform

logger = getLogger("clinicadl.transforms.transforms")


class OutputTransforms(ClinicaDLConfig):
    """
    A configuration class for applying transformations on the outputs of a network.

    Attributes
    ----------
    transforms : list[Union[Transform, TransformConfig]], (optional, default=[])
        A list of transformations to apply on the outputs.
    """

    transforms: list[Union[Transform, TransformConfig]] = []

    _transforms_processed: list[Transform] = []

    @model_validator(mode="after")
    def check_transforms(self):
        """
        Converts configuration classes to actual transforms.
        """
        self._transforms_processed = self._config_to_transform(self.transforms)

        return self

    @field_serializer("sample_transforms", check_fields=False)
    def serialize_transforms(
        self, transforms: list[Union[Transform, TransformConfig]]
    ) -> list[Union[str, dict]]:
        """
        Handles serialization of transforms that are not passed via
        TransformConfigs.
        """
        d = []
        for transform in transforms:
            if isinstance(transform, TransformConfig):
                d.append(transform.model_dump())
            else:
                d.append(CUSTOM_TRANSFORM + ": " + f"'{type(transform).__name__}'")

        return d

    @staticmethod
    def _config_to_transform(
        list_transforms: list[Union[Transform, TransformConfig]],
    ) -> list[Transform]:
        """
        Converts TransformConfig objects to transforms.
        """
        only_transforms = []
        for transform in list_transforms:
            if isinstance(transform, TransformConfig):
                real_transform = transform.get_object()
                only_transforms.append(real_transform)
            else:
                only_transforms.append(transform)

        return only_transforms

    def __str__(self) -> str:
        """
        Returns a detailed string representation of the `OutputTransforms` object.
        """
        # Start with a general description of the object
        str_ = "Output Transforms Configuration:\n"

        if self._transforms_processed:
            for transform in self._transforms_processed:
                str_ += f"  - {type(transform).__name__}\n"
        else:
            str_ += "No transform applied.\n"

        return str_

    def get_transforms(
        self,
    ) -> Transform:
        """
        Composes and returns the transformations.

        Returns
        -------
        Transform
            The composed transformations.
        """
        return tio.Compose(self._transforms_processed)

    def apply(
        self,
        data_point: DataPoint,
    ) -> DataPoint:
        output = self.get_transforms()(data_point)

        return output

    def batch_apply(
        self, batch_tensor: torch.Tensor, data: Batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the transformations to a batch of images and samples.

        Parameters
        ----------
        batch_tensor : torch.Tensor
            A batch of images.
        data : list[Sample]
            A batch of samples.

        Returns
        -------
        list[Sample]
            A batch of transformed samples.
        """
        transformed_outputs = []
        transformed_labels = []

        for i in range(batch_tensor.shape[0]):
            # output_sample = deepcopy(data[i])
            output_sample = batch_tensor[
                i
            ]  # Assuming batch_tensor has the same shape as the image in the samples

            output_datapoint = output_sample
            transformed_output_datapoint = self.apply(output_datapoint)

            transformed_outputs.append(transformed_output_datapoint.image)
            transformed_labels.append(transformed_output_datapoint.label)

            transformed_outputs_tensors = torch.cat(
                transformed_outputs, dim=0
            ).unsqueeze(1)
            transformed_labels_tensors = torch.tensor(
                transformed_labels, dtype=torch.float32
            ).unsqueeze(1)

        return (transformed_outputs_tensors, transformed_labels_tensors)
