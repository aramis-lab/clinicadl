from copy import deepcopy
from logging import getLogger
from typing import Optional, Tuple, Union

import torch
import torchio as tio
from pydantic import field_serializer, model_validator

from clinicadl.data.dataloader import BatchLoader
from clinicadl.data.structures import DataPoint
from clinicadl.dictionary.words import SAMPLE, TRANSFORMATION
from clinicadl.utils.config import ClinicaDLConfig

from .config import TransformConfig
from .factory import get_transform_from_config
from .types import Transform

logger = getLogger("clinicadl.transforms.transforms")

CUSTOM_TRANSFORM = "Custom transform passed by the user"


class OutputTransforms(ClinicaDLConfig):
    """
    A configuration class for applying transformations and augmentations to dataset images and
    samples (slices and patches).

    This class manages the various transformations applied to images and their corresponding samples,
    including image preprocessing, sample transformation and data augmentation.

    Attributes
    ----------
    sample_transforms : list[Union[Transform, TransformConfig]], (optional, default=[])
        A list of transformations to apply on samples (patches or slices).
    """

    sample_transforms: list[Union[Transform, TransformConfig]] = []

    _sample_transforms_processed: list[Transform] = []

    @model_validator(mode="after")
    def check_transforms(self):
        """
        Validates and adjusts the transformation configuration when image and sample transformations overlap.

        If the `extraction` is of type `Image` and sample transformations or augmentations are provided,
        they will be merged into the image transformations and augmentations. A warning is logged for
        potential configuration conflicts.

        Returns
        -------
        Transforms
            The updated `Transforms` object after ensuring the consistency of transformations.
        """

        self._sample_transforms_processed = self._config_to_transform(
            self.sample_transforms
        )

        return self

    @field_serializer(
        "sample_transforms",
    )
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
                real_transform, _ = get_transform_from_config(transform)
                only_transforms.append(real_transform)
            else:
                only_transforms.append(transform)

        return only_transforms

    def __str__(self) -> str:
        """
        Returns a detailed string representation of the `Transforms` object,
        showing the current configuration of image and sample transformations,
        augmentations, and other settings.

        Returns
        -------
        str
            A detailed string representation of the `Transforms` object.
        """
        # Start with a general description of the object
        transform_str = "Output Transforms Configuration:\n"

        def _to_str(
            list_: list[Transform],
            object_: str,
            transfo_: str,
        ):
            str_ = ""
            if list_:
                str_ += f"{object_} {transfo_}:\n"
                for transform in list_:
                    str_ += f"  - {type(transform).__name__}\n"
            else:
                str_ += f"No {object_} {transfo_} applied.\n"

            return str_

        transform_str += _to_str(
            self._sample_transforms_processed, object_=SAMPLE, transfo_=TRANSFORMATION
        )

        return transform_str

    def get_transforms(
        self,
    ) -> Transform:
        """
        Composes and returns the transformations and augmentations.

        Returns
        -------
        Tuple[Transform, Transform, Transform]
            A tuple containing:
            - The composed image transformations.
            - The composed sample transformations.
            - The composed sample augmentations.
        """
        logger.info(
            "Transforms will be applied in this order: image transforms, sample transforms, "
            " and augmentations (during training only)."
        )

        sample_transforms = tio.Compose(
            self._config_to_transform(self._sample_transforms_processed)
        )

        return sample_transforms

    def apply(
        self,
        data_point: DataPoint,
    ) -> DataPoint:
        output = self.get_transforms()(data_point)

        if data_point.image.shape != output.image.shape:
            raise ValueError(
                "One of the transform modifies the size of the tensor and can't be used"
            )

        return output

    def batch_apply(
        self, batch_tensor: torch.Tensor, data: BatchLoader
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
            output_sample = deepcopy(data[i])
            output_sample.sample = batch_tensor[
                i
            ]  # Assuming batch_tensor has the same shape as the image in the samples

            output_datapoint = output_sample.get_datapoint()
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
