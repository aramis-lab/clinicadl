from logging import getLogger
from typing import Any, Callable, List

import torchvision.transforms as torch_transforms
from pydantic import field_validator, model_validator

from clinicadl.dataset.transforms.extraction import (
    BaseExtraction,
    Image,
)
from clinicadl.dataset.transforms.factory import (
    MinMaxNormalization,
    NanRemoval,
    SizeReduction,
)
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import (
    ExtractionMethod,
    SizeReductionFactor,
    Transform,
)

logger = getLogger("clinicadl.transforms.transforms")


class Transforms(ClinicaDLConfig):
    extraction: BaseExtraction
    image_augmentation: list[Callable] = []
    object_augmentation: list[Callable] = []
    image_transforms: list[Callable] = []
    object_transforms: list[Callable] = []
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = SizeReductionFactor.TWO
    normalize: bool = True

    @model_validator(mode="after")
    def check_transforms(self):
        if isinstance(self.extraction, Image):
            if self.object_transforms:
                logger.warning(
                    "You provided object_transforms but in the chosen configuration, image and object are the same."
                )
                for trans in self.object_transforms:
                    self.image_transforms.append(trans)
                self.object_transforms = []

            if self.object_augmentation:
                logger.warning(
                    "You provided object_augmentation but in the chosen configuration, image and object are the same."
                )
                for aug in self.object_augmentation:
                    self.image_augmentation.append(aug)
                self.object_augmentation = []

        return self

    def __str__(self):
        return "transforms"

    def get_transforms(
        self,
        normalize: bool = True,
        size_reduction: bool = False,
        size_reduction_factor: int = 2,
    ):
        logger.info(
            "transforms will be apply in this order: image transforms, object transforms and then data augmentation during training."
        )

        self.image_transforms.append(NanRemoval())
        if normalize:
            self.image_transforms.append((MinMaxNormalization()))
        if size_reduction:
            self.image_transforms.append(
                SizeReduction(size_reduction_factor=size_reduction_factor)
            )
        image_transforms = torch_transforms.Compose(self.image_transforms)

        if self.object_transforms:
            object_transforms = torch_transforms.Compose(self.object_transforms)
        else:
            object_transforms = None

        if self.image_augmentation:
            image_augmentation = torch_transforms.Compose(self.image_augmentation)
        else:
            image_augmentation = None

        if self.object_augmentation:
            object_augmentation = torch_transforms.Compose(self.object_augmentation)
        else:
            object_augmentation = None

        return (
            image_transforms,
            object_transforms,
            image_augmentation,
            object_augmentation,
        )
