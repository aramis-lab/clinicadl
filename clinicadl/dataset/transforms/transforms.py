from logging import getLogger
from typing import Any, Callable, List, Optional

import torch
import torchio.transforms as transforms
import torchvision.transforms as torch_transforms
from pydantic import BaseModel, field_validator, model_validator

from clinicadl.dataset.config.extraction import (
    ALL_EXTRACTION_TYPES,
    ExtractionConfig,
    ExtractionImageConfig,
)
from clinicadl.dataset.transforms.factory import (
    MinMaxNormalization,
    NanRemoval,
    SizeReduction,
)
from clinicadl.utils.enum import (
    ExtractionMethod,
    SizeReductionFactor,
    Transform,
)

logger = getLogger("clinicadl.transforms.transforms")


class Transforms(BaseModel):
    extraction: ALL_EXTRACTION_TYPES
    image_augmentation: list[Callable] = []
    object_augmentation: list[Callable] = []
    image_transforms: list[Callable] = []
    object_transforms: list[Callable] = []
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = SizeReductionFactor.TWO
    normalize: bool = True

    @model_validator(mode="after")
    def check_transforms(self):
        if isinstance(self.extraction, ExtractionConfig):
            raise ValueError(
                "You need to provide a type of ExtractionConfig (Image, Patch, Roi or Slice). You can't just pass an ExtractionConfig."
            )

        elif isinstance(self.extraction, ExtractionImageConfig):
            if self.object_transforms:
                logger.warning(
                    "You provided object_transforms but in the chosen configuration, image and object are the same."
                )
                self.image_transforms.append(self.object_transforms)
                self.object_transforms = []

            if self.object_augmentation:
                logger.warning(
                    "You provided object_augmentation but in the chosen configuration, image and object are the same."
                )
                self.image_augmentation.append(self.object_augmentation)
                self.object_augmentation = []

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
            self.image_transforms.append((MinMaxNormalization))
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
