from logging import getLogger
from typing import Callable, Optional, Tuple

import torchvision.transforms as torch_transforms
from pydantic import model_validator

from clinicadl.transforms.extraction import BaseExtraction, Image
from clinicadl.transforms.factory import (
    MinMaxNormalization,
    NanRemoval,
    SizeReduction,
)
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import SizeReductionFactor

logger = getLogger("clinicadl.transforms.transforms")


type_ = Optional[torch_transforms.Compose]


class Transforms(ClinicaDLConfig):
    """
    A configuration class for applying transformations and augmentations to dataset images and objects.

    This class manages the various transformations applied to images and their corresponding objects,
    including image preprocessing, object transformation, data augmentation, and size reduction.

    Attributes
    ----------
    extraction : BaseExtraction
        The extraction method used for preprocessing the data.
    image_augmentation : list[Callable]
        A list of augmentation functions for images.
    object_augmentation : list[Callable]
        A list of augmentation functions for objects (e.g., masks or labels).
    image_transforms : list[Callable]
        A list of transformation functions for images.
    object_transforms : list[Callable]
        A list of transformation functions for objects.
    size_reduction : bool
        Flag indicating whether to apply size reduction to the images.
    size_reduction_factor : SizeReductionFactor
        Factor by which to reduce the size of the images (e.g., 2x, 4x).
    normalize : bool
        Flag indicating whether to apply normalization to the images.

    Methods
    -------
    check_transforms()
        Validates and adjusts the configuration for transformations when images and objects are the same.
    __str__()
        Returns a string representation of the `Transforms` object.
    get_transforms(normalize: bool, size_reduction: bool, size_reduction_factor: int)
        Returns a tuple of composed transformations for images, objects, and augmentations.
    """

    extraction: BaseExtraction
    image_augmentation: list[Callable] = []
    object_augmentation: list[Callable] = []
    image_transforms: list[Callable] = []
    object_transforms: list[Callable] = []

    # don't know if we keep these 3 values
    size_reduction: bool = False
    size_reduction_factor: SizeReductionFactor = SizeReductionFactor.TWO
    normalize: bool = True

    @model_validator(mode="after")
    def check_transforms(self):
        """
        Validates and adjusts the transformation configuration when image and object transformations overlap.

        If the `extraction` is of type `Image` and object transformations or augmentations are provided,
        they will be merged into the image transformations and augmentations. A warning is logged for
        potential configuration conflicts.

        Returns
        -------
        Transforms
            The updated `Transforms` object after ensuring the consistency of transformations.
        """
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

    def __str__(self) -> str:
        """
        Returns a detailed string representation of the `Transforms` object,
        showing the current configuration of image and object transformations,
        augmentations, and other settings.

        Returns
        -------
        str
            A detailed string representation of the `Transforms` object.
        """
        # Start with a general description of the object
        transform_str = f"Transforms Configuration for {self.extraction} extraction:\n"

        def _to_str(
            list_: list[Callable] = [],
            object_: str = "object",
            transfo_: str = "transformation",
        ):
            str_ = ""
            if list_:
                str_ += f"{object_} {transfo_}:\n"
                for transform in self.image_transforms:
                    str_ += f"  - {transform.__class__.__name__}\n"
            else:
                str_ += f"No {object_} {transfo_} applied.\n"

            return str_

        transform_str += _to_str(self.image_transforms, object_="image")
        transform_str += _to_str(self.object_transforms, object_="object")
        transform_str += _to_str(
            self.image_augmentation, object_="image", transfo_="augmentation"
        )
        transform_str += _to_str(
            self.object_augmentation, object_="object", transfo_="augmentation"
        )

        return transform_str

    def get_transforms(
        self,
        normalize: bool = True,
        size_reduction: bool = False,
        size_reduction_factor: int = 2,
    ) -> Tuple[torch_transforms.Compose, type_, type_, type_]:
        """
        Composes and returns the transformations and augmentations for images and objects.

        This method applies the following transformations in order:
        1. Image transformations (e.g., Nan removal, normalization, size reduction).
        2. Object transformations (if applicable).
        3. Data augmentation for both images and objects (if applicable).

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize the images (default is True).
        size_reduction : bool, optional
            Whether to apply size reduction (default is False).
        size_reduction_factor : int, optional
            The factor by which to reduce the image size (default is 2).

        Returns
        -------
        Tuple[torch_transforms.Compose, torch_transforms.Compose, torch_transforms.Compose, torch_transforms.Compose]
            A tuple containing:
            - The composed image transformations.
            - The composed object transformations (or None if not applicable).
            - The composed image augmentations (or None if not provided).
            - The composed object augmentations (or None if not provided).
        """
        logger.info(
            "Transforms will be applied in this order: image transforms, object transforms, and then data augmentation during training."
        )

        # Apply Nan removal and optional normalization
        self.image_transforms.append(NanRemoval())
        if normalize:
            self.image_transforms.append(MinMaxNormalization())

        # Apply size reduction if requested
        if size_reduction:
            self.image_transforms.append(
                SizeReduction(size_reduction_factor=size_reduction_factor)
            )

        # Compose image transformations
        image_transforms = torch_transforms.Compose(self.image_transforms)

        # Compose object transformations (if any)
        object_transforms = (
            torch_transforms.Compose(self.object_transforms)
            if self.object_transforms
            else None
        )

        # Compose image augmentations (if any)
        image_augmentation = (
            torch_transforms.Compose(self.image_augmentation)
            if self.image_augmentation
            else None
        )

        # Compose object augmentations (if any)
        object_augmentation = (
            torch_transforms.Compose(self.object_augmentation)
            if self.object_augmentation
            else None
        )

        return (
            image_transforms,
            object_transforms,
            image_augmentation,
            object_augmentation,
        )
