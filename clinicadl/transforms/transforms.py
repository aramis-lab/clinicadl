from logging import getLogger
from typing import Tuple, Union

import torchio as tio
from pydantic import model_validator

from clinicadl.dictionary.words import AUGMENTATION, IMAGE, SAMPLE, TRANSFORMATION
from clinicadl.transforms.config.intensity import NanRemovalConfig
from clinicadl.transforms.extraction import Extraction, Image
from clinicadl.utils.config import ClinicaDLConfig

from .config import TransformConfig
from .factory import get_transform_from_config
from .utils import Transform

logger = getLogger("clinicadl.transforms.transforms")


class Transforms(ClinicaDLConfig):
    """
    A configuration class for applying transformations and augmentations to dataset images and
    samples (slices, patches or ROIs).

    This class manages the various transformations applied to images and their corresponding samples,
    including image preprocessing, sample transformation and data augmentation.

    Attributes
    ----------
    extraction : Extraction
        The extraction method used for preprocessing the data.
    image_transforms : list[Union[Transform, TransformConfig]]
        A list of transformation functions for images.
    sample_transforms : list[Union[Transform, TransformConfig]]
        A list of transformation functions for samples.
    image_augmentations : list[Union[Transform, TransformConfig]]
        A list of augmentation functions for images.
    sample_augmentations : list[Union[Transform, TransformConfig]]
        A list of augmentation functions for samples (e.g., masks or labels).

    Methods
    -------
    check_transforms()
        Validates and adjusts the configuration for transformations when images and samples are the same.
    __str__()
        Returns a string representation of the `Transforms` object.
    get_transforms()
        Returns a tuple of composed transformations for images, samples, and augmentations.
    """

    extraction: Extraction = Image()
    image_transforms: list[Union[Transform, TransformConfig]] = [NanRemovalConfig()]
    sample_transforms: list[Union[Transform, TransformConfig]] = []
    image_augmentations: list[Union[Transform, TransformConfig]] = []
    sample_augmentations: list[Union[Transform, TransformConfig]] = []
    _image_transforms_processed: list[Transform] = []
    _sample_transforms_processed: list[Transform] = []
    _image_augmentations_processed: list[Transform] = []
    _sample_augmentations_processed: list[Transform] = []

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
        if isinstance(self.extraction, Image):
            if self.sample_transforms != []:
                logger.warning(
                    "You provided sample_transforms but in the chosen configuration, image and sample are the same."
                )
                for trans in self.sample_transforms:
                    self.image_transforms.append(trans)
                self.sample_transforms = []

            if self.sample_augmentations:
                logger.warning(
                    "You provided sample_augmentations but in the chosen configuration, image and sample are the same."
                )
                for aug in self.sample_augmentations:
                    self.image_augmentations.append(aug)
                self.sample_augmentations = []

        self._image_transforms_processed = self._config_to_transform(
            self.image_transforms
        )
        self._sample_transforms_processed = self._config_to_transform(
            self.sample_transforms
        )
        self._image_augmentations_processed = self._config_to_transform(
            self.image_augmentations
        )
        self._sample_augmentations_processed = self._config_to_transform(
            self.sample_augmentations
        )

        return self

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
                only_transforms.append(get_transform_from_config(transform))
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
        transform_str = f"Transforms Configuration for {self.extraction} extraction:\n"

        def _to_str(
            list_: list[Transform],
            object_: str,
            transfo_: str,
        ):
            str_ = ""
            if list_:
                str_ += f"{object_} {transfo_}:\n"
                for transform in list_:
                    str_ += f"  - {transform.__class__.__name__}\n"
            else:
                str_ += f"No {object_} {transfo_} applied.\n"

            return str_

        transform_str += _to_str(
            self._image_transforms_processed, object_=IMAGE, transfo_=TRANSFORMATION
        )
        transform_str += _to_str(
            self._sample_transforms_processed, object_=SAMPLE, transfo_=TRANSFORMATION
        )
        transform_str += _to_str(
            self._image_augmentations_processed, object_=IMAGE, transfo_=AUGMENTATION
        )
        transform_str += _to_str(
            self._sample_augmentations_processed, object_=SAMPLE, transfo_=AUGMENTATION
        )

        return transform_str

    def get_transforms(
        self,
    ) -> Tuple[Transform, Transform, Transform, Transform]:
        """
        Composes and returns the transformations and augmentations for images and samples.

        Returns
        -------
        Tuple[tio.Compose, tio.Compose, tio.Compose, tio.Compose]
            A tuple containing:
            - The composed image transformations.
            - The composed sample transformations.
            - The composed image augmentations.
            - The composed sample augmentations.
        """
        logger.info(
            "Transforms will be applied in this order: image transforms, image augmentations (during training only), sample transforms, "
            " and sample augmentations (during training only)."
        )

        image_transforms = tio.Compose(self._image_transforms_processed)
        sample_transforms = tio.Compose(
            self._config_to_transform(self._sample_transforms_processed)
        )
        image_augmentations = tio.Compose(
            self._config_to_transform(self._image_augmentations_processed)
        )
        sample_augmentations = tio.Compose(
            self._config_to_transform(self._sample_augmentations_processed)
        )

        return (
            image_transforms,
            sample_transforms,
            image_augmentations,
            sample_augmentations,
        )
