from logging import getLogger
from typing import Tuple, Union

import torchio as tio
from pydantic import field_serializer, model_validator

from clinicadl.dictionary.words import AUGMENTATION, IMAGE, SAMPLE, TRANSFORMATION
from clinicadl.transforms.config.intensity import NanRemovalConfig
from clinicadl.transforms.extraction import Extraction, Image
from clinicadl.utils.config import ClinicaDLConfig

from .config import TransformConfig
from .factory import get_transform_from_config
from .types import Transform

logger = getLogger("clinicadl.transforms.transforms")

CUSTOM_TRANSFORM = "Custom transform passed by the user"


class Transforms(ClinicaDLConfig):
    """
    A configuration class for applying transformations and augmentations to dataset images and
    samples (slices and patches).

    This class manages the various transformations applied to images and their corresponding samples,
    including image preprocessing, sample transformation and data augmentation.

    Attributes
    ----------
    extraction : Extraction, (optional, default=Image())
        The extraction method used for preprocessing the data.
    image_transforms : list[Union[Transform, TransformConfig]], (optional, default=[NanRemovalConfig()])
        A list of transformations to apply on the whole image.
    sample_transforms : list[Union[Transform, TransformConfig]], (optional, default=[])
        A list of transformations to apply on samples (patches or slices).
    augmentations : list[Union[Transform, TransformConfig]], (optional, default=[])
        A list of augmentation transforms, to apply on samples, only during training.
    """

    extraction: Extraction = Image()
    image_transforms: list[Union[Transform, TransformConfig]] = [
        NanRemovalConfig(nan=0.0, posinf=None, neginf=None)
    ]
    sample_transforms: list[Union[Transform, TransformConfig]] = []
    augmentations: list[Union[Transform, TransformConfig]] = []
    _image_transforms_processed: list[Transform] = []
    _sample_transforms_processed: list[Transform] = []
    _augmentations_processed: list[Transform] = []

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
        if isinstance(self.extraction, Image) and self.sample_transforms:
            logger.warning(
                "You provided 'sample_transforms' but in the chosen configuration, image and sample are the same."
            )
            for trans in self.sample_transforms:
                self.image_transforms.append(trans)
            self.sample_transforms = []

        self._image_transforms_processed = self._config_to_transform(
            self.image_transforms
        )
        self._sample_transforms_processed = self._config_to_transform(
            self.sample_transforms
        )
        self._augmentations_processed = self._config_to_transform(self.augmentations)

        return self

    @field_serializer(
        "image_transforms",
        "sample_transforms",
        "augmentations",
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
                    str_ += f"  - {type(transform).__name__}\n"
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
            self._augmentations_processed, object_=SAMPLE, transfo_=AUGMENTATION
        )

        return transform_str

    def get_transforms(
        self,
    ) -> Tuple[Transform, Transform, Transform]:
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

        image_transforms = tio.Compose(self._image_transforms_processed)
        sample_transforms = tio.Compose(
            self._config_to_transform(self._sample_transforms_processed)
        )
        augmentations = tio.Compose(
            self._config_to_transform(self._augmentations_processed)
        )

        return (
            image_transforms,
            sample_transforms,
            augmentations,
        )
