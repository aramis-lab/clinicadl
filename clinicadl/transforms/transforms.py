from logging import getLogger
from typing import Optional, Tuple, Union

import torchio as tio
from pydantic import field_serializer, field_validator, model_validator

from clinicadl.dictionary.words import AUGMENTATION, IMAGE, SAMPLE, TRANSFORMATION
from clinicadl.transforms.extraction import Extraction, Image
from clinicadl.transforms.zoo.config.factory import NanRemovalConfig
from clinicadl.utils.config import ClinicaDLConfig

from .config import TransformConfig
from .utils import Transform

logger = getLogger("clinicadl.transforms.transforms")

CUSTOM_TRANSFORM = "Custom transform passed by the user"


class Transforms(ClinicaDLConfig):
    """
    Configuration class to gather all the transforms applied to images.

    ClinicaDL defines 4 types of transforms:\n
    - ``extraction``: defines on what type of elements of the image we want to work
      (the whole image, patches or slices).
    - ``image_transforms``: transforms applied on the whole image, **before**
      potential extraction is applied. This is typically where you want to
      do normalization, to normalize on the whole image and not only on a patch
      or a slice.
    - ``sample_transforms``: transforms applied on a sample (a patch or a slice),
      **after** extraction. This is typically where you want to
      resize your sample so that it fits in your network.
    - ``augmentations``: transforms applied after ``image_transforms``, ``extraction``
      and ``sample_transforms``, only during training.

    .. note::
        :ref:`Extraction objects <extraction>` are not exactly transforms since
        they modify the size of the datasets: if you have 10 images with 100 slices each and you want to work on slices
        (so you passed ``extraction=Slice()``), the effective length of your dataset will be :math:`10\\times100=1,000`.

    For ``image_transforms``, ``sample_transforms`` and ``augmentations``, the transforms must be passed as lists.
    ``Transforms`` will compose the transforms in these lists, so **the order in the lists is important**.

    Finally, ``Transforms`` accepts preferably :ref:`transform configuration classes <supported_transforms>`, but also
    any custom transform created by the user (see :ref:`examples <examples>`). The only requirement is that this custom transforms
    works with :py:class:`DataPoint <clinicadl.data.structures.DataPoint>`. In line with :ref:`ClinicaDL's philosophy <api_introduction>`,
    you are encouraged to **use transform configuration classes for better reproducibility**.

    Parameters
    ----------
    extraction : Optional[Extraction], (optional, default=None)
        The extraction applied. See :ref:`extraction`. Default is ``None``, which means
        that no extraction is applied and that the :py:class:`CapsDataset <clinicadl.data.datasets.CapsDataset>`
        will output full images.
    image_transforms : list[Union[Transform, TransformConfig]], (optional, default=[NanRemovalConfig()])
        A list of transformations to apply on the whole image, before extraction.
    sample_transforms : list[Union[Transform, TransformConfig]], (optional, default=[])
        A list of transformations to apply on samples (patches or slices).

    .. note::
        If ``extraction=None``, ``image_transforms`` and ``sample_transforms`` are the same.
        They will therefore be merged in ``image_transforms``.

    augmentations : list[Union[Transform, TransformConfig]], (optional, default=[])
        A list of augmentation transforms, to apply on samples, only during training.

    .. _examples:

    Examples
    --------
    >>> from clinicadl.transforms import Transforms
    >>> from clinicadl.transforms.extraction import Patch
    >>> from clinicadl.transforms.config import ZNormalizationConfig, RandomFlipConfig
    >>> import torchio
    >>> Transforms(
            extraction=Patch(patch_size=32, stride=32),
            image_transforms=[ZNormalizationConfig(), torchio.CropOrPad(64)],  # torchio.CropOrPad is not a config class, so it is a custom transform
            sample_transforms=[],
            augmentations=[RandomFlipConfig(flip_probability=0.3)],
        )
    """

    extraction: Extraction = Image()
    image_transforms: list[Union[Transform, TransformConfig]] = [NanRemovalConfig()]
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
    @classmethod
    def serialize_transforms(
        cls, transforms: list[Union[Transform, TransformConfig]]
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
        sample_transforms = tio.Compose(self._sample_transforms_processed)
        augmentations = tio.Compose(self._augmentations_processed)

        return (
            image_transforms,
            sample_transforms,
            augmentations,
        )
