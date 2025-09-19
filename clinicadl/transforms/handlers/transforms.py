from logging import getLogger
from typing import Union

import torchio as tio
from pydantic import field_serializer, model_validator

from clinicadl.data.structures import DataPoint
from clinicadl.dictionary.words import AUGMENTATION, IMAGE, SAMPLE, TRANSFORMATION
from clinicadl.transforms.extraction import Extraction, Image

from ..types import Transform, TransformOrConfig
from .base import TransformsHandler

logger = getLogger("clinicadl.transforms.Transforms")


class Transforms(TransformsHandler):
    """
    Configuration class to define all the transforms applied to images in
    a :py:class:`~clinicadl.data.datasets.CapsDataset` (extraction, preprocessing, and augmentation).

    ``ClinicaDL`` defines 4 types of transforms:\n
    - ``extraction``: defines on what type of elements of the image we want to work
      (the whole image, patches or slices).
    - ``image_transforms``: transforms applied on the whole image, **before
      potential extraction** is applied. This is typically where you want to
      do normalization (to normalize on the whole image and not only on a patch
      or a slice).
    - ``sample_transforms``: transforms applied on a sample (a patch or a slice),
      **after extraction**. This is typically where you want to
      resize your sample so that it fits in your network.
    - ``augmentations``: transforms applied after ``image_transforms``, ``extraction``
      and ``sample_transforms``, only during training.

    .. note::
        :py:mod:`Extraction objects <clinicadl.transforms.extraction>` are not exactly transforms since
        they modify the size of the datasets: if you have 10 images with 100 slices each and you want to work on slices
        (so you passed ``extraction=Slice()``), the effective length of your dataset will be :math:`10\\times100=1,000`.

    For ``image_transforms``, ``sample_transforms`` and ``augmentations``, the transforms must be passed as lists.
    ``Transforms`` will compose the transforms in these lists, so **the order in the lists is important**.

    Finally, ``Transforms`` accepts preferably configuration classes (see :py:mod:`clinicadl.transforms.config`), but also
    any custom transform created by the user (see examples). The only requirement is that this custom transform
    is a callable that takes as input and returns a :py:class:`~clinicadl.data.structures.DataPoint`.

    Parameters
    ----------
    extraction : Extraction, default=Image()
        The extraction applied. See :py:mod:`clinicadl.transforms.extraction`. Default is
        that no extraction is applied, and thus the :py:class:`CapsDataset <clinicadl.data.datasets.CapsDataset>`
        will output full images.
    image_transforms : list[TransformOrConfig], default=[]
        A list of transforms to apply on the whole image, **before extraction**.
        Passed as configuration classes from :py:mod:`clinicadl.transforms.config`, or
        as custom transforms.
    sample_transforms : list[TransformOrConfig], default=[]
        A list of transforms to apply on samples (patches or slices).
        Passed as configuration classes from :py:mod:`clinicadl.transforms.config`, or
        as custom transforms.

        .. note::
            If ``extraction=Image()``, ``image_transforms`` and ``sample_transforms`` are the same.
            They will therefore be merged in ``image_transforms``.

    augmentations : list[TransformOrConfig], default=[]
        A list of augmentation transforms, to apply on samples, only during training.
        Passed as configuration classes from :py:mod:`clinicadl.transforms.config`, or
        as custom transforms.

    Examples
    --------
    .. code-block:: python

        >>> from clinicadl.transforms import Transforms
        >>> from clinicadl.transforms.extraction import Patch
        >>> from clinicadl.transforms.config import ZNormalizationConfig, RandomFlipConfig
        >>> import torchio
        >>> transforms = Transforms(
                extraction=Patch(patch_size=32, stride=32),
                image_transforms=[ZNormalizationConfig(), torchio.CropOrPad(64)],  # torchio.CropOrPad is not a config class, so it is a custom transform
                sample_transforms=[],
                augmentations=[RandomFlipConfig(flip_probability=0.3)],
            )

    """

    extraction: Extraction = Image()
    image_transforms: list[TransformOrConfig] = []
    sample_transforms: list[TransformOrConfig] = []
    augmentations: list[TransformOrConfig] = []
    _image_transforms_processed: tio.Compose = tio.Compose([])
    _sample_transforms_processed: tio.Compose = tio.Compose([])
    _augmentations_processed: tio.Compose = tio.Compose([])

    @model_validator(mode="after")
    def _check_transforms(self):
        """
        If the `extraction` is of type `Image` and sample transforms or augmentations are provided,
        they will be merged into the image transforms and augmentations. A warning is logged for
        potential configuration conflicts.

        Also converts the transform configs to actual transform objects.
        """
        if isinstance(self.extraction, Image) and self.sample_transforms:
            logger.warning(
                "You provided 'sample_transforms' but in the chosen configuration, image and sample are the same."
            )
            for trans in self.sample_transforms:
                self.image_transforms.append(trans)
            self.sample_transforms = []

        self._convert_transforms()

        return self

    @field_serializer(
        "image_transforms",
        "sample_transforms",
        "augmentations",
    )
    @classmethod
    def serialize_transforms(
        cls, transforms: list[TransformOrConfig]
    ) -> list[Union[str, dict]]:
        """
        Handles serialization of transforms that are not passed via
        TransformConfigs.
        """
        return super()._serialize_transforms(transforms)

    def __str__(self) -> str:
        """
        Returns a detailed string representation of the ``Transforms`` object,
        showing the current configuration of image and sample transforms,
        augmentations, and other settings.
        """
        transform_str = f"Transforms configuration for {self.extraction.extract_method} extraction:\n"

        def _to_str(
            list_: list[Transform],
            object_: str,
            transfo_: str,
        ):
            str_ = ""
            if list_:
                str_ += f"* {object_} {transfo_}:\n"
                for transform in list_:
                    str_ += f"  - {self._get_transform_name(transform)}\n"
            else:
                str_ += f"* No {object_} {transfo_} applied.\n"

            return str_

        transform_str += _to_str(
            self._image_transforms_processed.transforms,
            object_=IMAGE,
            transfo_=TRANSFORMATION,
        )
        transform_str += _to_str(
            self._sample_transforms_processed.transforms,
            object_=SAMPLE,
            transfo_=TRANSFORMATION,
        )
        transform_str += _to_str(
            self._augmentations_processed.transforms,
            object_=SAMPLE,
            transfo_=AUGMENTATION,
        )

        return transform_str

    def apply_image_transforms(self, datapoint: DataPoint) -> DataPoint:
        """
        Applies the transforms passed in ``image_transforms`` and returns the
        output.
        """
        return self._image_transforms_processed(datapoint)

    def extract_sample(self, datapoint: DataPoint, sample_index: int) -> DataPoint:
        """
        Extracts the sample.

        See: :py:class:`clinicadl.transforms.extraction.Extraction`.
        """
        return self.extraction.extract_sample(datapoint, sample_index)

    def apply_sample_transforms(self, datapoint: DataPoint) -> DataPoint:
        """
        Applies the transforms passed in ``sample_transforms`` and returns the
        output.
        """
        return self._sample_transforms_processed(datapoint)

    def apply_augmentations(self, datapoint: DataPoint) -> DataPoint:
        """
        Applies the transforms passed in ``augmentations`` and returns the
        output.
        """
        return self._augmentations_processed(datapoint)
