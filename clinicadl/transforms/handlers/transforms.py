from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any, Sequence, TypeVar

import torchio as tio
from pydantic import Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from clinicadl.transforms.config import TransformConfig
from clinicadl.transforms.extraction import Extraction, Image
from clinicadl.utils.config import ObjectConfig, SequenceOfObjects
from clinicadl.utils.dictionary.words import AUGMENTATION, IMAGE, SAMPLE, TRANSFORMATION
from clinicadl.utils.objects import HasConfig

from ..extraction import get_extraction_from_dict
from ..factory import get_transform_from_dict
from ..types import Transform, TransformOrConfig
from .utils import get_transform_name

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint

logger = getLogger("clinicadl.transforms.Transforms")

DataPointT = TypeVar("DataPointT", bound="DataPoint")


class TransformsConfig(ObjectConfig["Transforms"]):
    """Config class for ``Transforms``."""

    extraction: Extraction = Field(reader=get_extraction_from_dict)
    image_transforms: SequenceOfObjects[Transform, TransformConfig] = Field(
        reader=SequenceOfObjects.build_reader(get_transform_from_dict)
    )
    sample_transforms: SequenceOfObjects[Transform, TransformConfig] = Field(
        reader=SequenceOfObjects.build_reader(get_transform_from_dict)
    )
    augmentations: SequenceOfObjects[Transform, TransformConfig] = Field(
        reader=SequenceOfObjects.build_reader(get_transform_from_dict)
    )

    @field_validator(
        "image_transforms", "sample_transforms", "augmentations", mode="before"
    )
    @classmethod
    def _handle_sequence(cls, v: Any, info: ValidationInfo) -> SequenceOfObjects:
        return SequenceOfObjects.from_sequence(v, field_name=info.field_name)

    @model_validator(mode="after")
    def _check_transforms(self):
        """
        If the `extraction` is of type `Image` and sample transforms or augmentations are provided,
        they will be merged into the image transforms and augmentations. A warning is logged for
        potential configuration conflicts.

        Also converts the transform configs to actual transform objects.
        """
        if isinstance(self.extraction, Image) and self.sample_transforms.values:
            logger.warning(
                "You provided 'sample_transforms' but in the chosen configuration, image and sample are the same."
            )
            image_transforms = self.image_transforms.values
            image_transforms.extend(self.sample_transforms.values)
            self.__dict__["image_transforms"] = SequenceOfObjects(image_transforms)
            self.__dict__["sample_transforms"] = SequenceOfObjects([])

        return self

    @classmethod
    def _get_class(cls) -> type[Transforms]:
        """Returns the class associated to this config class."""
        return Transforms


class Transforms(HasConfig[TransformsConfig]):
    """
    Configuration class to define all the transforms applied to images in
    a :py:mod:`dataset <clinicadl.data.datasets>` (extraction, preprocessing, and augmentation).

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

    For ``image_transforms``, ``sample_transforms`` and ``augmentations``, the transforms must be passed as sequences.
    ``Transforms`` will compose the transforms in these sequences, so **the order in the sequences is important**.

    Finally, ``Transforms`` accepts preferably configuration classes (see :py:mod:`clinicadl.transforms.config`), but also
    any custom transform created by the user (see examples). The only requirement is that this custom transform
    is a callable that takes as input and returns a :py:class:`~clinicadl.data.structures.DataPoint`.

    Parameters
    ----------
    extraction : Extraction, default=Image()
        The extraction applied. See :py:mod:`clinicadl.transforms.extraction`. Default is
        that no extraction is applied, and thus the :py:mod:`dataset <clinicadl.data.datasets>`
        will output full images.
    image_transforms : Sequence[TransformOrConfig], default=[]
        A sequence of transforms to apply on the whole image, **before extraction**.
        Passed as configuration classes from :py:mod:`clinicadl.transforms.config`, or
        as custom transforms.
    sample_transforms : Sequence[TransformOrConfig], default=[]
        A sequence of transforms to apply on samples (patches or slices).
        Passed as configuration classes from :py:mod:`clinicadl.transforms.config`, or
        as custom transforms.

        .. note::
            If ``extraction=Image()``, ``image_transforms`` and ``sample_transforms`` are the same.
            They will therefore be merged in ``image_transforms``.

    augmentations : Sequence[TransformOrConfig], default=[]
        A sequence of augmentation transforms, to apply on samples, only during training.
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

    _config_type = TransformsConfig

    def __init__(
        self,
        extraction: Extraction = Image(),
        image_transforms: Sequence[TransformOrConfig] = [],
        sample_transforms: Sequence[TransformOrConfig] = [],
        augmentations: Sequence[TransformOrConfig] = [],
    ):
        self.config = TransformsConfig(
            extraction=extraction,
            image_transforms=image_transforms,
            sample_transforms=sample_transforms,
            augmentations=augmentations,
        )
        self.extraction = self.config.extraction
        self.image_transforms = tio.Compose(
            self.config.image_transforms.get_object(), copy=False
        )  # copy is specified in the transforms
        self.sample_transforms = tio.Compose(
            self.config.sample_transforms.get_object(), copy=False
        )
        self.augmentations = tio.Compose(
            self.config.augmentations.get_object(), copy=False
        )

    def __str__(self) -> str:
        """
        Returns a detailed string representation of the ``Transforms`` object,
        showing the current configuration of image and sample transforms,
        augmentations, and other settings.
        """
        transform_str = (
            f"Transforms configuration for {self.extraction.sample_type} extraction:\n"
        )

        def _to_str(
            list_: list[Transform],
            object_: str,
            transfo_: str,
        ):
            str_ = ""
            if list_:
                str_ += f"* {object_} {transfo_}:\n"
                for transform in list_:
                    str_ += f"  - {get_transform_name(transform)}\n"
            else:
                str_ += f"* No {object_} {transfo_} applied.\n"

            return str_

        transform_str += _to_str(
            self.image_transforms.transforms,
            object_=IMAGE,
            transfo_=TRANSFORMATION,
        )
        transform_str += _to_str(
            self.sample_transforms.transforms,
            object_=SAMPLE,
            transfo_=TRANSFORMATION,
        )
        transform_str += _to_str(
            self.augmentations.transforms,
            object_=SAMPLE,
            transfo_=AUGMENTATION,
        )

        return transform_str

    def apply_image_transforms(self, datapoint: DataPointT) -> DataPointT:
        """
        Applies the transforms passed in ``image_transforms`` and returns the
        output.

        Parameters
        ----------
        datapoint : DataPoint
            A :py:class:`~clinicadl.data.structures.DataPoint`.

        Returns
        -------
        DataPoint
            The transformed ``DataPoint``.
        """
        return self.image_transforms(datapoint)

    def extract_sample(self, datapoint: DataPointT, sample_index: int) -> DataPointT:
        """
        Extracts the sample.

        See: :py:class:`clinicadl.transforms.extraction.Extraction`.

        Parameters
        ----------
        datapoint : DataPoint
            A :py:class:`~clinicadl.data.structures.DataPoint`.

        Returns
        -------
        DataPoint
            The sample in a ``DataPoint``.
        """
        return self.extraction(datapoint, sample_index)

    def apply_sample_transforms(self, datapoint: DataPointT) -> DataPointT:
        """
        Applies the transforms passed in ``sample_transforms`` and returns the
        output.

        Parameters
        ----------
        datapoint : DataPoint
            A :py:class:`~clinicadl.data.structures.DataPoint`.

        Returns
        -------
        DataPoint
            The transformed ``DataPoint``.
        """
        return self.sample_transforms(datapoint)

    def apply_augmentations(self, datapoint: DataPointT) -> DataPointT:
        """
        Applies the transforms passed in ``augmentations`` and returns the
        output.

        Parameters
        ----------
        datapoint : DataPoint
            A :py:class:`~clinicadl.data.structures.DataPoint`.

        Returns
        -------
        DataPoint
            The transformed ``DataPoint``.
        """
        return self.augmentations(datapoint)
