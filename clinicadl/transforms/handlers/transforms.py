from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any, Optional, Sequence, TypeVar

import torchio as tio
from pydantic import Field, ValidationInfo, field_validator

from clinicadl.transforms.config import TransformConfig
from clinicadl.transforms.extraction import Extraction, Image
from clinicadl.utils.config import ObjectConfig, SequenceOfObjects
from clinicadl.utils.dictionary.words import AUGMENTATION, IMAGE, SAMPLE, TRANSFORMATION
from clinicadl.utils.objects import HasConfig, equal_if_config_equal

from ..extraction import get_extraction_from_dict
from ..factory import get_transform_from_dict
from ..types import Transform, TransformOrConfig
from .utils import get_transform_name

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint

logger = getLogger(__name__)

DataPointT = TypeVar("DataPointT", bound="DataPoint")


class TransformsHandlerConfig(ObjectConfig["TransformsHandler"]):
    """Config class for ``TransformsHandler``."""

    extraction: Extraction = Field(
        json_schema_extra={"reader": get_extraction_from_dict}
    )
    image_transforms: SequenceOfObjects[Transform, TransformConfig] = Field(
        json_schema_extra={
            "reader": SequenceOfObjects.build_reader(get_transform_from_dict)
        }
    )
    sample_transforms: SequenceOfObjects[Transform, TransformConfig] = Field(
        json_schema_extra={
            "reader": SequenceOfObjects.build_reader(get_transform_from_dict)
        }
    )
    augmentations: SequenceOfObjects[Transform, TransformConfig] = Field(
        json_schema_extra={
            "reader": SequenceOfObjects.build_reader(get_transform_from_dict)
        }
    )

    @field_validator(
        "image_transforms", "sample_transforms", "augmentations", mode="before"
    )
    @classmethod
    def _handle_sequence(cls, v: Any, info: ValidationInfo) -> SequenceOfObjects:
        return SequenceOfObjects.from_sequence(v, field_name=info.field_name)

    @classmethod
    def _get_class(cls) -> type[TransformsHandler]:
        """Returns the class associated to this config class."""
        return TransformsHandler


@equal_if_config_equal
class TransformsHandler(HasConfig[TransformsHandlerConfig]):
    """
    Handles the transformation pipeline applied to images.

    ``ClinicaDL`` defines 4 types of transforms:\n
    - ``extraction``: defines on what type of elements of the image one wants to work
      (the whole image, patches or slices).
    - ``image_transforms``: transforms applied on the whole image, **before the
      potential extraction** is applied. This is typically where you want to
      do normalization (to normalize on the whole image and not only on a patch
      or a slice).
    - ``sample_transforms``: transforms applied on a sample (a patch or a slice),
      **after extraction**. This is typically where you want to
      resize your sample so that it fits in your network.
    - ``augmentations``: transforms applied after ``image_transforms``, ``extraction``
      and ``sample_transforms``, **only during training**.

    For ``image_transforms``, ``sample_transforms`` and ``augmentations``, the transforms must be passed as sequences.
    ``TransformsHandler`` will compose the transforms in these sequences. So, **the order in the sequences is important**.

    Parameters
    ----------
    extraction : Optional[Extraction], default=None
        The extraction applied. See :py:mod:`clinicadl.transforms.extraction`. If ``None``,
        :py:class:`~clinicadl.transforms.extraction.Image` is used (which is equivalent to no extraction).
    image_transforms : Sequence[TransformOrConfig], default=()
        A sequence of transforms to apply on the whole image, **before extraction**.
        Passed as callables that take as input and return a :py:class:`~clinicadl.data.structures.DataPoint`,
        or :py:mod:`configuration class <clinicadl.transforms.config>`.
    sample_transforms : Sequence[TransformOrConfig], default=()
        A sequence of transforms to apply on samples (patches or slices).
        Passed as callables that take as input and return a :py:class:`~clinicadl.data.structures.DataPoint`,
        or :py:mod:`configuration class <clinicadl.transforms.config>`.

    augmentations : Sequence[TransformOrConfig], default=()
        A sequence of augmentation transforms, to apply on samples, only during training.
        Passed as callables that take as input and return a :py:class:`~clinicadl.data.structures.DataPoint`,
        or :py:mod:`configuration class <clinicadl.transforms.config>`.

    Examples
    --------
    .. code-block:: python

        >>> from clinicadl.transforms import TransformsHandler
        >>> from clinicadl.transforms.extraction import Patch
        >>> from clinicadl.transforms.config import ZNormalizationConfig, RandomFlipConfig
        >>> import torchio
        >>> transforms = TransformsHandler(
                extraction=Patch(patch_size=32, stride=32),
                image_transforms=[ZNormalizationConfig(), torchio.CropOrPad(64)],
                sample_transforms=[],
                augmentations=[RandomFlipConfig(flip_probability=0.3)],
            )

    """

    _config_type = TransformsHandlerConfig

    def __init__(
        self,
        extraction: Optional[Extraction] = None,
        image_transforms: Sequence[TransformOrConfig] = (),
        sample_transforms: Sequence[TransformOrConfig] = (),
        augmentations: Sequence[TransformOrConfig] = (),
    ):
        if not extraction:
            extraction = Image()

        self.config = TransformsHandlerConfig(
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
        Returns a detailed string representation of the ``TransformsHandler`` object,
        showing the current configuration of image and sample transforms,
        augmentations, and other settings.
        """
        transform_str = f"TransformsHandler configuration for {self.extraction.sample_type} extraction:\n"

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
            The input ``DataPoint``.

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
            The input ``DataPoint``.
        sample_index : int
            Index of the sample to extract.

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
            The input ``DataPoint``.

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
            The input ``DataPoint``.

        Returns
        -------
        DataPoint
            The transformed ``DataPoint``.
        """
        return self.augmentations(datapoint)
