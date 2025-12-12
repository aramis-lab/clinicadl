from abc import ABC
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import torchio as tio
from pydantic import NonNegativeInt, field_validator, model_validator
from typing_extensions import Self

from clinicadl.utils.enum import SliceDirection
from clinicadl.utils.typing import PathType

from ..datatypes import DataType
from .datapoint import DataPoint, DataPointConfig


class SampleType(str, Enum):
    """The types of sample supported in ClinicaDL."""

    IMAGE = "image"
    PATCH = "patch"
    SLICE = "slice"


class SampleConfig(DataPointConfig):
    """To check ``Sample`` inputs."""

    datatype: tuple[DataType, ...]
    image_path: tuple[Path, ...]
    sample_type: SampleType
    sample_position: Optional[
        Union[NonNegativeInt, tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]]
    ]

    @field_validator("datatype", mode="before")
    @classmethod
    def _validate_tuple(cls, value: Any) -> Self:
        """To accept a single value for 'datatype'."""
        if not isinstance(value, Sequence):
            return (value,)
        return value

    @field_validator("image_path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> Self:
        """To accept str and a single value for 'image_path'."""
        if isinstance(value, Sequence) and not isinstance(value, str):
            return tuple(Path(v) for v in value)
        return (Path(value),)

    @model_validator(mode="after")
    def _validate_image_channels(self) -> Self:
        """
        To validate the number of channels in the image.
        """
        if len(self.datatype) == 1:
            self.__dict__["datatype"] = self.datatype * self.image.num_channels
        elif self.image.num_channels != len(self.datatype):
            raise ValueError(
                f"'datatype' has {len(self.datatype)} value(s) but there are {self.image.num_channels} channel(s) in the image."
            )

        if len(self.image_path) == 1:
            self.__dict__["image_path"] = self.image_path * self.image.num_channels
        elif self.image.num_channels != len(self.image_path):
            raise ValueError(
                f"'image_path' has {len(self.image_path)} value(s) but there are {self.image.num_channels} channel(s) in the image."
            )

        return self

    @model_validator(mode="after")
    def _validate_sample_position(self) -> Self:
        """To check 'sample_position' depending on 'sample_type'."""
        if self.sample_type == SampleType.IMAGE:
            assert (
                self.sample_position is None
            ), f"if sample_type={SampleType.IMAGE.value}, 'sample_position' must be None"
        elif self.sample_type == SampleType.PATCH:
            assert isinstance(
                self.sample_position, tuple
            ), f"if sample_type={SampleType.PATCH.value}, 'sample_position' must be a 3D tuple corresponding to the position of the patch in the image"
        elif self.sample_type == SampleType.SLICE:
            assert isinstance(
                self.sample_position, int
            ), f"if sample_type={SampleType.SLICE.value}, 'sample_position' must be an int corresponding to the position of the slice in the image"

        return self


class Sample(DataPoint, ABC):
    """
    The output of :py:class:`~clinicadl.data.datasets.ClinicaDLDataset`.

    It is a :py:class:`DataPoint <clinicadl.data.structures.DataPoint>`, with additional attributes.

    Consistency of voxel spacings and spatial shapes of the different images inside the ``Sample``
    will be checked, unless ``check_consistency=False``.

    Attributes
    ----------
    image : torchio.ScalarImage
        The image, as a :py:class:`torchio.ScalarImage`.
    participant : str
        The id of the participant.
    session : str
        The id of the session.
    datatype : tuple[DataType, ...]
        The :py:class:`~clinicadl.data.datatypes.DataType`. If they are multiple images in ``image``
        (i.e. multiple channels), the :py:class:`~clinicadl.data.datatypes.DataType` of each of them
        is expected.
    image_path : tuple[Path, ...]
        The path to the image. If they are multiple images in ``image``
        (i.e. multiple channels), the path of each of them
        is expected.
    sample_type : SampleType
        The type of the sample, among {"image", "slice", "patch"}.
    sample_position : Optional[Union[int, tuple[int, int, int]]]
        The position of the sample in the image if relevant, ``None`` otherwise.

        - If ``sample_type="slice"``: the index of the slice in the original image is expected.
        - If ``sample_type="patch"``: the position of the patch (i.e. the position of its upper left voxel) in the
          original image is expected.

    label : Optional[Union[int, float, torchio.LabelMap]], default = None
        The label. Either ``None``, a scalar, a dict of scalars, or a mask, as a :py:class:`torchio.LabelMap`.
    """

    datatype: tuple[DataType, ...]
    image_path: tuple[Path, ...]
    sample_type: SampleType
    sample_position: Optional[Union[int, tuple[int, int, int]]] = None

    def __init__(
        self,
        image: Union[tio.ScalarImage, PathType],
        participant: str,
        session: str,
        datatype: Union[DataType, tuple[DataType, ...]],
        image_path: Union[Path, tuple[Path, ...]],
        sample_type: SampleType = SampleType.IMAGE,
        sample_position: Optional[Union[int, tuple[int, int, int]]] = None,
        label: Optional[
            Union[float, int, dict[str, float], tio.LabelMap, PathType]
        ] = None,
        check_consistency: bool = True,
        **kwargs: Any,
    ):
        config = SampleConfig(
            image=image,
            participant=participant,
            session=session,
            label=label,
            datatype=datatype,
            image_path=image_path,
            sample_type=sample_type,
            sample_position=sample_position,
        )
        kwargs.update(config.to_raw_dict())
        super().__init__(**kwargs)
        if check_consistency:
            _ = self.spatial_shape
            _ = self.spacing


class SliceSampleConfig(SampleConfig):
    """To check ``SliceSample`` inputs."""

    sample_type: SampleType = SampleType.SLICE
    slice_direction: SliceDirection
    squeeze: bool

    @model_validator(mode="after")
    def _validate_slice(self) -> Self:
        """
        To validate that it is indeed a slice.
        """
        assert self.image.spatial_shape[self.slice_direction] == 1, (
            f"The dimension along 'slice_direction' should be 1. But here got slice_direction={self.slice_direction} "
            f"and spatial_shape of {self.image.spatial_shape}"
        )

        return self


class Sample2D(Sample):
    """
    A slice :py:class:`Sample`. Here ``sample_type="slice"`` and ``sample_position`` is the position
    of the slice in the original image.

    Besides, there are two addition attribute:

    slice_direction : int
        The slicing direction. Can be ``0`` (sagittal direction), ``1`` (coronal)
        or ``2`` (axial).
    squeeze : bool
        Whether the tensors will be later squeezed to work with 2D slice, or whether the slices will stay 3D
        (with one dummy dimension). The attribute is useful for some ``ClinicaDL`` operations.
    """

    sample_position: int
    slice_direction: int
    squeeze: bool

    def __init__(
        self,
        image: Union[tio.ScalarImage, PathType],
        participant: str,
        session: str,
        datatype: DataType,
        image_path: Path,
        sample_position: int,
        slice_direction: int,
        squeeze: bool,
        label: Optional[
            Union[float, int, dict[str, float], tio.LabelMap, PathType]
        ] = None,
        check_consistency: bool = True,
        **kwargs: Any,
    ):
        config = SliceSampleConfig(
            image=image,
            participant=participant,
            session=session,
            label=label,
            datatype=datatype,
            image_path=image_path,
            sample_position=sample_position,
            slice_direction=slice_direction,
            squeeze=squeeze,
        )
        kwargs.update(config.to_raw_dict())
        super().__init__(**kwargs, check_consistency=check_consistency)
