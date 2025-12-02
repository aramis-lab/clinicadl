from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import torchio as tio
from pydantic import NonNegativeInt, model_validator
from typing_extensions import Self

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.enum import SliceDirection
from clinicadl.utils.typing import PathType

from ..datatypes import DataType
from ..structures.datapoint import DataPoint


class SampleType(str, Enum):
    """The types of sample supported in ClinicaDL."""

    IMAGE = "image"
    PATCH = "patch"
    SLICE = "slice"


class SampleConfig(ClinicaDLConfig):
    """To check ``Sample`` inputs."""

    datatype: DataType
    image_path: Path
    sample_type: SampleType
    sample_position: Optional[
        Union[NonNegativeInt, tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]]
    ]

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

    Attributes
    ----------
    image : torchio.ScalarImage
        The image, as a :py:class:`torchio.ScalarImage`.
    participant : str
        The id of the participant.
    session : str
        The id of the session.
    datatype : DataType
        The :py:class:`type of data <clinicadl.data.datatypes>`.
    image_path : Path
        The path to the image.
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

    datatype: DataType
    image_path: Path
    sample_type: SampleType
    sample_position: Optional[Union[int, tuple[int, int, int]]] = None

    def __init__(
        self,
        image: Union[tio.ScalarImage, PathType],
        participant: str,
        session: str,
        datatype: DataType,
        image_path: Path,
        sample_type: SampleType = SampleType.IMAGE,
        sample_position: Optional[Union[int, tuple[int, int, int]]] = None,
        label: Optional[
            Union[float, int, dict[str, float], tio.LabelMap, PathType]
        ] = None,
        **kwargs: Any,
    ):
        config = SampleConfig(
            datatype=datatype,
            image_path=image_path,
            sample_type=sample_type,
            sample_position=sample_position,
        )
        kwargs.update(config.to_raw_dict())
        super().__init__(
            image=image, participant=participant, session=session, label=label, **kwargs
        )


class SliceSampleConfig(ClinicaDLConfig):
    """To check ``SliceSample`` inputs."""

    slice_direction: SliceDirection
    squeeze: bool


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
        **kwargs: Any,
    ):
        config = SliceSampleConfig(
            slice_direction=slice_direction,
            squeeze=squeeze,
        )
        super().__init__(
            image=image,
            participant=participant,
            session=session,
            label=label,
            datatype=datatype,
            image_path=image_path,
            sample_type=SampleType.SLICE,
            sample_position=sample_position,
            slice_direction=config.slice_direction,
            squeeze=config.squeeze,
            **kwargs,
        )
