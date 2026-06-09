from abc import ABC
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torchio as tio
from pydantic import NonNegativeInt, field_validator, model_validator
from torch import Tensor
from typing_extensions import Self

from clinicadl.io.bids import BidsFileType
from clinicadl.utils.enum import SliceDirection
from clinicadl.utils.typing import PathType

from .datapoint import DataPoint, DataPointConfig


class SampleType(str, Enum):
    """The types of sample supported in ClinicaDL."""

    IMAGE = "image"
    PATCH = "patch"
    SLICE = "slice"


class SampleConfig(DataPointConfig):
    """To check ``Sample`` inputs."""

    file_type: tuple[BidsFileType, ...]
    image_path: tuple[Path, ...]
    sample_type: SampleType
    sample_position: Optional[
        Union[NonNegativeInt, tuple[NonNegativeInt, NonNegativeInt, NonNegativeInt]]
    ]

    @field_validator("file_type", mode="before")
    @classmethod
    def _validate_tuple(cls, value: Any) -> Self:
        """To accept a single value for 'file_type'."""
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
        if len(self.file_type) == 1:
            self.__dict__["file_type"] = self.file_type * self.image.num_channels
        elif self.image.num_channels != len(self.file_type):
            raise ValueError(
                f"'file_type' has {len(self.file_type)} value(s) but there are {self.image.num_channels} channel(s) in the image."
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
    The output of a :py:class:`~clinicadl.data.datasets.Dataset`.

    It is a :py:class:`DataPoint <clinicadl.data.structures.DataPoint>`, with additional attributes.

    Attributes
    ----------
    image : torchio.ScalarImage
        The image, in a :py:class:`torchio.ScalarImage`.
    participant_id : str
        The id of the participant_id.
    session_id : str
        The id of the session.
    file_type : tuple[BidsFileType, ...]
        The :py:class:`~clinicadl.io.bids.BidsFileType`. If they are multiple images in ``image``
        (i.e. multiple channels), the :py:class:`~clinicadl.data.datatypes.BidsFileType` of each of them
        is expected.
    image_path : tuple[Path, ...]
        The :pathlib.Path:`path <>` to the image. If they are multiple images in ``image``
        (i.e. multiple channels), the path of each of them
        is expected.
    sample_type : str | SampleType
        The type of the sample, among {"image", "slice", "patch"}.
    sample_position : Optional[Union[int, tuple[int, int, int]]]
        The position of the sample in the image if relevant, ``None`` otherwise.

        - If ``sample_type="slice"``: the index of the slice in the original image is expected.
        - If ``sample_type="patch"``: the position of the patch (i.e. the position of its upper left voxel) in the
          original image is expected.
    """

    file_type: tuple[BidsFileType, ...]
    image_path: tuple[Path, ...]
    sample_type: SampleType
    sample_position: Optional[Union[int, tuple[int, int, int]]] = None

    def __init__(
        self,
        image: Union[tio.ScalarImage, PathType],
        participant_id: str,
        session_id: str,
        file_type: Union[BidsFileType, tuple[BidsFileType, ...]],
        image_path: Union[Path, tuple[Path, ...]],
        sample_type: str | SampleType = SampleType.IMAGE,
        sample_position: Optional[Union[int, tuple[int, int, int]]] = None,
        **kwargs: Any,
    ):
        config = SampleConfig(
            image=image,
            participant_id=participant_id,
            session_id=session_id,
            file_type=file_type,
            image_path=image_path,
            sample_type=sample_type,
            sample_position=sample_position,
        )
        kwargs.update(config.to_raw_dict())
        super().__init__(**kwargs)


class Sample2DConfig(SampleConfig):
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
    A :py:class:`Sample` corresponding to a 2D slice.

    Here ``sample_type="slice"`` and ``sample_position`` is the position of the slice in the original image.

    Besides, there are two additional attributes:

    slice_direction : int
        The slicing direction (``0``, ``1`` or ``2``).

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
        participant_id: str,
        session_id: str,
        file_type: BidsFileType,
        image_path: Path,
        sample_position: int,
        slice_direction: int,
        squeeze: bool,
        **kwargs: Any,
    ):
        config = Sample2DConfig(
            image=image,
            participant_id=participant_id,
            session_id=session_id,
            file_type=file_type,
            image_path=image_path,
            sample_position=sample_position,
            slice_direction=slice_direction,
            squeeze=squeeze,
        )
        kwargs.update(config.to_raw_dict())
        super().__init__(**kwargs)

    def get_image_tensor(self, image_name: str) -> Tensor:
        """
        Returns a copy of the tensor associated to a field that is a :py:class:`torchio.Image`.

        If ``squeeze=True``, the output tensor will be squeezed.

        Parameters
        ----------
        image_name : str
            The name of the image in the ``Sample2D``.

        Returns
        -------
        torch.Tensor
            The tensor image.
        """
        tensor = super().get_image_tensor(image_name)

        if self.squeeze:
            tensor.squeeze_(dim=self.slice_direction + 1)

        return tensor

    def add_image(
        self,
        image: Union[tio.ScalarImage, PathType, torch.Tensor],
        image_name: str,
    ) -> None:
        """
        To add an image to the ``Sample``.

        Parameters
        ----------
        image : Union[tio.ScalarImage, PathType, torch.Tensor]
            The image to add, as a :py:class:`torchio.ScalarImage``, a path to the :term:`NIfTI` file containing the image,
            or a :py:class:`torch.Tensor`. In the latter case, it is expected to be a 4D ``Tensor`` (including one channel dimension)
            if ``squeeze=False``, or a 3D ``Tensor`` if ``squeeze=True``.

            If a ``Tensor`` is passed, the same affine matrix as ``self.image`` will be used.
        image_name : str
            The name that the image will take in the ``Sample``.

        See Also
        --------
        :py:meth:`DataPoint.add_image <clinicadl.data.structures.DataPoint.add_image>`
        """
        if isinstance(image, torch.Tensor):
            self._unsqueeze_tensor(image)

        super().add_image(image, image_name)

    def add_mask(self, mask: Union[tio.LabelMap, PathType], mask_name: str) -> None:
        """
        To add a mask to the ``Sample``.

        Parameters
        ----------
        mask : Union[tio.ScalarImage, PathType, torch.Tensor]
            The mask to add, as a :py:class:`torchio.LabelMap`, a path to the :term:`NIfTI` file containing the mask,
            or a :py:class:`torch.Tensor`. In the latter case, it is expected to be a 4D ``Tensor`` (including one channel dimension)
            if ``squeeze=False``, or a 3D ``Tensor`` if ``squeeze=True``.

            If a ``Tensor`` is passed, the same affine matrix as ``self.image`` will be used.
        mask_name : str
            The name that the image will take in the ``Sample``.

        See Also
        --------
        :py:meth:`DataPoint.add_mask <clinicadl.data.structures.DataPoint.add_mask>`
        """
        if isinstance(mask, torch.Tensor):
            self._unsqueeze_tensor(mask)

        super().add_mask(mask, mask_name)

    def _unsqueeze_tensor(self, tensor: torch.Tensor) -> None:
        """
        Unsqueeze tensors if squeeze=True.
        """
        if self.squeeze:
            assert (
                len(tensor.shape) == 3
            ), f"If squeeze=True, a 3D tensor is expected (including one channel dimension). Got: {tensor.shape}"
            tensor.unsqueeze_(dim=self.slice_direction + 1)
        else:
            assert (
                len(tensor.shape) == 4
            ), f"If squeeze=False, a 4D tensor is expected (including one channel dimension). Got: {tensor.shape}"


SAMPLE_FIELDS = tuple(Sample2DConfig.model_fields.keys())
