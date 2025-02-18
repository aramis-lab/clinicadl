from logging import getLogger
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchio as tio
from pydantic import (
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)
from typing_extensions import Self

from clinicadl.data.structures import DataPoint
from clinicadl.utils.enum import (
    ExtractionMethod,
    SliceDirection,
)
from clinicadl.utils.typing import PathType

from .base import Extraction, Sample

logger = getLogger("clinicadl.extraction.slice")


class SliceSample(Sample):
    """
    Output of a CapsDataset when slice extraction is performed.

    Attributes
    ----------
    sample : torch.Tensor
       the slice as a PyTorch tensor with 1 channel dimension and
       2 (`squeeze=True`) or 3 (`squeeze=False`) spatial dimensions.
    affine : np.ndarray
        the affine matrix associated to the patch.
    participant : str
        the participant concerned.
    session : str
        the session concerned.
    image_path : str
        the path to the image from which the slice has been extracted.
    slice_position : int
        position of the slice in the original image.
    slice_direction : SliceDirection
        the slicing direction. Can be 0 (sagittal direction), 1 (coronal) or 2 (axial).
    """

    slice_position: NonNegativeInt
    slice_direction: SliceDirection

    @computed_field
    @property
    def extraction(self) -> str:
        """The extraction method."""
        return ExtractionMethod.SLICE.value


class Slice(Extraction):
    """
    Transform class to extract slices from an image in a specified direction.

    This class allows users to define extraction configurations for obtaining slices from a 4D image tensor
    (with one channel dimension).
    The slices can be extracted in different directions (e.g., sagittal, coronal, axial).

    Parameters
    ----------
    slices : Optional[List[NonNegativeInt]] (optional, default=None)
        the slices to select. If None, slices will be selected with `discarded_slices``
        and/or `borders`. If all these three parameters are None, all slices will be
        kept.
    discarded_slices : Optional[List[NonNegativeInt]] (optional, default=None)
        indices of the slices to discard. Cannot be used with `slices`.
    borders : Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]] (optional, default=None)
        the number of border slices, that will be filtered out. If an integer `a` is passed, the first
        `a` slices and the last `a` slices will be filtered out. If a tuple `(a, b)` is passed, the first
        `a` slices and the last `b` slices will be filtered out.
    slice_direction : SliceDirection (optional, default=SliceDirection.SAGITTAL)
        the slicing direction. Can be 0 (sagittal direction), 1 (coronal) or 2 (axial).
    squeeze : bool (optional, default=True)
        whether to squeeze slices to have images with 2 spatial dimensions.
        If False, slices will still have 3 spatial dimensions.
        ..note::squeezing is performed after applying transforms because ClinicaDL transforms
        only accepts images with 3 spatial dimensions.
    """

    slices: Optional[List[NonNegativeInt]] = None
    discarded_slices: Optional[List[NonNegativeInt]] = None
    borders: Optional[Tuple[PositiveInt, PositiveInt]] = None
    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    squeeze: bool = True

    def __init__(
        self,
        *,
        slices: Optional[List[NonNegativeInt]] = None,
        discarded_slices: Optional[List[NonNegativeInt]] = None,
        borders: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]] = None,
        slice_direction: SliceDirection = SliceDirection.SAGITTAL,
        squeeze: bool = True,
    ) -> None:
        super().__init__(
            slices=slices,
            discarded_slices=discarded_slices,
            borders=self._ensure_tuple(borders),
            slice_direction=slice_direction,
            squeeze=squeeze,
        )

    @computed_field
    @property
    def extract_method(self) -> str:
        """The method to be used for the extraction process (Image, Patch, Slice)."""
        return ExtractionMethod.SLICE.value

    @staticmethod
    def _ensure_tuple(
        value: Union[PositiveInt, Tuple[PositiveInt, PositiveInt]],
    ) -> Tuple[PositiveInt, PositiveInt]:
        """
        Ensures that 'borders' is always a tuple.
        """
        if isinstance(value, int):
            return (value, value)
        else:
            return value

    @model_validator(mode="after")
    def validate_slices(self) -> Self:
        """
        Checks consistency between 'slices', 'discarded_slices' and 'borders'.
        """
        if (self.slices is not None) and (self.discarded_slices is not None):
            raise ValueError(
                "'slices' and 'discarded_slices' can't be passed simultaneously. Specify the wanted slices "
                "in 'slices'."
            )
        elif (self.slices is not None) and (self.borders is not None):
            raise ValueError(
                "'slices' and 'borders' can't be passed simultaneously. Specify the wanted slices "
                "in 'slices'."
            )
        return self

    def extract_sample(
        self, data_point: DataPoint, sample_index: int
    ) -> Tuple[DataPoint, int]:
        """
        Extracts a slice from a DataPoint.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint to perform extraction on.
        sample_index : int
            Index indicating the slice to extract.

        Returns
        -------
        DataPoint
            A new DataPoint object with the extracted slices for each image
            present in the original `data_point`. The slice extracted from an
            image is accessible via the same name as was the image in the original
            `data_point`.
        int
            The slice position in the original image.

        Raises
        ------
        IndexError
            If 'slices' or 'discarded_slices' mention slices that are not in the image.
        IndexError
            If 'sample_index' is greater or equal to the number of selected slices in the image.
        """
        return super().extract_sample(data_point, sample_index)

    def num_samples_per_image(self, image: torch.Tensor) -> int:
        """
        Returns the number of slices that can be extracted from the input image tensor.

        If 'slices', 'discarded_slices' and 'borders' have not been passed, there is no
        slice filtering, so the function will simply output the number of slices in the
        image.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor (4D), where the first dimension represents the channel dimension.

        Returns
        -------
        int
            The number of slices remaining after slice filtering.

        Raises
        ------
        IndexError
            If 'slices' or 'discarded_slices' mention slices that are not in the image.
        """
        return self._get_slice_selection(image).sum()

    def format_output(
        self,
        data_point: DataPoint,
        image_path: PathType,
        description: int,
    ) -> SliceSample:
        """
        Puts all the output information in an SliceSample object.

        Parameters
        ----------
        data_point : DataPoint
            the `DataPoint` object associated to the slice.
        image_path : PathType
            the path of the image from which the slice is extracted.
        description : int
            the position of the slice in the original image.

        Returns
        -------
        SliceSample
            a SliceSample object with the slice (a 2D or 3D tensor with a channel dimension)
            and all the relevant information on the slice.
        """
        slice_ = data_point.image.tensor
        if self.squeeze:
            slice_ = slice_.squeeze(self.slice_direction + 1)

        label = data_point.label
        if isinstance(label, tio.LabelMap):
            label = label.tensor
            if self.squeeze:
                label = label.squeeze(self.slice_direction + 1)

        return SliceSample(
            sample=slice_,
            affine=data_point.image.affine,
            participant=data_point.participant,
            session=data_point.session,
            image_path=str(image_path),
            label=label,
            slice_position=description,
            slice_direction=self.slice_direction,
        )

    def _extract_tensor_sample(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> torch.Tensor:
        """
        Extracts a single slice from an image.

        Raises
        ------
        IndexError
            If 'slices' or 'discarded_slices' mention slices that are not in the image.
        IndexError
            If 'sample_index' is greater or equal to the number of selected slices in the image.
        """
        slice_position = self._get_slice_position(image_tensor, sample_index)
        slice_tensor = self._get_slice(image_tensor, slice_position)

        return slice_tensor.clone()

    def _get_sample_description(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> int:
        """
        The sample description for slice extraction is the position of the slice
        in the original image.
        """
        return self._get_slice_position(image_tensor, sample_index)

    def _get_slice_selection(self, image: torch.Tensor) -> np.ndarray[bool]:
        """
        Returns the slices of an image that can be extracted, depending on 'slices',
        'discarded_slices' and 'borders'.
        """
        n_slices = image.size(self.slice_direction + 1)
        selection = np.ones(n_slices).astype(bool)

        if self.slices:
            selection = ~selection
            try:
                selection[self.slices] = True
            except IndexError as exc:
                raise IndexError(
                    "Invalid slices in 'slices': "
                    f"slices in the image are indexed from 0 to {n_slices - 1}, but got "
                    f"slices={self.slices}."
                ) from exc
        else:
            if self.discarded_slices:
                try:
                    selection[self.discarded_slices] = False
                except IndexError as exc:
                    raise IndexError(
                        "Invalid slices in 'discarded_slices': "
                        f"slices in the image are indexed from 0 to {n_slices - 1}, but got "
                        f"discarded_slices={self.discarded_slices}."
                    ) from exc

            if self.borders:
                selection[: self.borders[0]] = False
                selection[n_slices - self.borders[1] :] = False

        return selection

    def _get_slice_position(self, image: torch.Tensor, slice_index: int) -> int:
        """
        Returns the position in the image of 'slice_index'. They may differ as
        'slice_index' is the index among the selected slices.
        """
        selection = self._get_slice_selection(image)
        slice_positions = np.arange(len(selection))[selection]

        try:
            return int(slice_positions[slice_index])
        except IndexError as exc:
            raise IndexError(
                f"'sample_index' {slice_index} is out of range as there are only "
                f"{len(slice_positions)} selected slices in the image."
            ) from exc

    def _get_slice(self, image: torch.Tensor, slice_position: int) -> torch.Tensor:
        """
        Gets the wanted slice, according to the slicing direction.
        """
        if self.slice_direction == 0:
            slice_tensor = image[:, slice_position, :, :]
        elif self.slice_direction == 1:
            slice_tensor = image[:, :, slice_position, :]
        elif self.slice_direction == 2:
            slice_tensor = image[:, :, :, slice_position]

        return slice_tensor.unsqueeze(self.slice_direction + 1)  # pylint: disable=possibly-used-before-assignment
