from logging import getLogger
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import (
    NonNegativeInt,
    PositiveInt,
    computed_field,
    model_validator,
)
from typing_extensions import Self

from clinicadl.data.structures import DataPoint
from clinicadl.utils.enum import SliceDirection

from .base import Extraction, ExtractionMethod, Sample

logger = getLogger("clinicadl.extraction.slice")


class SliceSample(Sample):
    """
    Output of a CapsDataset when slice extraction is performed (i.e.
    when :py:class:`~Slice` is used).

    It is simply a :py:class:`~clinicadl.data.structures.DataPoint`, with
    additional information on the slice extraction.

    Attributes
    ----------
    image : torchio.ScalarImage
        The slice, as a :py:class:`torchio.ScalarImage`.
    label : Optional[Union[float, int, torchio.LabelMap]]
        The label associated to the slice. Can be a ``float`` (regression),
        an ``int`` (classification), a mask (as a :py:class:`torchio.LabelMap`; for segmentation)
        or ``None`` if no label (reconstruction). If the label is a mask, slice extraction
        was also performed on it.
    participant : str
        The participant concerned.
    session : str
        The session concerned.
    preprocessing : Preprocessing
        The proprocessing of the image (see :ref:`api_data_types`).
    image_path : Union[str, Path]
        The path to the image.
    slice_position : int
        The position of the slice in the original image.
    slice_direction : SliceDirection
        The slicing direction. Can be ``0`` (sagittal direction), ``1`` (coronal)
        or ``2`` (axial).
    squeeze : bool
        Whether the tensors will be squeezed.
    """

    slice_position: int
    slice_direction: SliceDirection
    squeeze: bool

    @property
    def _sample_index(self) -> int:
        """The index of the sample. Equal to 'slice_position' here."""
        return self.slice_position


class Slice(Extraction):
    """
    Transform class to extract slices from an image in a specified direction.

    Adds the following keys to the input :py:class:`~clinicadl.data.structures.DataPoint`:

    - ``slice_position``: int
        The position of the slice in the original image.
    - ``slice_direction``: 0, 1 or 2
        The slicing direction.
    - ``squeeze``: bool
        Whether the tensors will be squeezed to work with 2D neural networks.

    Parameters
    ----------
    slices : Optional[List[NonNegativeInt]], default=None
        The slices to select. If ``None``, slices will be selected with ``discarded_slices``
        and/or ``borders``. If all these three parameters are ``None``, all slices will be
        kept.
    discarded_slices : Optional[List[NonNegativeInt]], default=None
        Indices of the slices to discard. Cannot be used with ``slices``.
    borders : Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]], default=None
        The number of border slices that will be filtered out. If an integer ``a`` is passed, the first
        ``a`` slices and the last ``a`` slices will be filtered out. If a tuple ``(a, b)`` is passed, the first
        ``a`` slices and the last ``b`` slices will be filtered out.
    slice_direction : SliceDirection, default=0
        The slicing direction. Can be ``0`` (sagittal direction), ``1`` (coronal) or ``2`` (axial).
    squeeze : bool, default=True
        Whether to squeeze slices to have images with 2 spatial dimensions.
        If ``False``, slices will still have 3 spatial dimensions.

        .. note::
            Squeezing will be performed by ``ClinicaDL`` just before putting the images in the neural
            network. This is because most of ``ClinicaDL`` tools work with 3D images.
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

    def extract_sample(self, data_point: DataPoint, sample_index: int) -> SliceSample:
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
        SliceSample
            A :py:func:`~ImageSample` object with the extracted slices for each image
            present in the original ``data_point``. The slice extracted from an
            image is accessible via the same name as was the image in the original
            ``data_point``.
            Additional information on the extraction is added.

        Raises
        ------
        IndexError
            If ``slices`` or ``discarded_slices`` mention slices that are not in the image.
        IndexError
            If ``sample_index`` is greater or equal to the number of selected slices in the image.
        """
        slice_position = self._get_slice_position(data_point.image.tensor, sample_index)
        extracted_datapoint = self._extract_datapoint_sample(data_point, sample_index)
        sample = SliceSample(
            **extracted_datapoint,
            extraction=self.extract_method,
            slice_position=slice_position,
            slice_direction=self.slice_direction,
            squeeze=self.squeeze,
        )
        sample.applied_transforms = extracted_datapoint.applied_transforms

        return sample

    def num_samples_per_image(self, image: torch.Tensor) -> int:
        """
        Returns the number of slices that can be extracted from the input image tensor.

        If ``slices``, ``discarded_slices`` and ``borders`` have not been passed, there is no
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
            If ``slices`` or ``discarded_slices`` mention slices that are not in the image.
        """
        return self._get_slice_selection(image).sum()

    def _extract_tensor_sample(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> torch.Tensor:
        """
        Extracts a single slice from an image.

        Raises
        ------
        IndexError
            If ``slices`` or ``discarded_slices`` mention slices that are not in the image.
        IndexError
            If ``sample_index`` is greater or equal to the number of selected slices in the image.
        """
        slice_position = self._get_slice_position(image_tensor, sample_index)
        slice_tensor = self._get_slice(image_tensor, slice_position)

        return slice_tensor

    def _get_slice_selection(self, image: torch.Tensor) -> np.ndarray[bool]:
        """
        Returns the slices of an image that can be extracted, depending on ``slices``,
        ``discarded_slices`` and ``borders``.
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
        Returns the position in the image of ``slice_index``. They may differ as
        ``slice_index`` is the index among the selected slices.
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
