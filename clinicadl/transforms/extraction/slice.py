from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchio as tio
from pydantic import (
    NonNegativeInt,
    PositiveInt,
    computed_field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from clinicadl.utils.enum import (
    ExtractionMethod,
    SliceDirection,
    SliceMode,
)

from .base import Extraction, Sample

logger = getLogger("clinicadl.extraction.slice")

PT = ".pt"


class SliceSample(Sample):
    """
    Output of a CapsDataset when slice extraction is performed.

    Attributes
    ----------
    sample : torch.Tensor
       the 2D slice as 3D PyTorch tensor (with one channel dimension).
    participant_id : str
        the subject concerned.
    session_id : str
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
    def extraction(self) -> ExtractionMethod:
        """The extraction method."""
        return ExtractionMethod.SLICE


class Slice(Extraction):
    """
    Transform class to extract slices from an image in a specified direction.

    This class allows users to define extraction configurations for obtaining slices from a 3D image tensor.
    The extracted slices can be processed in different directions (e.g., sagittal, coronal, axial) and can
    be adjusted for RGB mode.

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
    """

    slices: Optional[List[NonNegativeInt]] = None
    discarded_slices: Optional[List[NonNegativeInt]] = None
    borders: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]] = None
    slice_direction: SliceDirection = SliceDirection.SAGITTAL

    @computed_field
    @property
    def extract_method(self) -> ExtractionMethod:
        """The method to be used for the extraction process (ROI, Image, Patch, Slice)."""
        return ExtractionMethod.SLICE

    @field_validator("borders", mode="after")
    @classmethod
    def validate_borders(
        cls, v: Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]
    ) -> Tuple[PositiveInt, PositiveInt]:
        """
        Ensures that 'borders' is always a tuple.
        """
        if isinstance(v, int):
            return (v, v)
        else:
            return v

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
        """
        return self._get_slice_selection(image).sum()

    # TODO : remove?
    def extract(self, nii_path: Path) -> List[Tuple[Path, torch.Tensor]]:
        """
        Extracts all the selected slices from the image in the specified direction.

        Parameters
        ----------
        nii_path : Path
            The path to the input NIfTI image file.

        Returns
        -------
        List[Tuple[Path, torch.Tensor]]
            A list of tuples, where each tuple contains an extracted slice,
            and the path where to store it.
        """
        image_tensor = self.load_image(nii_path)
        slices = []
        for i in range(self.num_samples_per_image(image_tensor)):
            slice_tensor = self.extract_sample(image_tensor, i).squeeze(
                self.slice_direction + 1
            )
            slices.append((self.sample_path(nii_path, i), slice_tensor))

        return slices

    def extract_sample(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> torch.Tensor:
        """
        Extracts a single slice from an image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The input image tensor, which is a 4D tensor with a channel dimension and 3 spatial
            dimensions.
        sample_index : int
            The index of the slice to extract in the specified direction.

        Returns
        -------
        torch.Tensor
            The extracted slice as a tensor. The tensor is still 4D (with the dimension in
            the slice direction equal to 1).

        Raises
        ------
        IndexError
            If 'sample_index' is greater or equal to the number of slices in the image.
        """
        slice_position = self._get_slice_position(image_tensor, sample_index)
        slice_tensor = self._get_slice(image_tensor, slice_position)

        return slice_tensor.clone()

    # TODO : remove?
    def sample_path(self, image_path: Path, sample_index: int) -> Path:
        """
        Constructs the file path for saving a given slice, based on the input image path and
        the slice index.

        Parameters
        ----------
        img_path : Path
            The path to the input image file.
        sample_index : int
            The index of the slice being saved.

        Returns
        -------
        Path
            The constructed file path for the slice.
        """
        parent = image_path.parent
        prefix_suffix = image_path.name.rsplit("_", 1)
        slice_dict = {0: "sag", 1: "cor", 2: "axi"}

        return (
            (
                parent / f"{prefix_suffix[0]}_axis-{slice_dict[self.slice_direction]}"
                f"_slice-{sample_index}_{prefix_suffix[1]}"
            )
            .with_suffix("")
            .with_suffix(PT)
        )

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
                    f"slices in the image are indexed from 0 to {n_slices-1}, but got "
                    f"slices={self.slices}."
                ) from exc
        else:
            if self.discarded_slices:
                try:
                    selection[self.discarded_slices] = False
                except IndexError as exc:
                    raise IndexError(
                        "Invalid slices in 'discarded_slices': "
                        f"slices in the image are indexed from 0 to {n_slices-1}, but got "
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
            return slice_positions[slice_index]
        except IndexError as exc:
            raise IndexError(
                f"'sample_index' {slice_index} is out of range as there are only "
                f"{len(slice_positions)} slices in the image."
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

    def _get_sample_description(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> int:
        """
        The sample description for slice extraction is the position of the slice
        in the original image.
        """
        return self._get_slice_position(image_tensor, sample_index)

    def format_output(
        self,
        tio_sample: tio.Subject,
        participant_id: str,
        session_id: str,
        image_path: Union[str, Path],
    ) -> SliceSample:
        """
        Puts all the output information in an SliceSample object.

        Parameters
        ----------
        tio_sample : tio.Subject
            a TorchIO Subject corresponding to the slice, with at least a ScalarImage named 'sample',
            an attribute named 'label' and an attribute named 'description'.
        participant_id : str
            the subject concerned.
        session_id : str
            the session concerned.
        image_path : Union[str, Path]
            the path of the image from which the slice is extracted.

        Returns
        -------
        SliceSample
            a SliceSample object with all the relevant information on the slice.

        Raises
        ------
        AttributeError
            if `tio_sample` doesn't have a TorchIO ScalarImage named 'sample', and attributes
            'label' and 'description'.
        """
        self._check_tio_sample(tio_sample)

        sample = tio_sample.sample.tensor.squeeze(self.slice_direction + 1)
        if isinstance(tio_sample.label, tio.Image):
            label = tio_sample.label.tensor.squeeze(self.slice_direction + 1)
        else:
            label = tio_sample.label

        return SliceSample(
            sample=sample,
            participant_id=participant_id,
            session_id=session_id,
            image_path=str(image_path),
            label=label,
            slice_position=tio_sample.description,
            slice_direction=self.slice_direction,
        )
