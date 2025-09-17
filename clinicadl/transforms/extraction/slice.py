from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
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
from clinicadl.utils.exceptions import ClinicaDLTSVError
from clinicadl.utils.typing import PathType

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
    image_path : PathType
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

    .. note::
        To select slices, use ``slices``, ``tsv_path``, ``discarded_slices``, or
        ``borders``.

        - If none of these parameters is passed, all slices will be kept.
        - ``slices`` and ``tsv_path`` cannot be used in conjunction another slice selection
          parameter, but ``discarded_slices`` and ``borders`` can be passed together.

    Parameters
    ----------
    slices : Optional[List[NonNegativeInt]], default=None
        The slices to select. The slices selected will be the same for all images; if you
        want a different selection for each image, use ``tsv_path``.
    tsv_path : Optional[PathType], default=None
        Path to a ``TSV`` file containing slice indices for each image.
        The ``TSV`` table must have the columns: ``participant_id``, ``session_id``, and ``slice_idx``.
    discarded_slices : Optional[List[NonNegativeInt]], default=None
        Indices of the slices to discard. Cannot be used with ``slices`` or ``tsv_path``.
    borders : Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]], default=None
        The number of border slices that will be filtered out. If an integer ``a`` is passed, the first
        ``a`` slices and the last ``a`` slices will be filtered out. If a tuple ``(a, b)`` is passed, the first
        ``a`` slices and the last ``b`` slices will be filtered out.\n
        Cannot be used with ``slices`` or ``tsv_path``.
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
    tsv_path: Optional[Path] = None
    discarded_slices: Optional[List[NonNegativeInt]] = None
    borders: Optional[Tuple[PositiveInt, PositiveInt]] = None
    slice_direction: SliceDirection = SliceDirection.SAGITTAL
    squeeze: bool = True
    _map: Optional[Dict[Tuple[str, str], List[int]]] = None

    def __init__(
        self,
        *,
        slices: Optional[List[NonNegativeInt]] = None,
        tsv_path: Optional[PathType] = None,
        discarded_slices: Optional[List[NonNegativeInt]] = None,
        borders: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]] = None,
        slice_direction: SliceDirection = SliceDirection.SAGITTAL,
        squeeze: bool = True,
    ) -> None:
        super().__init__(
            slices=slices,
            tsv_path=tsv_path,
            discarded_slices=discarded_slices,
            borders=self._ensure_tuple(borders),
            slice_direction=slice_direction,
            squeeze=squeeze,
        )
        if self.tsv_path is not None:
            self._map = self._load_tsv(self.tsv_path)

    @computed_field
    @property
    def extract_method(self) -> str:
        """The method to be used for the extraction process (Image, Patch, Slice)."""
        return ExtractionMethod.SLICE.value

    @staticmethod
    def _ensure_tuple(
        value: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]],
    ) -> Optional[Tuple[PositiveInt, PositiveInt]]:
        """
        Ensures that 'borders' is always a tuple.
        """
        if isinstance(value, int):
            return (value, value)
        return value

    @classmethod
    def _normalize_cols(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lowercases the column names and look for 'participant_id', 'session_id', and 'slice_idx'.
        """
        cols = {c.lower(): c for c in df.columns}
        subj_col = cols.get("participant_id")
        sess_col = cols.get("session_id")
        slice_col = cols.get("slice_idx")

        if not subj_col or not sess_col or not slice_col:
            raise ClinicaDLTSVError(
                "TSV must contain columns: 'participant_id', 'session_id', 'slice_idx'"
            )

        return df.rename(
            columns={
                subj_col: "participant_id",
                sess_col: "session_id",
                slice_col: "slice_idx",
            }
        )

    @classmethod
    def _load_tsv(cls, path: Path) -> Dict[Tuple[str, str], List[int]]:
        """
        Reads the TSV file and returns the mapping as a dict. The keys
        are the (participant, session) pairs and the values are the list
        of selected slices.
        """
        df = pd.read_csv(path, sep="\t")
        df = cls._normalize_cols(df)

        if not np.issubdtype(df["slice_idx"].dtype, np.integer):
            try:
                df["slice_idx"] = df["slice_idx"].astype(int)
            except Exception as e:
                raise ValueError("Column 'slice_idx' must contain integers.") from e

        mapping: Dict[Tuple[str, str], List[int]] = {}
        for (sub, ses), g in df.groupby(["participant_id", "session_id"]):
            mapping[(str(sub), str(ses))] = list(map(int, g["slice_idx"].tolist()))

        return mapping

    def _slices_for(self, data_point: DataPoint) -> List[int]:
        """
        Gets the slice selection for a specific image.
        """
        if self._map is None:
            raise RuntimeError("Called _slices_for but no TSV was provided.")

        key = (data_point.participant, data_point.session)
        if key not in self._map:
            raise ValueError(
                f"No slices found in TSV for participant={key[0]}, session={key[1]}."
            )

        return self._map[key]

    @model_validator(mode="after")
    def validate_slices(self) -> Self:
        """
        Checks consistency between 'slices', 'tsv_path', 'discarded_slices' and 'borders'.
        """
        if self.slices and self.tsv_path:
            raise ValueError("'slices' and 'tsv_path' can't be passed simultaneously.")

        slices_or_tsv = self.slices or self.tsv_path
        if slices_or_tsv and self.discarded_slices:
            raise ValueError(
                "You can't pass 'discarded_slices' if 'slices' or 'tsv_path' was passed."
            )
        elif slices_or_tsv and self.borders:
            raise ValueError(
                "You can't pass 'borders' if 'slices' or 'tsv_path' was passed."
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
        slice_position = self._get_sample_position(data_point, sample_index)
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

    def num_samples_per_image(self, data_point: DataPoint) -> int:
        """
        Returns the number of slices that can be extracted from the input image tensor.

        If ``slices``, ``discarded_slices`` and ``borders`` have not been passed, there is no
        slice filtering, so the function will simply output the number of slices in the
        image.

        Parameters
        ----------
        data_point : DataPoint
            The DataPoint containing the image to perform extraction on.

        Returns
        -------
        int
            The number of slices remaining after slice filtering.

        Raises
        ------
        IndexError
            If ``slices`` or ``discarded_slices`` mention slices that are not in the image.
        """
        return self._get_slice_selection(data_point).sum()

    def _get_slice_selection(self, data_point: DataPoint) -> np.ndarray[bool]:
        """
        Returns the slices of an image that can be extracted, depending on ``slices``, ``tsv_path``,
        ``discarded_slices`` and ``borders``.
        """
        n_slices = data_point.image.tensor.size(self.slice_direction + 1)
        selection = np.ones(n_slices).astype(bool)

        slice_indices = None
        if self._map:
            slice_indices = self._slices_for(data_point)
        elif self.slices:
            slice_indices = self.slices

        if slice_indices:
            selection = ~selection
            try:
                selection[slice_indices] = True
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

    def _get_sample_position(self, data_point: DataPoint, sample_index: int) -> int:
        """
        Returns the position in the image of ``sample_index``. They may differ as
        ``sample_index`` is the index among the selected slices.

        Raises
        ------
        IndexError
            If ``sample_index`` is greater or equal to the number of selected slices in the image.
        """
        selection = self._get_slice_selection(data_point)
        slice_positions = np.arange(len(selection))[selection]

        try:
            return int(slice_positions[sample_index])
        except IndexError as exc:
            raise IndexError(
                f"'sample_index' {sample_index} is out of range as there are only "
                f"{len(slice_positions)} selected slices in the image."
            ) from exc

    def _extract_tensor_sample(
        self, image_tensor: torch.Tensor, sample_position: int
    ) -> torch.Tensor:
        """
        Gets the wanted slice, according to the slicing direction.
        """
        if self.slice_direction == 0:
            slice_tensor = image_tensor[:, sample_position, :, :]
        elif self.slice_direction == 1:
            slice_tensor = image_tensor[:, :, sample_position, :]
        elif self.slice_direction == 2:
            slice_tensor = image_tensor[:, :, :, sample_position]

        return slice_tensor.unsqueeze(self.slice_direction + 1)  # pylint: disable=possibly-used-before-assignment
