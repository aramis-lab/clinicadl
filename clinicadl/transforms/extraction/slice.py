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
    tsv_path : Optional[Union[str, Path]], default=None
        Path to a TSV file containing explicit slice indices per (participant, session).
        TSV must have columns: ``participant_id``, ``session_id``, ``slice_idx``.
        If provided, the TSV overrides ``slices``, ``discarded_slices`` and ``borders``.
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
    _tsv_path: Optional[str] = None
    _map: Optional[Dict[Tuple[str, str], List[int]]] = None

    def __init__(
        self,
        *,
        slices: Optional[List[NonNegativeInt]] = None,
        tsv_path: Optional[Union[str, Path]] = None,
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
        if tsv_path is not None:
            self._tsv_path = str(tsv_path)
            self._map = self._load_tsv(self._tsv_path)

    @computed_field
    @property
    def extract_method(self) -> str:
        return ExtractionMethod.SLICE.value

    @staticmethod
    def _ensure_tuple(
        value: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]],
    ) -> Optional[Tuple[PositiveInt, PositiveInt]]:
        if value is None:
            return None
        if isinstance(value, int):
            return (value, value)
        return value

    @staticmethod
    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        subj_col = cols.get("participant_id")
        sess_col = cols.get("session_id")
        slice_col = cols.get("slice_idx")
        if not subj_col or not sess_col or not slice_col:
            raise ValueError(
                "TSV must contain columns: participant_id, session_id, slice_idx"
            )
        return df.rename(
            columns={
                subj_col: "participant_id",
                sess_col: "session_id",
                slice_col: "slice_idx",
            }
        )

    @staticmethod
    def _load_tsv(path: Union[str, Path]) -> Dict[Tuple[str, str], List[int]]:
        df = pd.read_csv(path, sep="\t")
        df = Slice._normalize_cols(df)
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
        if self._map is not None:
            # TSV takes full precedence, ignore validation
            return self
        if (self.slices is not None) and (self.discarded_slices is not None):
            raise ValueError(
                "'slices' and 'discarded_slices' can't be passed simultaneously."
            )
        elif (self.slices is not None) and (self.borders is not None):
            raise ValueError("'slices' and 'borders' can't be passed simultaneously.")
        return self

    def extract_sample(self, data_point: DataPoint, sample_index: int) -> SliceSample:
        # store context for TSV lookup
        self._current_datapoint = data_point

        slice_tensor = self._extract_tensor_sample(
            data_point.image.tensor, sample_index
        )

        if self._map is not None:
            slice_position = self._slices_for(data_point)[sample_index]
        else:
            slice_position = self._get_slice_position(
                data_point.image.tensor, sample_index
            )

        extracted = self._extract_datapoint_sample(data_point, sample_index)
        sample = SliceSample(
            **extracted,
            extraction=self.extract_method,
            slice_position=slice_position,
            slice_direction=self.slice_direction,
            squeeze=self.squeeze,
        )
        sample.applied_transforms = extracted.applied_transforms
        return sample

    def num_samples_per_image(self, data_point: DataPoint) -> int:
        if self._map is not None:  # TSV mode
            slices = self._slices_for(data_point)
            n_slices = int(data_point.image.tensor.size(self.slice_direction + 1))
            bad = [p for p in slices if p < 0 or p >= n_slices]
            if bad:
                raise IndexError(
                    f"Invalid slice indices {bad} for {data_point.participant}, {data_point.session} "
                    f"(image has {n_slices} slices)."
                )
            return len(slices)
        return self._get_slice_selection(data_point.image.tensor).sum()

    def _extract_tensor_sample(
        self, image_tensor: torch.Tensor, sample_index: int
    ) -> torch.Tensor:
        if self._map is not None:
            if (
                not hasattr(self, "_current_datapoint")
                or self._current_datapoint is None
            ):
                raise RuntimeError("TSV mode requires a current DataPoint context.")
            slices = self._slices_for(self._current_datapoint)
            try:
                slice_position = int(slices[sample_index])
            except IndexError as exc:
                raise IndexError(
                    f"'sample_index' {sample_index} out of range: TSV lists {len(slices)} slices "
                    f"for {self._current_datapoint.participant}, {self._current_datapoint.session}."
                ) from exc
        else:
            slice_position = self._get_slice_position(image_tensor, sample_index)

        return self._get_slice(image_tensor, slice_position)

    def _get_slice_selection(self, image: torch.Tensor) -> np.ndarray:
        n_slices = image.size(self.slice_direction + 1)
        selection = np.ones(n_slices, dtype=bool)

        if self.slices:
            selection[:] = False
            try:
                selection[self.slices] = True
            except IndexError as exc:
                raise IndexError(
                    f"Invalid slices: image has 0..{n_slices - 1}, got {self.slices}."
                ) from exc
        else:
            if self.discarded_slices:
                try:
                    selection[self.discarded_slices] = False
                except IndexError as exc:
                    raise IndexError(
                        f"Invalid discarded_slices: image has 0..{n_slices - 1}, got {self.discarded_slices}."
                    ) from exc
            if self.borders:
                selection[: self.borders[0]] = False
                selection[n_slices - self.borders[1] :] = False

        return selection

    def _get_slice_position(self, image: torch.Tensor, slice_index: int) -> int:
        selection = self._get_slice_selection(image)
        slice_positions = np.arange(len(selection))[selection]
        try:
            return int(slice_positions[slice_index])
        except IndexError as exc:
            raise IndexError(
                f"'sample_index' {slice_index} is out of range (only {len(slice_positions)} selected slices)."
            ) from exc

    def _get_slice(self, image: torch.Tensor, slice_position: int) -> torch.Tensor:
        if self.slice_direction == 0:
            slice_tensor = image[:, slice_position, :, :]
        elif self.slice_direction == 1:
            slice_tensor = image[:, :, slice_position, :]
        elif self.slice_direction == 2:
            slice_tensor = image[:, :, :, slice_position]
        return slice_tensor.unsqueeze(self.slice_direction + 1)
