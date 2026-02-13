from __future__ import annotations

from collections.abc import Sequence
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pydantic import (
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import (
    SAMPLE_POSITION,
    SAMPLE_TYPE,
    SLICE_DIRECTION,
    SQUEEZE,
)
from clinicadl.utils.enum import SliceDirection
from clinicadl.utils.exceptions import ClinicaDLTSVError
from clinicadl.utils.typing import PathType

from .base import Extraction, ImplementedExtraction

if TYPE_CHECKING:
    from clinicadl.data.structures import DataPoint

logger = getLogger(__name__)


class SliceConfig(ObjectConfig["Slice"]):
    """
    Config class for slice extraction.
    """

    slices: Optional[list[NonNegativeInt]]
    tsv_path: Optional[Path]
    discarded_slices: Optional[list[NonNegativeInt]]
    borders: Optional[Tuple[PositiveInt, PositiveInt]]
    slice_direction: SliceDirection
    squeeze: bool

    @field_validator("borders", mode="before")
    @classmethod
    def _ensure_tuple(
        cls,
        value: Optional[Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]],
    ) -> Optional[Tuple[PositiveInt, PositiveInt]]:
        """
        Ensures that 'borders' is always a tuple.
        """
        if not isinstance(value, Sequence) and value is not None:
            return (value, value)
        return value

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

    @classmethod
    def _get_class(cls) -> type[Slice]:
        """Returns the class associated to this config class."""
        return Slice


class Slice(Extraction[SliceConfig]):
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
    slices : Optional[list[NonNegativeInt]], default=None
        The slices to select. The slices selected will be the same for all images; if you
        want a different selection for each image, use ``tsv_path``.
    tsv_path : Optional[PathType], default=None
        Path to a ``TSV`` file containing slice indices for each image.
        The ``TSV`` table must have the columns: ``participant_id``, ``session_id``, and ``slice_idx``.
    discarded_slices : Optional[list[NonNegativeInt]], default=None
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

    config: SliceConfig
    _config_type = SliceConfig

    def __init__(
        self,
        slices: Optional[list[int]] = None,
        tsv_path: Optional[PathType] = None,
        discarded_slices: Optional[list[int]] = None,
        borders: Optional[Union[int, Tuple[int, int]]] = None,
        slice_direction: SliceDirection = SliceDirection.SAGITTAL,
        squeeze: bool = True,
    ) -> None:
        self.config = SliceConfig(
            slices=slices,
            tsv_path=tsv_path,
            discarded_slices=discarded_slices,
            borders=borders,
            slice_direction=slice_direction,
            squeeze=squeeze,
        )
        self._map: Optional[Dict[Tuple[str, str], list[int]]] = None
        if self.config.tsv_path is not None:
            self._map = self._load_tsv(self.config.tsv_path)

    @property
    def sample_type(self) -> str:
        """
        The type of the sample returned by this extraction, among {"image", "slice", "patch"}.
        """
        return ImplementedExtraction.SLICE.value.lower()

    def _extract_tensor_sample(
        self, image_tensor: torch.Tensor, sample_position: int
    ) -> torch.Tensor:
        """
        Gets the wanted slice, according to the slicing direction.
        """
        if self.config.slice_direction == 0:
            slice_tensor = image_tensor[:, sample_position, :, :]
        elif self.config.slice_direction == 1:
            slice_tensor = image_tensor[:, :, sample_position, :]
        elif self.config.slice_direction == 2:
            slice_tensor = image_tensor[:, :, :, sample_position]

        return slice_tensor.unsqueeze(self.config.slice_direction + 1)  # pylint: disable=possibly-used-before-assignment

    def _get_sample_positions(self, data_point: DataPoint) -> list[int]:
        """
        Returns the positions of the selected slices in the image.
        """
        n_slices = data_point.image.tensor.size(self.config.slice_direction + 1)
        selection = np.ones(n_slices).astype(bool)

        slice_indices = None
        if self._map:
            slice_indices = self._slices_for(data_point)
        elif self.config.slices:
            slice_indices = self.config.slices

        if slice_indices:
            selection = ~selection
            try:
                selection[slice_indices] = True
            except IndexError as exc:
                raise IndexError(
                    "Invalid slices in 'slices': "
                    f"slices in the image are indexed from 0 to {n_slices - 1}, but got "
                    f"slices={self.config.slices}."
                ) from exc
        else:
            if self.config.discarded_slices:
                try:
                    selection[self.config.discarded_slices] = False
                except IndexError as exc:
                    raise IndexError(
                        "Invalid slices in 'discarded_slices': "
                        f"slices in the image are indexed from 0 to {n_slices - 1}, but got "
                        f"discarded_slices={self.config.discarded_slices}."
                    ) from exc

            if self.config.borders:
                selection[: self.config.borders[0]] = False
                selection[n_slices - self.config.borders[1] :] = False

        return np.arange(len(selection))[selection]

    def _add_info(self, data_point: DataPoint, sample_position: int) -> None:
        """
        Adds relevant info in the datapoint.
        """
        data_point[SAMPLE_TYPE] = self.sample_type
        data_point[SAMPLE_POSITION] = sample_position
        data_point[SLICE_DIRECTION] = self.config.slice_direction
        data_point[SQUEEZE] = self.config.squeeze

    @classmethod
    def _load_tsv(cls, path: Path) -> Dict[Tuple[str, str], list[int]]:
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

        mapping: Dict[Tuple[str, str], list[int]] = {}
        for (sub, ses), g in df.groupby(["participant_id", "session_id"]):
            mapping[(str(sub), str(ses))] = list(map(int, g["slice_idx"].tolist()))

        return mapping

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

    def _slices_for(self, data_point: DataPoint) -> list[int]:
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
