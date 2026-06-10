from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, TypeVar

import pandas as pd
from pydantic import Field, field_serializer

from clinicadl.transforms import TransformsHandler
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.tsvtools import create_participants_sessions_df
from clinicadl.utils.typing import DataFrameType

from ..structures import (
    CommonMask,
    DataPoint,
    IndividualMask,
)
from .utils import (
    CheckableDataset,
    ColumnsType,
    MultimodalSamplerDataset,
)

if TYPE_CHECKING:
    from ..structures.images import Image, SubjectSpecificImage, Tensor


class _BidsTypeDataset(MultimodalSamplerDataset, CheckableDataset):
    """
    A :py:class:`clinicadl.data.datasets.utils.MultimodalSamplerDataset` that is
    able to read in a :term:`BIDS` directory to load images.
    """

    def __init__(
        self,
        image: SubjectSpecificImage,
        data: Optional[DataFrameType],
        transforms: TransformsHandler,
        columns: Optional[ColumnsType],
    ):
        self.image = image
        data = self._create_df(data)

        super().__init__(data, transforms, columns)

        self._look_for_images()

    def describe(self) -> dict[str, Any]:
        """
        Returns a description of the dataset.

        Returns
        -------
        dict[str, Any]
            A dictionary describing the dataset.
        """
        return {
            "participant_session_pairs": self.get_participant_session_couples(),
            "file_type": dict(self.image.file_type.to_dict()),
            "extraction": dict(self.transforms.extraction.to_dict()),
            "total_number_samples": len(self),
        }

    def _create_df(self, data: Optional[DataFrameType]) -> DataFrameType:
        """
        If no DataFrame-like is passed, it creates it by looking for all the
        (participant, session) couples that have the wanted image.
        """
        if data is not None:
            return data

        participants_sessions = self.image.bids.get_participants_sessions_with(
            self.image.file_type
        )
        if len(participants_sessions) == 0:
            raise RuntimeError(
                f"No image found in {self.image.bids.path} for {self.image.file_type}"
            )

        return create_participants_sessions_df(participants_sessions)

    def _look_for_images(self) -> None:
        """
        Checks that all the (participant, session) pairs in the dataset have
        the wanted image.
        """
        for participant, session in self.get_participant_session_couples():
            assert self.image.bids.has_file_type(
                participant, session, self.image.file_type
            ), f"For ({participant}, {session}), no image associated with {self.image.file_type}"


def _read_df(dict_df: dict) -> pd.DataFrame:
    """
    To read a DataFrame from a JSON file.
    """
    df = pd.DataFrame.from_dict(dict_df)
    try:
        df.index = df.index.astype(int)
    except (ValueError, TypeError):
        pass
    return df


def _read_columns(
    columns: Sequence[str] | dict[str, Any] | None,
) -> Sequence[str] | None:
    """
    To read columns. No need for column processing functions here
    as the DataFrame saved has already been processed.
    """
    if isinstance(columns, dict):
        return list(columns.keys())
    return columns


class BidsTypeDatasetConfig(ClinicaDLConfig):
    """
    Base config class for datasets that inherit from ``_BidsTypeDataset``.
    """

    data: Optional[DataFrameType] = Field(json_schema_extra={"reader": _read_df})
    transforms: TransformsHandler = Field(
        json_schema_extra={"reader": TransformsHandler.from_dict}
    )
    columns: Optional[ColumnsType] = Field(json_schema_extra={"reader": _read_columns})

    @field_serializer("data")
    def _serialize_df(self, df: pd.DataFrame) -> dict:
        return df.to_dict()

    def __eq__(self, other: Any) -> bool:
        if self.__class__ is not other.__class__:
            return NotImplemented
        for field in self.__class__.model_fields.keys():
            v1, v2 = getattr(self, field), getattr(other, field)
            if isinstance(v1, pd.DataFrame) and isinstance(v2, pd.DataFrame):
                if not v1.equals(v2):
                    return False
            elif v1 != v2:
                return False
        return True


class BidsTypeDatasetWithConfig(_BidsTypeDataset):
    """
    To synchronize the DataFrame in the dataset and the one
    the config class.
    """

    config: BidsTypeDatasetConfig

    @property
    def _df(self) -> pd.DataFrame:
        return self.config.data

    @_df.setter
    def _df(self, df: pd.DataFrame) -> None:
        self.config.data = df


T = TypeVar("T")


class BidsNiftiDataset(_BidsTypeDataset):
    """
    A :py:class:`BidsTypeDataset` that reads :term:`NIfTI` images.
    """

    image: Image

    def __init__(
        self,
        image: Image,
        data: Optional[DataFrameType],
        transforms: TransformsHandler,
        columns: Optional[ColumnsType],
        masks: Optional[dict[str, IndividualMask | CommonMask]],
    ):
        super().__init__(image, data, transforms, columns)
        self.individual_masks, self.common_masks = _differentiate_masks(
            self._validate_masks(masks)
        )
        self._look_for_masks()

    def _validate_masks(
        self,
        masks: Optional[dict[str, T]],
    ) -> dict[str, T]:
        """
        Checks mask inputs.
        """
        masks = self._check_keys(masks or {}, "mask")
        for mask in masks:
            if mask in self.columns:
                raise ValueError(f"'{mask}' is in columns and masks!")

        return masks

    def _look_for_masks(self) -> None:
        """
        Checks that all the (participant, session) pairs in the dataset have
        the wanted masks.
        """
        for mask in self.individual_masks.values():
            for participant, session in self.get_participant_session_couples():
                assert mask.bids.has_file_type(
                    participant, session, mask.file_type
                ), f"For ({participant}, {session}), no mask associated with {mask.file_type}"

        for mask in self.common_masks.values():
            assert mask.file.path.exists(), f"Cannot find the mask in {mask.file.path}"

    def _get_images(self, participant_id: str, session_id: str) -> DataPoint:
        img = self.image.get(participant_id, session_id)
        masks = {
            name: mask.get(participant_id, session_id)
            for name, mask in self.individual_masks.items()
        }
        masks.update({name: mask.get() for name, mask in self.common_masks.items()})

        return DataPoint(
            image=img,
            participant_id=participant_id,
            session_id=session_id,
            image_path=img.path,
            file_type=self.image.file_type,
            **masks,
        )


def _differentiate_masks(
    masks: dict[str, IndividualMask | CommonMask],
) -> tuple[dict[str, IndividualMask], dict[str, CommonMask]]:
    """
    Separates image-specific and common masks.
    """
    individual_masks = {
        name: mask for name, mask in masks.items() if isinstance(mask, IndividualMask)
    }
    common_masks = {
        name: mask for name, mask in masks.items() if isinstance(mask, CommonMask)
    }

    return individual_masks, common_masks


class BidsTensorDataset(_BidsTypeDataset):
    """
    A :py:class:`BidsTypeDataset` that reads images saved as tensors in ``.pt`` files.
    """

    image: Tensor

    def __init__(
        self,
        tensor: Tensor,
        data: Optional[DataFrameType],
        transforms: TransformsHandler,
        columns: Optional[ColumnsType],
    ):
        super().__init__(
            tensor,
            data=data,
            transforms=transforms,
            columns=columns,
        )

    def _get_images(self, participant_id: str, session_id: str) -> DataPoint:
        return self.image.get(participant_id, session_id)
