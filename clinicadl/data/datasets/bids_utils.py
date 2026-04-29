from typing import Any, Iterable, Optional, TypeVar

import pandas as pd
from pydantic import Field, field_serializer

from clinicadl.transforms import TransformsHandler
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.tsvtools import create_participants_sessions_df
from clinicadl.utils.typing import DataFrameType

from ..structures import (
    CommonMask,
    DataPoint,
    Image,
    IndividualMask,
    Tensor,
)
from ..structures.images import SubjectSpecificImage
from .utils import ColumnType, DatasetChecker, MultimodalSamplerDataset, SpatialCheck


class _BidsTypeDataset(MultimodalSamplerDataset):
    """
    A :py:class:`clinicadl.data.datasets.utils.MultimodalSamplerDataset` that is
    able to read in a :term:`BIDS` directory to load images.
    """

    def __init__(
        self,
        image: SubjectSpecificImage,
        data: Optional[DataFrameType],
        transforms: TransformsHandler,
        columns: Optional[ColumnType],
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

    def sanity_check(
        self,
        spatial_checks: Optional[Iterable[str | SpatialCheck]] = [
            "affine",
            "shape",
            "global_spacing",
        ],
    ) -> None:
        """
        Performs a sanity check on the current dataset.

        It will iterate over the whole dataset to check if images are loaded correctly,
        and potentially perform spatial checks on the loaded images.

        Parameters
        ----------
        spatial_checks : Optional[Iterable[str  |  SpatialCheck]], default=[ "affine", "shape", "global_spacing"]
            Spatial checks to perform on the images:

            - ``"spacing"``: checks **intra-sample voxel spacing consistency**, i.e. that all the images and masks
              in a :py:class:`~clinicadl.data.structures.Sample` have the same voxel spacing.
            - ``"affine"``: checks **intra-sample affine matrix consistency** (so it includes ``"spacing"``).
            - ``"shape"``: checks **intra-sample spatial shape consistency**.
            - ``"global_spacing"``: checks **inter-sample voxel spacing consistency**, i.e. that all the ``Samples``
              in the dataset have the same voxel spacing (so it includes ``"spacing"``).
            - "``global_shape"``: checks **inter-sample spatial shape consistency** (so it includes ``"shape"``).

            If ``None``, no spatial check.
        """
        DatasetChecker(spatial_checks).check(self)

    def _create_df(self, data: Optional[DataFrameType]) -> DataFrameType:
        """
        If no DataFrame-like is passed, it creates it by looking for all the
        (participant, session) couples that have the wanted image.
        """
        if data is not None:
            return data

        participants_sessions = self.image.bids.get_participants_sessions(
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
    finally:
        return df


class BidsTypeDatasetConfig(ClinicaDLConfig):
    """
    Base config class for datasets that inherit from ``_BidsTypeDataset``.
    """

    data: Optional[DataFrameType] = Field(reader=_read_df)
    transforms: TransformsHandler = Field(reader=TransformsHandler.from_dict)
    columns: Optional[ColumnType]

    @field_serializer("data")
    def _serialize_df(self, df: pd.DataFrame) -> dict:
        return df.to_dict()


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
    A :py:class:`BidsTypeDataset` that reads NIfTI images.
    """

    image: Image

    def __init__(
        self,
        image: Image,
        data: Optional[DataFrameType],
        transforms: TransformsHandler,
        columns: ColumnType,
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

    def _get_images(self, participant: str, session: str) -> DataPoint:
        img = self.image.get(participant, session)
        masks = {
            name: mask.get(participant, session)
            for name, mask in self.individual_masks.items()
        }
        masks.update({name: mask.get() for name, mask in self.common_masks.items()})

        return DataPoint(
            image=img,
            participant=participant,
            session=session,
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
        columns: ColumnType,
    ):
        super().__init__(
            tensor,
            data=data,
            transforms=transforms,
            columns=columns,
        )

    def _get_images(self, participant: str, session: str) -> DataPoint:
        return self.image.get(participant, session)
