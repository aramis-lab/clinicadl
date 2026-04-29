from typing import Optional, TypeVar

from clinicadl.transforms import TransformsHandler
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
from .utils import ColumnType, MultimodalSamplerDataset


class BidsTypeDataset(MultimodalSamplerDataset):
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


T = TypeVar("T")


class BidsNiftiDataset(BidsTypeDataset):
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


class BidsTensorDataset(BidsTypeDataset):
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
