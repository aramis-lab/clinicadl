from collections.abc import Sequence
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Iterable, Optional, TypeAlias

import torchio as tio
from pydantic import Field, field_validator

from clinicadl.io import Bids, BidsFileType
from clinicadl.transforms import TransformsHandler
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.typing import DataFrameType, PathType

from ..structures import (
    CommonMask,
    Image,
    IndividualMask,
)
from ..tensors import TensorConversion
from ..utils import DEFAULT_SPATIAL_CHECKS, SpatialCheck
from .bids_utils import (
    BidsNiftiDataset,
    BidsTypeDatasetConfig,
    BidsTypeDatasetWithConfig,
    ColumnsType,
)
from .tensor import TensorDataset

MasksType: TypeAlias = dict[str, PathType | BidsFileType | tuple[Bids, BidsFileType]]


def _deserialize_masks(serialized_masks: Optional[dict]) -> Optional[MasksType]:
    """
    To read serialized masks.
    """
    if serialized_masks is None:
        return None

    masks = dict()
    for name, mask in serialized_masks.items():
        if isinstance(mask, dict):
            masks[name] = BidsFileType.from_dict(mask)
        elif isinstance(mask, Sequence) and not isinstance(mask, str):
            masks[name] = (Bids.from_dict(mask[0]), BidsFileType.from_dict(mask[1]))
        else:
            masks[name] = mask

    return masks


class BidsDatasetConfig(ObjectConfig["BidsDataset"], BidsTypeDatasetConfig):
    """Config class to check ``BidsDataset`` inputs."""

    bids: Bids = Field(reader=Bids.from_dict)
    file_type: BidsFileType = Field(reader=BidsFileType.from_dict)
    masks: Optional[dict[str, Path | BidsFileType | tuple[Bids, BidsFileType]]] = Field(
        reader=_deserialize_masks
    )

    @field_validator("bids", mode="before")
    @classmethod
    def _convert_to_bids(cls, v: Any) -> Any:
        """
        Convert a path to a ``Bids``.
        """
        if isinstance(v, (str, Path)):
            return Bids(v)
        return v

    @classmethod
    def _get_class(cls):
        return BidsDataset


class BidsDataset(
    BidsNiftiDataset, HasConfig[BidsDatasetConfig], BidsTypeDatasetWithConfig
):
    """
    Careful with n_samples in columns.
    """

    _config_type = BidsDatasetConfig

    def __init__(
        self,
        bids: PathType | Bids,
        file_type: BidsFileType,
        data: Optional[DataFrameType] = None,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
        masks: Optional[
            dict[str, PathType | BidsFileType | tuple[Bids, BidsFileType]]
        ] = None,
    ):
        self.config = self._config_type(
            bids=bids,
            file_type=file_type,
            data=data,
            transforms=transforms,
            columns=columns,
            masks=masks,
        )
        super().__init__(
            image=Image(self.config.bids, self.config.file_type),
            data=self.config.data,
            transforms=self.config.transforms,
            columns=self.config.columns,
            masks=self._read_masks(copy(self.config.masks)),
        )

    def _read_masks(
        self,
        masks: Optional[dict[str, PathType | BidsFileType | tuple[Bids, BidsFileType]]],
    ) -> Optional[dict[str, IndividualMask | CommonMask]]:
        """
        Converts masks to the right format.
        """
        if not masks:
            return None

        for name, mask in masks.items():
            if isinstance(mask, BidsFileType):
                masks[name] = IndividualMask(self.config.bids, mask)
            elif isinstance(mask, tuple):
                masks[name] = IndividualMask(mask[0], mask[1])
            else:
                masks[name] = CommonMask(mask)

        return masks

    def to_tensors(
        self,
        conversion_name: Optional[str] = None,
        spatial_checks: Optional[Iterable[str | SpatialCheck]] = DEFAULT_SPATIAL_CHECKS,
        save_transforms: bool = False,
        description: Optional[str] = None,
        overwrite: bool = False,
        check_transforms: bool = True,
        n_proc: int = 1,
    ) -> TensorDataset:
        converter = TensorConversion(self)
        conversion = converter.to_tensors(
            conversion_name=conversion_name,
            spatial_checks=spatial_checks,
            save_transforms=save_transforms,
            description=description,
            overwrite=overwrite,
            check_transforms=check_transforms,
            n_proc=n_proc,
        )
        transforms = deepcopy(self.transforms)
        if conversion.transforms:
            transforms.image_transforms = tio.Compose([])

        return TensorDataset(
            conversion.get_json_path(converter.tensors_dir.path),
            data=copy(self.df),
            transforms=transforms,
            columns=copy(self.columns),
            to_load=list(conversion.masks.keys()) + conversion.additional_data,
        )
