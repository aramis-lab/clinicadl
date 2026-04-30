from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import PositiveFloat, PositiveInt
from typing_extensions import Self

from clinicadl.io import Bids, BidsFileType, TensorType
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.types import TransformOrConfig
from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig
from clinicadl.utils.dictionary.suffixes import JSON, TSV
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.tsvtools import df_to_tsv, read_data
from clinicadl.utils.typing import DataFrameType, PathType

from ..structures import Tensor
from .bids_utils import (
    BidsTensorDataset,
    BidsTypeDatasetConfig,
    BidsTypeDatasetWithConfig,
    ColumnsType,
)


class TensorDatasetConfig(ObjectConfig["TensorDataset"], BidsTypeDatasetConfig):
    """Config class to check ``TensorDataset`` inputs."""

    description_json: Path
    to_load: Optional[Sequence[str]]

    @classmethod
    def _get_class(cls):
        return TensorDataset


class TensorDescription(ClinicaDLConfig):
    tensor_type: BidsFileType
    images: list[str]
    masks: list[str]
    additional_data: list[str]
    transforms: list[TransformOrConfig]
    spacing: Optional[tuple[PositiveFloat, PositiveFloat, PositiveFloat]]
    spatial_shape: Optional[tuple[PositiveInt, PositiveInt, PositiveInt]]
    interrupted: bool
    participants_sessions: pd.DataFrame

    def get_json_filename(self, tensor_dir: Path) -> Path:
        return self._get_filename(tensor_dir, suffix="description", extension=JSON)

    def get_df_filename(self, tensor_dir: Path) -> Path:
        return self._get_filename(
            tensor_dir, suffix="participantsXsessions", extension=TSV
        )

    def _get_filename(self, tensor_dir: Path, suffix: str, extension: str) -> Path:
        file_type = self.tensor_type.model_copy()
        file_type.extension = extension
        file_type.suffix = suffix
        file_type.data_type = None
        return Bids(tensor_dir).build_path(file_type)

    def write(self, tensor_dir: Path) -> None:
        super().to_json(
            self.get_json_filename(tensor_dir),
            exclude=["participants_sessions"],
            overwrite=True,
        )
        df_to_tsv(self.get_df_filename(tensor_dir), self.participants_sessions)

    @classmethod
    def read(cls, description_json: Path) -> Self:
        tsv = (
            str(description_json)
            .replace(JSON, TSV)
            .replace("_description.", "_participantsXsessions.")
        )
        df = read_data(tsv)
        return cls.from_json(
            description_json, participants_sessions=df, transforms=()
        )  # transforms may be impossible to read and is not needed

    @classmethod
    def _check_dict(cls, dict_: dict[str, Any]) -> dict[str, Any]:
        return dict_


class TensorDataset(
    BidsTensorDataset, HasConfig[TensorDatasetConfig], BidsTypeDatasetWithConfig
):
    """
    Careful with n_samples in columns.
    """

    _config_type = TensorDatasetConfig

    def __init__(
        self,
        description_json: PathType,
        data: Optional[DataFrameType] = None,
        transforms: TransformsHandler = TransformsHandler(),
        columns: Optional[ColumnsType] = None,
        to_load: Optional[Sequence[str]] = None,
    ):
        self.config = self._config_type(
            description_json=description_json,
            data=data,
            transforms=transforms,
            columns=columns,
            to_load=to_load,
        )
        tensors = TensorDescription.read(self.config.description_json)
        super().__init__(
            tensor=Tensor(
                self.config.description_json.parent,
                tensors.tensor_type,
                to_load=self.config.to_load,
            ),
            data=self.config.data,
            transforms=self.config.transforms,
            columns=self.config.columns,
        )
