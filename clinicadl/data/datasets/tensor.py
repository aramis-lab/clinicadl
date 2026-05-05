from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from clinicadl.io import Bids
from clinicadl.transforms import TransformsHandler
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.typing import DataFrameType, PathType

from ..structures import Tensor
from ..tensors import TensorDescription
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
                Bids(self.config.description_json.parent),
                tensors.tensor_type,
                to_load=self.config.to_load,
            ),
            data=self.config.data,
            transforms=self.config.transforms,
            columns=self.config.columns,
        )
