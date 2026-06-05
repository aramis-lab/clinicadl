from dataclasses import asdict, dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt, field_validator
from typing_extensions import Self

from clinicadl.io.bids import Bids, BidsFileType
from clinicadl.transforms.config import TransformConfig
from clinicadl.transforms.factory import get_transform_from_dict
from clinicadl.transforms.types import Transform
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.dictionary.suffixes import JSON, TSV
from clinicadl.utils.dictionary.utils import TSV_SEP
from clinicadl.utils.names import camel_to_snake, snake_to_camel
from clinicadl.utils.tsvtools import df_to_tsv, read_df

logger = getLogger(__name__)


def _read_transform(
    serialized: str | dict[str, Any],
) -> str | TransformConfig:
    """
    Handles deserialization of transforms.
    """
    if isinstance(serialized, dict):
        return get_transform_from_dict(serialized)
    return serialized


@dataclass
class ConversionRow:
    """
    A row of the conversions.tsv
    """

    conv_id: str
    description: str
    description_json: str


class TensorDescription(ClinicaDLConfig):
    """
    A dataclass containing the description of a tensor conversion.
    """

    tensor_type: BidsFileType
    image: tuple[Path, BidsFileType]
    masks: dict[str, tuple[Path, BidsFileType] | Path]
    additional_data: list[str]
    transforms: list[str | Transform | TransformConfig] = Field(
        reader=lambda x: list(map(_read_transform, x))
    )  # str: to be able to read transforms serialized as a string
    spacing: Optional[tuple[PositiveFloat, PositiveFloat, PositiveFloat]]
    spatial_shape: Optional[tuple[PositiveInt, PositiveInt, PositiveInt]]
    interrupted: bool
    description: Optional[str]
    participants_sessions: pd.DataFrame

    @field_validator("image", mode="after")
    @classmethod
    def _resolve_image_path(
        cls, v: tuple[Path, BidsFileType]
    ) -> tuple[Path, BidsFileType]:
        return v[0].resolve(), v[1]

    @field_validator("masks", mode="after")
    @classmethod
    def _resolve_masks_path(
        cls, v: dict[str, tuple[Path, BidsFileType] | Path]
    ) -> dict[str, tuple[Path, BidsFileType] | Path]:
        for key, value in v.items():
            if isinstance(value, Path):
                v[key] = value.resolve()
            else:
                v[key] = value[0].resolve(), value[1]

        return v

    @staticmethod
    def get_conversions_tsv_path(tensors_dir: Path) -> Path:
        """
        Gets the path to the ``.tsv`` file enumerating all the conversions.

        Parameters
        ----------
        tensors_dir : Path
            The BIDS derivative where the tensors are saved.

        Returns
        -------
        Path
            The path to the ``.tsv`` file.
        """
        return tensors_dir / "conversions.tsv"

    def get_json_path(self, tensors_dir: Path) -> Path:
        """
        Gets the path to the ``.json`` description file.

        Parameters
        ----------
        tensors_dir : Path
            The BIDS derivative where the tensors are saved.

        Returns
        -------
        Path
            The path to the ``.json`` file.
        """
        return self._get_filename(tensors_dir, suffix="description", extension=JSON)

    def get_tsv_path(self, tensors_dir: Path) -> Path:
        """
        Gets the path to the ``.tsv`` file containing the (participant, session) couples
        converted.

        Parameters
        ----------
        tensors_dir : Path
            The BIDS derivative where the tensors are saved.

        Returns
        -------
        Path
            The path to the ``.tsv`` file.
        """
        return self._get_filename(
            tensors_dir, suffix="participantsXsessions", extension=TSV
        )

    def _get_filename(self, tensors_dir: Path, suffix: str, extension: str) -> Path:
        file_type = self.tensor_type.model_copy()
        file_type.extension = extension
        file_type.suffix = suffix
        file_type.data_type = None

        return Bids(tensors_dir).build_path(file_type)

    def to_dict(self, **kwargs):
        dict_ = super().to_dict(**kwargs)
        return {snake_to_camel(name): value for name, value in dict_.items()}

    @classmethod
    def from_dict(cls, dict_, **kwargs):
        return super().from_dict(
            {camel_to_snake(name): value for name, value in dict_.items()}, **kwargs
        )

    def write(self, tensors_dir: Path) -> None:
        """
        Writes the description of the conversion in a ``.json`` file and the
        (participant, session) couples whose images have been converted in
        ``.tsv`` file.

        Parameters
        ----------
        tensors_dir : Path
            The BIDS derivative where the tensors are saved.
        """
        self.to_json(
            json_file := self.get_json_path(tensors_dir),
            exclude=["participants_sessions"],
            overwrite=True,
        )
        logger.info("Tensor conversion description saved in %s", str(json_file))
        df_to_tsv(
            tsv_file := self.get_tsv_path(tensors_dir), self.participants_sessions
        )
        logger.info("(participant, session) pairs converted saved in %s", str(tsv_file))
        self._update_conversions_tsv(tensors_dir)

    @classmethod
    def read(cls, description_json: Path) -> Self:
        """
        Loads info on a tensor conversion.

        This function will read ``.json`` and ``.tsv`` files saved with :py:meth:`write`.

        Parameters
        ----------
        description_json : Path
            The ``.json`` file describing the tensor conversion.

        Returns
        -------
        Self
            A ``TensorDescription`` object associated to the tensor conversion.
        """
        tsv = (
            str(description_json)
            .replace(JSON, TSV)
            .replace("_description.", "_participantsXsessions.")
        )
        df = read_df(tsv)
        return cls.from_json(
            description_json, participants_sessions=df
        )  # transforms may be impossible to read and is not needed

    def _update_conversions_tsv(self, tensors_dir: Path) -> None:
        """
        Adds the conversion in the conversions.tsv file.
        """
        tsv_path = self.get_conversions_tsv_path(tensors_dir)

        if not tsv_path.exists():
            df = pd.DataFrame()
        else:
            df = pd.read_csv(tsv_path, sep=TSV_SEP)

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        asdict(
                            ConversionRow(
                                conv_id=self.tensor_type.with_entities["conv"].pattern,
                                description=self.description,
                                description_json=self.get_tsv_path(tensors_dir).name,
                            )
                        )
                    ],
                ),
            ]
        )
        df.to_csv(tsv_path, sep=TSV_SEP, index=False)

    @classmethod
    def _check_dict(
        cls, dict_: dict[str, Any]
    ) -> dict[
        str, Any
    ]:  # not to raise error because 'participants_sessions' is missing in json file
        return dict_
