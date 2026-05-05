from logging import getLogger
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import Field, PositiveFloat, PositiveInt, field_serializer
from typing_extensions import Self

from clinicadl.io import Bids, BidsFileType
from clinicadl.transforms.config import TransformConfig
from clinicadl.transforms.factory import get_transform_from_dict
from clinicadl.transforms.types import Transform
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.dictionary.suffixes import JSON, TSV
from clinicadl.utils.names import camel_to_snake, snake_to_camel
from clinicadl.utils.tsvtools import df_to_tsv, read_data

from ..structures import CommonMask, Image, IndividualMask

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


def _read_images(images: dict[str, tuple[str, dict[str, Any]]]) -> dict[str, Image]:
    return {
        name: Image(Bids(value[0]), BidsFileType(**value[1]))
        for name, value in images.items()
    }


def _read_masks(
    masks: dict[str, tuple[str, dict[str, Any]] | str],
) -> dict[str, IndividualMask | CommonMask]:
    return {
        name: CommonMask(value)
        if isinstance(value, str)
        else IndividualMask(Bids(value[0]), BidsFileType(**value[1]))
        for name, value in masks.items()
    }


class TensorDescription(ClinicaDLConfig):
    """
    A dataclass containing the description of a tensor conversion.
    """

    tensor_type: BidsFileType
    images: dict[str, Image] = Field(reader=_read_images)
    masks: dict[str, IndividualMask | CommonMask] = Field(reader=_read_masks)
    additional_data: list[str]
    transforms: list[str | Transform | TransformConfig] = Field(
        reader=lambda x: list(map(_read_transform, x))
    )  # str: to be able to read transforms serialized as a string
    spacing: Optional[tuple[PositiveFloat, PositiveFloat, PositiveFloat]]
    spatial_shape: Optional[tuple[PositiveInt, PositiveInt, PositiveInt]]
    interrupted: bool
    participants_sessions: pd.DataFrame

    def get_json_filename(self, tensor_dir: Path) -> Path:
        """
        Gets the path to the ``.json`` description file.

        Parameters
        ----------
        tensor_dir : Path
            The BIDS derivative where the tensors are saved.

        Returns
        -------
        Path
            The path to the ``.json`` file.
        """
        return self._get_filename(tensor_dir, suffix="description", extension=JSON)

    def get_df_filename(self, tensor_dir: Path) -> Path:
        """
        Gets the path to the ``.tsv`` file containing the (participant, session) couples
        converted.

        Parameters
        ----------
        tensor_dir : Path
            The BIDS derivative where the tensors are saved.

        Returns
        -------
        Path
            The path to the ``.tsv`` file.
        """
        return self._get_filename(
            tensor_dir, suffix="participantsXsessions", extension=TSV
        )

    def _get_filename(self, tensor_dir: Path, suffix: str, extension: str) -> Path:
        file_type = self.tensor_type.model_copy()
        file_type.extension = extension
        file_type.suffix = suffix
        file_type.data_type = None
        return Bids(tensor_dir).build_path(file_type)

    @field_serializer("images")
    def _serialize_images(
        self, images: dict[str, Image]
    ) -> dict[str, tuple[str, dict[str, Any]]]:
        """
        To convert Images to tuples (bids_path, file_type).
        """
        return {
            name: (value.bids.path, value.file_type.to_dict())
            for name, value in images.items()
        }

    @field_serializer("masks")
    def _serialize_masks(
        self, images: dict[str, IndividualMask | CommonMask]
    ) -> dict[str, tuple[str, dict[str, Any]] | str]:
        """
        To convert IndividualMasks to tuples (bids_path, file_type) and CommonMasks to str (the path of the mask).
        """
        return {
            name: (value.bids.path, value.file_type.to_dict())
            if isinstance(value, IndividualMask)
            else value.file.path.resolve()
            for name, value in images.items()
        }

    def to_dict(self, **kwargs):
        dict_ = super().to_dict(**kwargs)
        return {snake_to_camel(name): value for name, value in dict_.items()}

    @classmethod
    def from_dict(cls, dict_, **kwargs):
        return super().from_dict(
            {camel_to_snake(name): value for name, value in dict_.items()}, **kwargs
        )

    def write(self, tensor_dir: Path) -> None:
        """
        Writes the description of the conversion in a ``.json`` file and the
        (participant, session) couples whose images have been converted in
        ``.tsv`` file.

        Parameters
        ----------
        tensor_dir : Path
            The :BIDS derivative where the tensors are saved.
        """
        self.to_json(
            json_file := self.get_json_filename(tensor_dir),
            exclude=["participants_sessions"],
            overwrite=True,
        )
        logger.info("Tensor conversion description saved in %s", str(json_file))
        df_to_tsv(
            tsv_file := self.get_df_filename(tensor_dir), self.participants_sessions
        )
        logger.info("(participant, session) pairs converted saved in %s", str(tsv_file))

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
        df = read_data(tsv)
        return cls.from_json(
            description_json, participants_sessions=df
        )  # transforms may be impossible to read and is not needed

    @classmethod
    def _check_dict(
        cls, dict_: dict[str, Any]
    ) -> dict[
        str, Any
    ]:  # not to raise error because 'participants_sessions' is missing in json file
        return dict_
