from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
)

import pandas as pd
import torchio as tio
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from clinicadl.transforms.handlers import Transforms
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.dictionary.words import (
    AFFINE,
    DF,
    IMAGE,
    LABEL,
    PARTICIPANT,
    PARTICIPANT_ID,
    SESSION,
    SESSION_ID,
)
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.tsvtools import read_data
from clinicadl.utils.typing import DataFrameType, PathType

from ..datatypes import DataType
from ..datatypes.factory import get_datatype_from_dict
from ..structures import Column, DataPoint, Mask, Sample
from .sampler import SamplerDataset

logger = getLogger("clinicadl.data.datasets.base")

T = TypeVar("T")


def _dataframe_from_dict(serialized_df: Union[dict, Any]) -> None:
    """
    To deserialized DataFrames.
    """
    if isinstance(serialized_df, dict):
        return pd.DataFrame.from_dict(serialized_df)
    return serialized_df


class BaseDatasetConfig(ObjectConfig["BaseDataset"]):
    """Config class to check ``BaseDataset`` inputs."""

    directory: Path
    datatype: DataType = Field(reader=get_datatype_from_dict)
    data: Optional[DataFrameType] = Field(reader=_dataframe_from_dict)
    label: Optional[Union[str, list[str]]]
    transforms: Transforms = Field(reader=Transforms.from_dict)
    columns: dict[str, Optional[Callable[[pd.Series], pd.Series]]]
    masks: list[PathType]

    # state
    df: Optional[pd.DataFrame] = Field(default=None, reader=_dataframe_from_dict)

    _individual_masks: list[PathType] = []
    _common_masks: list[PathType] = []

    @property
    def _columns_names(self) -> list[str]:
        return list(self.columns.keys())

    @property
    def _individual_mask_names(self) -> list[str]:
        return self._individual_masks

    @property
    def _common_mask_names(self) -> list[str]:
        return list(map(Mask.get_mask_name, self._common_masks))

    @field_validator("label", mode="after")
    @classmethod
    def _sort_labels(cls, labels: T) -> T:
        """Sort the labels if list."""
        if isinstance(labels, list):
            return sorted(labels)
        return labels

    @field_validator("columns", mode="before")
    @classmethod
    def _uniformize_columns(
        cls,
        columns: Optional[
            Union[Sequence[str], dict[str, Optional[Callable[[pd.Series], pd.Series]]]]
        ],
    ) -> dict[str, Optional[Callable[[pd.Series], pd.Series]]]:
        """
        Return 'columns' as a dict, no matter the input.
        """
        if columns is None:
            return dict()

        if isinstance(columns, Sequence):
            return {col: None for col in columns}

        return columns

    @field_validator("columns", mode="after")
    @classmethod
    def _check_columns(
        cls,
        columns: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Checks column names.
        """
        for col in columns:
            if col in {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION}:
                raise ValueError(
                    f"A column cannot be named '{col}'. {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION} "
                    "are protected names."
                )

        return columns

    @field_validator("masks", mode="before")
    @classmethod
    def _check_masks(
        cls, masks: Optional[list[Union[str, PathType]]]
    ) -> list[Union[str, PathType]]:
        """
        Checks that there are no duplicates in the mask, and that no
        protected names are used.
        """
        if masks is None:
            return []

        for mask in masks:
            if mask in {
                IMAGE,
                LABEL,
                AFFINE,
                PARTICIPANT,
                SESSION,
            }:
                raise ValueError(
                    f"Mask cannot be named '{mask}'. {IMAGE, LABEL, AFFINE, PARTICIPANT, SESSION} "
                    "are protected names."
                )

        mask_names = list(map(Mask.get_mask_name, masks))
        if len(mask_names) != len(set(mask_names)):
            raise ValueError(
                f"Duplicated mask names in 'masks' (got masks={masks}). "
                "Beware that if you pass a path in 'masks' (e.g. 'leftHippocampus.nii.gz'), "
                "ClinicaDL will name the mask with its file name, without "
                "the extension (e.g. 'leftHippocampus')."
            )

        return masks

    @model_validator(mode="after")
    def _validate_args(self) -> Self:
        df = deepcopy(read_data(self.data)) if self.data is not None else None
        self._validate_columns(df)
        self._validate_masks()
        self._validate_label(df)

        return self

    def _validate_columns(self, df: Optional[pd.DataFrame]) -> None:
        """
        Checks if the columns are in the DataFrame.
        """
        if df is not None:
            for column in self.columns:
                if column not in df.columns:
                    raise KeyError(
                        f"'{column}' was passed in 'columns', but there is no such column in the DataFrame "
                        f"you passed in 'data'. Present columns are: {df.columns}"
                    )

        else:
            if len(self.columns) > 0:
                raise ValueError(
                    f"You passed {self._columns_names} in 'columns', but 'data' is None."
                )

    def _validate_masks(
        self,
    ) -> None:
        """
        Checks that a mask is not also a column, and separate masks between individual
        and common masks.
        """
        individual_masks, common_masks = [], []
        for mask in self.masks:
            if mask in self.columns:
                raise ValueError(
                    f"Conflict: '{mask}' has been passed in 'columns' AND 'masks'!"
                )
            if Mask.is_file(mask):
                common_masks.append(mask)
            else:
                individual_masks.append(mask)

        self.__dict__["_individual_masks"] = individual_masks
        self.__dict__["_common_masks"] = common_masks

    def _validate_label(self, df: Optional[pd.DataFrame]) -> None:
        """
        Checks if 'label' is a valid column name (or column names), a valid mask suffix or None.
        """
        label = self.label
        if isinstance(label, str):
            if label in self._common_mask_names:
                raise ValueError(
                    f"A segmentation mask must be specific to each image, but you passed label={label}, which is "
                    "a non image-specific mask."
                )
            elif label in self._individual_mask_names:
                return
            elif label in self.columns:
                label = [label]
            else:
                raise ValueError(
                    f"Got '{label}' for 'label', but there is no such column or mask."
                )

        if isinstance(label, list):
            self._validate_column_labels(label, df)

    def _validate_column_labels(
        self, labels: list[str], df: Optional[pd.DataFrame]
    ) -> None:
        """
        Validates labels that are columns.
        """
        for col in labels:
            if col in self.columns:
                df: pd.DataFrame  # df is not None if there are columns.
                if self.columns[col]:
                    try:
                        df[col] = self.columns[col](df[col])
                    except Exception as e:
                        raise ValueError(
                            f"Unable to process the column '{col}' with the function you passed. "
                            "Make sure that this function takes as input a Pandas Series, and returns a Pandas Series."
                        ) from e
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(
                        f"'{col}' was passed in 'label', but this column is not numeric!"
                    )
            else:
                raise ValueError(
                    f"You passed a list in 'label', and this list can only contain columns passed in 'columns'. But got: '{col}'"
                )


class BaseDataset(HasConfig[BaseDatasetConfig], SamplerDataset):
    """
    Abstract class with the main logic of all the :py:class:` ~clinicadl.data.datasets.Dataset`
    natively implemented in ``ClinicaDL``.
    """

    def __init__(
        self,
        directory: PathType,
        datatype: DataType,
        data: Optional[DataFrameType] = None,
        label: Optional[Union[str, Sequence[str]]] = None,
        transforms: Transforms = Transforms(),
        columns: Optional[
            Union[Sequence[str], dict[str, Optional[Callable[[pd.Series], pd.Series]]]]
        ] = None,
        masks: Optional[Sequence[Union[str, PathType]]] = None,
    ):
        self.config = self._config_type(
            directory=directory,
            datatype=datatype,
            data=data,
            label=label,
            transforms=transforms,
            columns=columns,
            masks=masks,
        )

        self.eval_mode = False

        self.transforms = deepcopy(
            self.config.transforms
        )  # self.transforms may be modified

        df = self._get_df_from_input(data)
        self._df = self._process_columns(df, self.config.columns)
        self.columns: list[str] = self.config._columns_names

        self.individual_masks: list[Mask] = list(
            map(self._read_mask, self.config._individual_masks)
        )
        self.common_masks: list[Mask] = list(
            map(self._read_mask, self.config._common_masks)
        )

        self.label = self._read_label(label)

        self._check_datatype()

    @property
    def _df(self) -> pd.DataFrame:
        return self.config.df

    @_df.setter
    def _df(self, df: pd.DataFrame) -> None:
        self.config.df = df

    def describe(self) -> dict[str, Any]:
        return {
            "participant_session_pairs": self.get_participant_session_couples(),
            "datatype": dict(self.config.datatype.to_dict()),
            "extraction": dict(self.config.transforms.extraction.to_dict()),
            "total_samples": len(self),
        }

    ### to read user inputs ###
    def _get_df_from_input(self, data: Optional[DataFrameType]) -> pd.DataFrame:
        """
        Generates or validates the DataFrame from the input data.
        """
        if data is None:
            data = self._create_df()
            logger.info("Creating a TSV file at %s", data)

        df = read_data(data)

        return deepcopy(
            df.sort_values(by=[PARTICIPANT_ID, SESSION_ID]).reset_index(drop=True)
        )

    @abstractmethod
    def _create_df(self) -> pd.DataFrame:
        """
        Creates a DataFrame enumerating the (participant, session) couples with
        the current datatype.
        """

    @staticmethod
    def _process_columns(
        df: pd.DataFrame,
        columns: dict[Column, Optional[Callable[[pd.Series], pd.Series]]],
    ) -> pd.DataFrame:
        """
        Processes the DataFrame with encoding functions passed by the user.
        """
        for column, encoding in columns.items():
            if encoding is None:
                continue
            try:
                df[column] = encoding(df[column])
            except Exception as e:
                raise ValueError(
                    f"Unable to process the column '{column}' with the function you passed. "
                    "Make sure that this function takes as input a Pandas Series, and returns a Pandas Series."
                ) from e

        return df

    def _read_mask(self, mask: PathType) -> Mask:
        """
        Determines if a mask is a common or an individual mask.
        """
        if Mask.is_file(mask):  # it is a file
            return Mask(self._get_common_mask_path(mask))
        else:
            return Mask(mask)

    @abstractmethod
    def _get_common_mask_path(self, mask_name: str) -> Path:
        """
        Gets the full path to the wanted mask.
        """

    def _read_label(
        self, label: Optional[Union[str, Sequence[str]]]
    ) -> Optional[Union[Column, list[Column], Mask]]:
        """
        Reads the label and determines its type (scalar, mask, or None).
        """
        if isinstance(label, str):
            if label in self.config._individual_mask_names:
                return Mask(label)

            return Column(label)

        if isinstance(label, list):
            return [Column(lab) for lab in label]

        return None

    def _check_datatype(self) -> None:
        """
        Checks that all the (participant, session) pairs in the dataset have
        the specified datatype.
        """
        for participant, session in self.get_participant_session_couples():
            if not self._has_datatype(participant, session, self.config.datatype):
                raise RuntimeError(
                    f"For ({participant}, {session}), no data corresponding to datatype={self.config.datatype}"
                )

    @abstractmethod
    def _has_datatype(self, participant: str, session: str, datatype: DataType) -> bool:
        """
        Determines if a (participant, session) has the specified datatype.
        """

    ### for __getitem__ ###
    def _get_data(self, participant: str, session: str) -> DataPoint:
        """
        Returns that data for a (participant, session) in a DataPoint.
        """
        image, image_path, additional_data = self._load_data(participant, session)

        # create datapoint
        datapoint = DataPoint(
            image=image,
            participant=participant,
            session=session,
            image_path=image_path,
            datatype=self.config.datatype,
            **additional_data,
        )

        # common masks (already loaded)
        for mask in self.common_masks:
            datapoint.add_mask(deepcopy(mask.get_associated_mask()), mask.name)

        # columns
        for col in self.columns:
            datapoint[col] = self._get_image_info(participant, session, col)

        return datapoint

    @abstractmethod
    def _load_data(
        self, participant: str, session: str
    ) -> tuple[tio.Image, Path, dict[str, Any]]:
        """
        Loads the image and any additional data (e.g. masks).
        Also returns the path to the image.
        """

    def _format_output(self, output: DataPoint) -> Sample:
        if isinstance(self.label, Mask):
            output[LABEL] = output.pop(self.label.name)
        elif isinstance(self.label, list):
            output[LABEL] = [output.pop(lab) for lab in self.label]
        elif self.label is not None:
            output[LABEL] = output.pop(self.label)

        return super()._format_output(output)

    @classmethod
    def _from_config(cls, config: BaseDatasetConfig) -> Self:
        dataset = cls(**config.to_raw_dict(exclude=[DF]))
        dataset._df = config.df

        return dataset
