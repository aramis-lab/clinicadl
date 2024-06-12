import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from clinicadl.caps_dataset.data_utils import check_multi_cohort_tsv, load_data_test
from clinicadl.caps_dataset.extraction.preprocessing import read_preprocessing
from clinicadl.utils.enum import Mode
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLTSVError,
)

logger = getLogger("clinicadl.data_config")


class DataConfig(BaseModel):  # TODO : put in data module
    """Config class to specify the data.

    caps_directory and preprocessing_json are arguments
    that must be passed by the user.
    """

    caps_directory: Path
    baseline: bool = False
    diagnoses: Tuple[str, ...] = ("AD", "CN")
    data_df: Optional[pd.DataFrame] = None
    label: Optional[str] = None
    label_code: Union[str, dict[str, int], None] = {}
    multi_cohort: bool = False
    mask_path: Optional[Path] = None
    preprocessing_json: Optional[str] = None
    data_tsv: Optional[Path] = None
    n_subjects: int = 300
    # pydantic config
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @field_validator("diagnoses", mode="before")
    def validator_diagnoses(cls, v):
        """Transforms a list to a tuple."""
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check if columns are in tsv

    def create_groupe_df(self):
        group_df = None
        if self.data_tsv is not None and self.data_tsv.is_file():
            group_df = load_data_test(
                self.data_tsv,
                self.diagnoses,
                multi_cohort=self.multi_cohort,
            )
        return group_df

    def is_given_label_code(self, _label: str, _label_code: Union[str, Dict[str, int]]):
        return (
            self.label is not None
            and self.label != ""
            and self.label != _label
            and _label_code == "default"
        )

    def check_label(self, _label: str):
        if not self.label:
            self.label = _label

    @field_validator("data_tsv", mode="before")
    @classmethod
    def check_data_tsv(cls, v) -> Path:
        if v is not None:
            if not isinstance(v, Path):
                v = Path(v)
            if not v.is_file():
                raise ClinicaDLTSVError(
                    "The participants_list you gave is not a file. Please give an existing file."
                )
            if v.stat().st_size == 0:
                raise ClinicaDLTSVError(
                    "The participants_list you gave is empty. Please give a non-empty file."
                )
        return v

    @computed_field
    @property
    def caps_dict(self) -> Dict[str, Path]:
        from clinicadl.utils.clinica_utils import check_caps_folder

        if self.multi_cohort:
            if self.caps_directory.suffix != ".tsv":
                raise ClinicaDLArgumentError(
                    "If multi_cohort is True, the CAPS_DIRECTORY argument should be a path to a TSV file."
                )
            else:
                caps_df = pd.read_csv(self.caps_directory, sep="\t")
                check_multi_cohort_tsv(caps_df, "CAPS")
                caps_dict = dict()
                for idx in range(len(caps_df)):
                    cohort = caps_df.loc[idx, "cohort"]
                    caps_path = Path(caps_df.at[idx, "path"])
                    check_caps_folder(caps_path)
                    caps_dict[cohort] = caps_path
        else:
            check_caps_folder(self.caps_directory)
            caps_dict = {"single": self.caps_directory}

        return caps_dict

    @computed_field
    @property
    def preprocessing_dict(self) -> Dict[str, Any]:
        """
        Gets the preprocessing dictionary from a preprocessing json file.

        Returns
        -------
        Dict[str, Any]
            The preprocessing dictionary.

        Raises
        ------
        ValueError
            In case of multi-cohort dataset, if no preprocessing file is found in any CAPS.
        """
        if not self.multi_cohort:
            preprocessing_json = (
                self.caps_directory / "tensor_extraction" / self.preprocessing_json
            )
        else:
            caps_dict = self.caps_dict
            json_found = False
            for caps_name, caps_path in caps_dict.items():
                preprocessing_json = (
                    caps_path / "tensor_extraction" / self.preprocessing_json
                )
                if preprocessing_json.is_file():
                    logger.info(
                        f"Preprocessing JSON {preprocessing_json} found in CAPS {caps_name}."
                    )
                    json_found = True
            if not json_found:
                raise ValueError(
                    f"Preprocessing JSON {self.preprocessing_json} was not found for any CAPS "
                    f"in {caps_dict}."
                )

        preprocessing_dict = read_preprocessing(preprocessing_json)

        if (
            preprocessing_dict["mode"] == "roi"
            and "roi_background_value" not in preprocessing_dict
        ):
            preprocessing_dict["roi_background_value"] = 0

        return preprocessing_dict

    # @computed_field
    # @property
    # def check_preprocessing_json(self) -> Path:
    #     if not self.multi_cohort:
    #         preprocessing_json = (
    #             self.caps_directory / "tensor_extraction" / self.preprocessing_json
    #         )
    #     else:
    #         caps_dict = self.caps_dict
    #         json_found = False
    #         for caps_name, caps_path in caps_dict.items():
    #             preprocessing_json = (
    #                 caps_path / "tensor_extraction" / self.preprocessing_json
    #             )
    #             if preprocessing_json.is_file():
    #                 logger.info(
    #                     f"Preprocessing JSON {preprocessing_json} found in CAPS {caps_name}."
    #                 )
    #                 json_found = True
    #         if not json_found:
    #             raise ValueError(
    #                 f"Preprocessing JSON {self.preprocessing_json} was not found for any CAPS "
    #                 f"in {caps_dict}."
    #             )
    #     return preprocessing_json

    @field_validator("preprocessing_json", mode="before")
    def compute_preprocessing_json(cls, v: str):
        if v is None:
            return f"extract_{int(time())}.json"
        elif not v.endswith(".json"):
            return f"{v}.json"
        else:
            return v
