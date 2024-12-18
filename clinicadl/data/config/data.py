from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Union

from pydantic import field_validator

from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.exceptions import ClinicaDLTSVError

logger = getLogger("clinicadl.data_config")


# TODO: check if this file is still useful


class DataConfig(ClinicaDLConfig):  # TODO : put in data module
    """Config class to specify the data.

    caps_directory and preprocessing_json are arguments
    that must be passed by the user.
    """

    caps_directory: Optional[Path] = None
    baseline: bool = False
    mask_path: Optional[Path] = None
    data_tsv: Optional[Path] = None
    n_subjects: int = 300

    # @field_validator("diagnoses", mode="before")
    # def validator_diagnoses(cls, v):
    #     """Transforms a list to a tuple."""
    #     if isinstance(v, list):
    #         return tuple(v)
    #     return v  # TODO : check if columns are in tsv

    def create_groupe_df(self):
        group_df = None
        # if self.data_tsv is not None and self.data_tsv.is_file():
        # group_df = load_data_test(
        #     self.data_tsv,
        #     multi_cohort=False,
        # )
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
