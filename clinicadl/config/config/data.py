from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from clinicadl.utils.caps_dataset.data import load_data_test
from clinicadl.utils.enum import Mode
from clinicadl.utils.maps_manager.maps_manager_utils import read_json
from clinicadl.utils.preprocessing import read_preprocessing

logger = getLogger("clinicadl.data_config")


class DataConfig(BaseModel):  # TODO : put in data module
    """Config class to specify the data.

    caps_directory and preprocessing_json are arguments
    that must be passed by the user.
    """

    caps_directory: Path
    baseline: bool = False
    diagnoses: Tuple[str, ...] = ("AD", "CN")
    label: Optional[str] = None
    label_code: Dict[str, int] = {}
    multi_cohort: bool = False
    preprocessing_json: Path
    data_tsv: Optional[Path] = None
    n_subjects: int = 300
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    def create_groupe_df(self):
        group_df = None
        if self.data_tsv is not None and self.data_tsv.is_file():
            group_df = load_data_test(
                self.data_tsv,
                self.diagnoses,
                multi_cohort=self.multi_cohort,
            )
        return group_df

    @field_validator("diagnoses", mode="before")
    def validator_diagnoses(cls, v):
        """Transforms a list to a tuple."""
        if isinstance(v, list):
            return tuple(v)
        return v  # TODO : check if columns are in tsv

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
        from clinicadl.utils.caps_dataset.data import CapsDataset

        if not self.multi_cohort:
            preprocessing_json = (
                self.caps_directory / "tensor_extraction" / self.preprocessing_json
            )
        else:
            caps_dict = CapsDataset.create_caps_dict(
                self.caps_directory, self.multi_cohort
            )
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

    @computed_field
    @property
    def mode(self) -> Mode:
        return Mode(self.preprocessing_dict["mode"])
