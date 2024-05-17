from enum import Enum
from logging import getLogger
from pathlib import Path
from time import time
from typing import Annotated, Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator

from clinicadl.utils.caps_dataset.data_config import DataConfig
from clinicadl.utils.exceptions import ClinicaDLTSVError
from clinicadl.utils.mode.mode_config import ModeConfig
from clinicadl.utils.preprocessing.preprocessing_config import PreprocessingConfig

logger = getLogger("clinicadl.prepare_data_config")


class PrepareDataConfig(BaseModel):
    preprocessing: PreprocessingConfig
    mode: ModeConfig
    data: DataConfig

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

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
