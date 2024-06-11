from logging import getLogger
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, computed_field

from clinicadl.caps_dataset.extraction.preprocessing import read_preprocessing

logger = getLogger("clinicadl.ssda_config")


class SSDAConfig(BaseModel):
    """Config class to perform SSDA."""

    caps_target: Path = Path("")
    preprocessing_json_target: Path = Path("")
    ssda_network: bool = False
    tsv_target_lab: Path = Path("")
    tsv_target_unlab: Path = Path("")
    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    @computed_field
    @property
    def preprocessing_dict_target(self) -> Dict[str, Any]:
        """
        Gets the preprocessing dictionary from a target preprocessing json file.

        Returns
        -------
        Dict[str, Any]
            The preprocessing dictionary.
        """
        if not self.ssda_network:
            return {}

        preprocessing_json_target = (
            self.caps_target / "tensor_extraction" / self.preprocessing_json_target
        )

        return read_preprocessing(preprocessing_json_target)
