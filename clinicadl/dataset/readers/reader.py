import json
import re
from abc import abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

from clinicadl.dataset.config.preprocessing import PreprocessingConfig
from clinicadl.dataset.config.utils import (
    get_infos_from_json,
)
from clinicadl.dataset.transforms.extraction import (
    BaseExtraction,
    Image,
)
from clinicadl.dataset.transforms.transforms import Transforms
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)

logger = getLogger("clinicadl.caps_reader")


class Reader:
    """Base reader class for BIDS and CAPS directories.

    Args:
        input_dir (Path): Path to the BIDS or CAPS directory.
    """

    def __init__(self, input_dir: Path) -> None:
        self.input_directory = input_dir
        self._check_folder()

    def _check_folder(self) -> None:
        """Utility function which performs checks common to BIDS and CAPS folder structures."""

        if not isinstance(self.input_directory, (Path, str)):
            raise ValueError(
                "Argument you provided to check__folder() is not a string."
            )
        if not self.input_directory.is_dir():
            raise ClinicaDLArgumentError(
                f"The directory you gave is not a folder.\n"
                "Error explanations:\n"
                f"\t- Clinica expected the following path to be a folder: {self.input_directory}\n"
                "\t- If you gave relative path, did you run Clinica on the good folder?"
            )

    @abstractmethod
    def get_participant_path(self, participant: str) -> Path:
        pass

    def get_session_path(self, participant: str, session: str) -> Path:
        return self.get_participant_path(participant) / session

    def get_participant_session_from_filename(self, filename: Path) -> Tuple[str, str]:
        """Extract container from BIDS or CAPS file.

        Parameters
        ----------
        filename : str
            Full path to BIDS or CAPS filename.

        Returns
        -------
        str :
            Container path of the form "<participant_id>/<session_id>".

        Examples
        --------
        >>> container_from_filename('/path/to/bids/sub-CLNC01/ses-M000/anat/sub-CLNC01_ses-M000_T1w.nii.gz')
        'sub-CLNC01/ses-M000'
        >>> container_from_filename('caps/subjects/sub-CLNC01/ses-M000/dwi/preprocessing/sub-CLNC01_ses-M000_preproc.nii')
        'sub-CLNC01/ses-M000'
        """

        m = re.search(r"(sub-[a-zA-Z0-9]+)/(ses-[a-zA-Z0-9]+)", str(filename))
        if not m:
            raise ValueError(
                f"Input filename {filename} is not in a BIDS or CAPS compliant format."
                "It does not contain the participant and session ID."
            )
        participant = m.group(1)
        session = m.group(2)
        return participant, session

    def get_infos_from_json(
        self, preprocessing_json: Path
    ) -> Tuple[PreprocessingConfig, BaseExtraction, Transforms]:
        """Load preprocessing and extraction configuration from JSON file."""
        if not preprocessing_json.is_file():
            raise FileNotFoundError(
                f"The provided preprocessing JSON file {preprocessing_json} does not exist."
            )

        return get_infos_from_json(preprocessing_json)

    def check_test_path(self, test_path: Path, baseline: bool = True) -> Path:
        if baseline:
            train_filename = "train_baseline.tsv"
            label_filename = "labels_baseline.tsv"
        else:
            train_filename = "train.tsv"
            label_filename = "labels.tsv"

        if not (test_path.parent / train_filename).is_file():
            if not (test_path.parent / label_filename).is_file():
                raise ClinicaDLTSVError(
                    f"There is no {train_filename} nor {label_filename} in your folder {test_path.parents[0]} "
                )
            else:
                test_path = test_path.parent / label_filename
        else:
            test_path = test_path.parent / train_filename

        return test_path
