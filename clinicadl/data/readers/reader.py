import re
from abc import ABC, abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Tuple

from clinicadl.utils.exceptions import ClinicaDLArgumentError, ClinicaDLTSVError
from clinicadl.utils.typing import PathType

logger = getLogger("clinicadl.reader")


class Reader(ABC):
    """
    Base reader class for handling BIDS and CAPS directories.

    Argument
    --------
    input_dir : Path
        Path to the BIDS or CAPS directory. This directory should contain the participant
        and session subdirectories for the corresponding dataset.
    """

    def __init__(self, input_dir: PathType) -> None:
        """
        Initializes the Reader object with the input directory and performs folder validation.

        Parameters
        ----------
        input_dir : Path
            Path to the input BIDS or CAPS directory.

        Raises
        ------
        ClinicaDLArgumentError
            If the input directory is not valid or is not a directory.
        """
        self.input_directory = Path(input_dir)
        self._check_folder()

    def _check_folder(self) -> None:
        """
        Utility function which performs checks common to BIDS and CAPS folder structures.

        This function checks if the provided input directory exists and is a valid folder.
        If the directory is invalid, it raises an error.

        Raises
        ------
        ClinicaDLArgumentError
            If the directory is not a valid folder.
        """

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
        """
        Abstract method to retrieve the path for a specific participant's data.

        Parameters
        ----------
        participant : str
            The participant identifier (e.g., "sub-CLNC01").

        Returns
        -------
        Path
            The full path to the participant's data directory.

        Note
        ----
        This method must be implemented by subclasses.
        """
        pass

    def get_session_path(self, participant: str, session: str) -> Path:
        """
        Returns the path to the session directory for a given participant and session.

        Parameters
        ----------
        participant : str
            The participant identifier (e.g., "sub-CLNC01").
        session : str
            The session identifier (e.g., "ses-M000").

        Returns
        -------
        Path
            The full path to the session directory for the specified participant and session.
        """
        return self.get_participant_path(participant) / session

    def get_participant_session_from_filename(self, filename: Path) -> Tuple[str, str]:
        """
        Extracts the participant and session identifiers from a BIDS or CAPS filename.

        Parameters
        ----------
        filename : Path
            Full path to a BIDS or CAPS filename.

        Returns
        -------
        Tuple[str, str]
            A tuple containing the participant ID and session ID.

        Raises
        ------
        ValueError
            If the filename does not conform to the BIDS or CAPS format, i.e., it does not contain
            the participant and session identifiers.

        Examples
        --------
        >>> self.get_participant_session_from_filename('/path/to/bids/sub-CLNC01/ses-M000/anat/sub-CLNC01_ses-M000_T1w.nii.gz')
        ('sub-CLNC01', 'ses-M000')
        >>> self.get_participant_session_from_filename('caps/subjects/sub-CLNC01/ses-M000/dwi/preprocessing/sub-CLNC01_ses-M000_preproc.nii')
        ('sub-CLNC01', 'ses-M000')
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

    def check_test_path(self, test_path: Path, baseline: bool = True) -> Path:
        """
        Validates and adjusts the test path based on the baseline or regular dataset configuration.

        Parameters
        ----------
        test_path : Path
            The path to the test file.
        baseline : bool, optional
            Whether the dataset corresponds to the baseline configuration. Default is True.

        Returns
        -------
        Path
            The valid path to the test file, which could either be `train.tsv` or `labels.tsv`.

        Raises
        ------
        ClinicaDLTSVError
            If neither the `train.tsv` nor the `labels.tsv` file is found in the folder.
        """

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
