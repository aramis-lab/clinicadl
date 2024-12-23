from logging import getLogger
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from clinicadl.data.preprocessing import Preprocessing
from clinicadl.data.readers.reader import Reader
from clinicadl.data.utils import insensitive_glob
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.enum import PreprocessingMethod
from clinicadl.utils.exceptions import (
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
)

logger = getLogger("clinicadl.caps_reader")


class CapsReader(Reader):
    """
    A class to handle reading and accessing data from a CAPS (Clinica Application for Processing and Structuring) directory.

    This class provides methods to interact with a single-cohort CAPS directory, retrieve preprocessing paths,
    and manage data file retrievals such as images and tensors.

    Parameters
    ----------
    caps_directory : Path
        The path to the CAPS directory containing preprocessed neuroimaging data.
    """

    def __init__(
        self,
        caps_directory: Path,
    ):
        """
        Initializes the CAPS reader by verifying the structure of the CAPS directory.

        Args:
            caps_directory (Path): Path to the CAPS directory.
        """

        super().__init__(caps_directory)
        self._check_caps_folder()
        self.subject_directory = self.input_directory / "subjects"

    def _check_caps_folder(self) -> None:
        """
        Validates if the provided `caps_directory` is a valid CAPS directory.

        Raises
        ------
        ValueError :
            If `caps_directory` is not a valid string or directory.

        ClinicaDLCAPSError :
            If the `caps_directory` is a BIDS folder or does not contain the expected structure.
        """
        sub_folders = [
            f for f in self.input_directory.iterdir() if f.name.startswith("sub-")
        ]
        if len(sub_folders) > 0:
            error_string = (
                "Your CAPS directory contains at least one folder whose name "
                "starts with 'sub-'. Check that you did not swap BIDS and CAPS folders.\n"
                "Folder(s) found that match(es) BIDS architecture:\n"
            )
            for directory in sub_folders:
                error_string += f"\t{directory}\n"
            error_string += (
                "A CAPS directory has a folder 'subjects' at its root, in which "
                "are stored the output of the pipeline for each participant."
            )
            raise ClinicaDLCAPSError(error_string)

    def __str__(self) -> str:
        """
        String representation of the CAPS Reader.

        Returns
        -------
        str
            Description of the CAPS directory.
        """
        return f"CAPS reader for {self.input_directory}"

    def get_preprocessing_folder(
        self, participant: str, session: str, preprocessing: PreprocessingMethod
    ) -> Path:
        """
        Retrieves the folder path for a specific preprocessing step.

        Args:
            participant (str): ID of the participant.
            session (str): ID of the session.
            preprocessing (PreprocessingMethod): Preprocessing step for which the folder path is needed.

        Returns
        -------
        Path
            Path to the folder containing the preprocessing data.
        """
        return self.get_session_path(participant=participant, session=session) / (
            preprocessing.value
        ).replace("-", "_")

    def get_participant_path(self, participant: str) -> Path:
        """
        Retrieves the path to the participant's directory.

        Args:
            participant (str): ID of the participant.

        Returns
        -------
        Path
            Path to the participant directory.
        """
        return self.subject_directory / participant

    def get_tensor_dir(
        self, participant: str, session: str, preprocessing: Preprocessing
    ) -> Path:
        """
        Retrieves the directory for storing tensor data for a given participant, session, and preprocessing.

        Args:
            participant (str): ID of the participant.
            session (str): ID of the session.
            preprocessing (Preprocessing): Configuration of the preprocessing steps.

        Returns
        -------
        Path
            Directory where tensor data is stored.
        """
        return (
            self.get_session_path(participant, session)
            / "deeplearning_prepare_data"
            / "image_based"
            / preprocessing.preprocessing.value.replace("-", "_")
        )

    def get_tensor_path(
        self, participant: str, session: str, preprocessing: Preprocessing
    ) -> Path:
        """
        Retrieves the path to the tensor image (*.pt) for a given participant, session, and preprocessing.

        Parameters
        ----------
            participant: str
                ID of the participant.
            session: str
                ID of the session.
            preprocessing: Preprocessing
                Configuration of the preprocessing steps.

        Returns
        -------
        Path
            Path to the tensor containing the image.

        Raises
        ------
        ClinicaDLCAPSError
            If the path for the tensor image cannot be found.
        """

        try:
            filepath = self.get_image_path(participant, session, preprocessing)
            image_filename = (
                filepath.with_suffix("").with_suffix("").with_suffix(".pt").name
            )  # two times to handle double extensions
            image_path = (
                self.get_tensor_dir(participant, session, preprocessing)
                / image_filename
            )

            return image_path

        except ClinicaDLCAPSError:
            raise ClinicaDLCAPSError(
                f"Could not find the pt path for participant {participant} and session {session}"
            )

    def get_image_path(
        self, participant: str, session: str, preprocessing: Preprocessing
    ) -> Path:
        """
        Retrieves the path to the image file for a given participant, session, and preprocessing.

        Parameters
        ----------
            participant: str
                ID of the participant.
            session: str
                ID of the session.
            preprocessing: Preprocessing
                Configuration of the preprocessing steps.

        Returns
        -------
        Path
            Path to the image file.

        Raises
        ------
        ClinicaDLCAPSError
            If more than one or no image file is found.
        """

        current_pattern = (
            self.get_session_path(participant, session)
            / "**"
            / preprocessing.file_type.pattern
        )
        current_glob_found = insensitive_glob(str(current_pattern), recursive=True)
        if len(current_glob_found) > 1:
            error_str = f"\t*  ({participant} | {session}): More than 1 file found:\n"
            for found_file in current_glob_found:
                error_str += f"\t\t{found_file}\n"
            raise ClinicaDLCAPSError(error_str)
        elif len(current_glob_found) == 0:
            raise ClinicaDLCAPSError(
                f"\t* ({participant} | {session}): No file found\n"
            )
        else:
            return Path(current_glob_found[0])

    def _write_caps_json(
        self,
        transforms: Transforms,
        preprocessing: Preprocessing,
        data_tsv: Path,
        name: Optional[str] = None,
    ) -> None:
        """
        Writes the preprocessing and transformation configurations into a JSON file.

        Args:
            transforms: Transforms
                The transformations applied to the data.
            preprocessing: Preprocessing
                Preprocessing configuration.
            data_tsv: Path
                Path to the data TSV file.
            name: str, optional
                Optional name for the JSON file. Defaults to "caps.json".

        Raises
        ------
        ClinicaDLCAPSError
            If the specified JSON file already exists.
        """
        if name:
            if not name.endswith(".json"):
                name += ".json"
            caps_json = self.input_directory / name
        else:
            caps_json = self.input_directory / "caps.json"

        if caps_json.is_file():
            raise ClinicaDLCAPSError(
                f"The JSON file {caps_json} already exists, please give another name."
            )
        else:
            # dict_ = transforms.model_dump()
            # dict_.update(preprocessing.model_dump())
            # dict_["data_tsv"] = str(data_tsv)

            # with open(caps_json, "w") as f:
            #     json.dump(dict_, f)

            # Future implementation to write the JSON file (commented out for now)
            print("Writing caps.json is not yet implemented.")

    def load_data_test(self, test_path: Path, baseline=True):
        """
        Loads a test dataset from a provided TSV file, checking the baseline sessions if specified.

        Args:
            test_path (Path): Path to the TSV file containing test data.
            baseline (bool): If True, only baseline sessions are used (relevant for multi-cohort).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the test data.

        Raises
        ------
        ClinicaDLConfigurationError
            If the provided test path is not a valid TSV file.
        """
        # TODO: computes baseline sessions on-the-fly to manager TSV file case

        if test_path.suffix != ".tsv" or not test_path.is_file():
            raise ClinicaDLConfigurationError(
                "Test path should be a TSV file. Please provide a valid TSV file path."
            )
        tsv_df = pd.read_csv(test_path, sep="\t")
        multi_col = {"cohort", "path"}
        if multi_col.issubset(tsv_df.columns.values):
            raise ClinicaDLConfigurationError(
                "To use multi-cohort framework, please add 'multi_cohort=true' in your configuration file or '--multi_cohort' flag to the command line."
            )
        test_path = self.check_test_path(test_path=test_path, baseline=baseline)
        test_df = pd.read_csv(test_path, sep="\t")
        test_df.reset_index(inplace=True, drop=True)
        test_df["cohort"] = "single"

        return test_df

    def check_preprocessing(
        self,
        subjects_sessions: Sequence[Tuple[str, str]],
        preprocessing: Preprocessing,
    ):
        """
        Validates that all subject/session pairs have a specific preprocessing.

        Returns
        -------
        subjects_sessions : Sequence[Tuple[str, str]]
            The list of (subject, session) that should be checked.
        preprocessing: Preprocessing
            The preprocessing.

        Raises
        ------
        ClinicaDLConfigurationError
            If the preprocessing is not found for a subject/session pair.
        """
        pattern = preprocessing.file_type.pattern
        for participant, session in subjects_sessions:
            folder = self.get_session_path(participant=participant, session=session)
            if not list(folder.glob(pattern)):
                raise ClinicaDLConfigurationError(
                    f"Could not find preprocessing {preprocessing.preprocessing.value} for "
                    f"participant {participant} and session {session} with pattern: {pattern}"
                )
