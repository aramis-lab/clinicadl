from glob import glob
from logging import getLogger
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from clinicadl.data.datatypes.enum import PreprocessingMethod
from clinicadl.data.datatypes.preprocessing import Preprocessing
from clinicadl.data.readers.reader import Reader
from clinicadl.dictionary.suffixes import PT
from clinicadl.dictionary.words import PARTICIPANT_ID, SESSION_ID, SUBJECTS, TENSORS
from clinicadl.tsvtools.utils import df_to_tsv
from clinicadl.utils.exceptions import (
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
)
from clinicadl.utils.typing import PathType

logger = getLogger("clinicadl.data.readers.caps_reader")

COMMON_MASKS_DIR = "masks"
CONVERSION_JSON_DIRECTORY = "tensor_conversion"


class CapsReader(Reader):
    """
    A class to handle reading and accessing data from a CAPS (Clinica Application for Processing and Structuring) directory.

    This class provides methods to interact with a single-cohort CAPS directory, retrieve preprocessing paths,
    and manage data file retrievals such as images and tensors.

    Parameters
    ----------
    caps_directory : Path
        The path to the CAPS directory containing preprocessed neuroimaging data.

    Raises
    ------
    ClinicaDLCAPSError :
        If the `caps_directory` is a BIDS folder or does not contain the expected structure.
    """

    def __init__(
        self,
        caps_directory: PathType,
    ):
        super().__init__(caps_directory)
        self._check_caps_folder()
        self.subject_directory = self.input_directory / SUBJECTS

    @property
    def tensor_conversion_json_dir(self) -> Path:
        out_dir = self.input_directory / CONVERSION_JSON_DIRECTORY
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

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
        # TODO : more checks
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
                "A CAPS directory must have a folder 'subjects' at its root, in which "
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
        preprocessing = PreprocessingMethod(preprocessing)
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

    @staticmethod
    def path_to_tensor(
        path: PathType,
        conversion_name: str,
    ) -> Path:
        """
        Converts the path of an image to the path of the associated
        tensor (even if it does not exist yet).

        Parameters
        ----------
        path : PathType
            Path of the image.
        conversion_name : str
            The name of the tensor conversion.

        Returns
        -------
        Path
            Path to the associated tensor.
        """
        path = Path(path)
        parent = path.parent
        pt_file_name = (
            path.with_suffix("")
            .with_suffix(PT)
            .name  # with_suffix("") to handle double extensions
        )
        return parent / TENSORS / conversion_name / pt_file_name

    def get_tensor_path(
        self,
        participant: str,
        session: str,
        preprocessing: Preprocessing,
        conversion_name: str,
        check: bool = True,
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
        conversion_name: str
            The name of the tensor conversion.
        check : bool, default=True
            Whether to check if the tensor path exists.

        Returns
        -------
        Path
            Path to the tensor containing the image.

        Raises
        ------
        ClinicaDLCAPSError
            If there is no or more than one images associated with the participant/session pair.
        FileNotFoundError
            If `check` is true and the tensor image cannot be found.
        """

        filepath = self.get_image_path(participant, session, preprocessing)
        tensor_path = self.path_to_tensor(filepath, conversion_name=conversion_name)
        if check and not tensor_path.is_file():
            raise FileNotFoundError(
                f"Could not find the .pt path for participant {participant}, session {session} and preprocessing {preprocessing}"
            )

        return tensor_path

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
        file_pattern = preprocessing.file_type.pattern
        file_pattern = file_pattern.replace("sub-*", participant)
        file_pattern = file_pattern.replace("ses-*", session)
        global_pattern = self.get_session_path(participant, session) / file_pattern

        current_glob_found = glob(str(global_pattern))
        error_msg = f"For ({participant} | {session}), an error occurred while trying to get {preprocessing}: "
        if len(current_glob_found) > 1:  # e.g. a nii and a nii.gz file
            error_msg += "more than 1 file found:\n"
            for found_file in current_glob_found:
                error_msg += f"\t * {found_file}\n"
            raise ClinicaDLCAPSError(error_msg)
        elif len(current_glob_found) == 0:
            error_msg += "no file found"
            raise ClinicaDLCAPSError(error_msg)
        else:
            return Path(current_glob_found[0])

    def get_common_mask_path(self, mask_name: PathType) -> Path:
        """
        Gives the full path of a common mask, from the file name.
        """
        return self.input_directory / COMMON_MASKS_DIR / mask_name

    def get_common_mask_tensor_path(
        self, mask_name: PathType, conversion_name: str
    ) -> Path:
        """
        Gives the full tensor path of a common mask, from the mask filename.
        """
        return (
            self.input_directory
            / COMMON_MASKS_DIR
            / TENSORS
            / conversion_name
            / Path(mask_name).with_suffix("").with_suffix(PT)
        )

    def _write_caps_json(
        self,
        # transforms: Transforms,
        # preprocessing: Preprocessing,
        # data_tsv: Path,
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
        for participant, session in subjects_sessions:
            self.get_image_path(participant, session, preprocessing)

    def get_participants_sessions(
        self,
        preprocessing: Preprocessing,
    ) -> pd.DataFrame:
        """
        Finds all the (participant, session) for a specific preprocessing.
        """
        pattern = (
            self.subject_directory / "sub-*" / "ses-*" / preprocessing.file_type.pattern
        )
        files_found = glob(str(pattern))
        if len(files_found) == 0:
            raise ClinicaDLCAPSError("No image found for this preprocessing!")

        participants_sessions = set()
        for file in files_found:
            participant_session = (
                Path(file).relative_to(self.subject_directory).parents[1]
            )
            participant = str(participant_session.parent)
            session = participant_session.name
            participants_sessions.add((participant, session))

        return (
            pd.DataFrame(
                np.array(list(participants_sessions)),
                columns=[PARTICIPANT_ID, SESSION_ID],
            )
            .sort_values([PARTICIPANT_ID, SESSION_ID])
            .reset_index(drop=True)
        )

    def create_subjects_sessions_tsv(
        self,
        preprocessing: Preprocessing,
    ) -> str:
        """
        Finds all the (participant, session) for a specific preprocessing
        and saves them in a tsv.
        """
        tsv_path = self.input_directory / preprocessing.tsv_filename
        df = self.get_participants_sessions(preprocessing)
        df_to_tsv(tsv_path, df)

        return str(tsv_path)
