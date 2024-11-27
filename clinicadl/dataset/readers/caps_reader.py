import json
import re
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import pandas as pd

from clinicadl.dataset.config.preprocessing import PreprocessingConfig
from clinicadl.dataset.transforms.transforms import Transforms
from clinicadl.dataset.utils import insensitive_glob
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)
from clinicadl.utils.iotools.utils import path_encoder

from .reader import Reader

logger = getLogger("clinicadl.caps_reader")


class CapsReader(Reader):
    def __init__(
        self,
        caps_directory: Path,
    ):
        """CAPS reader class for handling single-cohort CAPS directories.

        Args:
            caps_directory (Path): Path to the CAPS directory.
            from_bids (Optional[Path], optional): Path to BIDS directory, if applicable. Defaults to None.
        """
        super().__init__(caps_directory)
        self._check_caps_folder()
        self.subject_directory = self.input_directory / "subjects"

    def _check_caps_folder(self):
        """Check if provided `caps_directory`is a CAPS folder.

        Raises
        ------
        ValueError :
            If `caps_directory` is not a string.

        ClinicaCAPSError :
            If the provided path does not exist, or is not a directory.
            If the provided path is a BIDS folder (BIDS and CAPS could be
            swapped by user). We simply check that there is not a folder
            whose name starts with 'sub-' in the provided path (that exists
            in BIDS hierarchy).

        Notes
        -----
        Keep in mind that a CAPS folder can be empty.
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
        return f"CAPS Reader for {self.input_directory}"

    def get_preprocessing_folder(
        self, participant: str, session: str, preprocessing: Preprocessing
    ) -> Path:
        return self.get_session_path(participant=participant, session=session) / (
            preprocessing.value
        ).replace("-", "_")

    def get_participant_path(self, participant: str) -> Path:
        return self.subject_directory / participant

    def get_tensor_dir(
        self, participant: str, session: str, preprocessing: PreprocessingConfig
    ) -> Path:
        return (
            self.get_session_path(participant, session)
            / "deeplearning_prepare_data"
            / "image_based"
            / preprocessing.preprocessing.value.replace("-", "_")
        )

    def get_tensor_path(
        self, participant: str, session: str, preprocessing: PreprocessingConfig
    ) -> Path:
        """
        Gets the path to the tensor image (*.pt)

        Args:
            participant: ID of the participant.
            session: ID of the session.
        Returns:
            image_path: path to the tensor containing the whole image.
        """

        try:
            filepath = self.get_image_path(participant, session, preprocessing)
            image_filename = filepath.name.replace(".nii.gz", ".pt")
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
        self, participant: str, session: str, preprocessing: PreprocessingConfig
    ) -> Path:
        """Get the path"""

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
        preprocessing: PreprocessingConfig,
        data_tsv: Path,
        name: Optional[str] = None,
    ) -> None:
        """TODO: COMPLETE this method so that it writes all the info needed in a caps.json file"""
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
            print("yes")

    def load_data_test(self, test_path: Path, baseline=True):
        """
        Load data not managed by split_manager.

        Args:
            test_path (str): path to the test TSV files / split directory / TSV file for multi-cohort
            baseline (bool): If True baseline sessions only used (split_dir handling only).
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
