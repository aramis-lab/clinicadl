from glob import glob
from logging import getLogger
from pathlib import Path

from clinicadl.data.datatypes.file_type import FileType
from clinicadl.utils.exceptions import ClinicaDLBIDSError

from .reader import Reader

logger = getLogger("clinicadl.bids_reader")


class BidsReader(Reader):
    """
    BidsReader is a custom reader class for handling BIDS directories.

    This class extends the `Reader` class to provide additional functionality specific to
    handling BIDS (Brain Imaging Data Structure) directory formats. It supports validation
    and navigation of the BIDS directory structure.

    Args
    ----------
    bids_directory : Path
        Path to the BIDS directory containing neuroimaging data.
    """

    def __init__(
        self,
        bids_directory: Path,
    ):
        """
        Initializes the BidsReader.

        Parameters
        ----------
        bids_directory : Path
            Path to the BIDS directory containing neuroimaging data.

        Raises
        ------
        ClinicaDLBIDSError
            If the provided directory structure does not adhere to BIDS standards.
        """
        super().__init__(bids_directory)
        self.check_bids_folder()

    def check_bids_folder(self) -> None:
        """
        Validates whether the provided `bids_directory` adheres to the BIDS format.

        This function performs multiple checks to ensure the provided folder follows
        the BIDS standard:
        - It checks that the path is not a CAPS directory (by the absence of a 'subjects' folder).
        - It checks that the directory is not empty.
        - It verifies that the directory contains at least one participant folder starting with 'sub-'.

        Raises
        ------
        ValueError
            If `bids_directory` is not a string or valid path.
        ClinicaDLBIDSError
            If the provided path does not exist, is empty, or is not a valid BIDS directory.
        """
        # Check if the directory is mistakenly a CAPS folder (contains a 'subjects' folder).
        if (self.input_directory / "subjects").is_dir():
            raise ClinicaDLBIDSError(
                f"The BIDS directory ({self.input_directory}) you provided seems to "
                "be a CAPS directory due to the presence of a 'subjects' folder."
            )

        # Check if the directory is empty.
        if len([f for f in self.input_directory.iterdir()]) == 0:
            raise ClinicaDLBIDSError(
                f"The BIDS directory you provided is empty. ({self.input_directory})."
            )

        # Check if the directory contains at least one participant folder starting with 'sub-'.
        subj = [f for f in self.input_directory.iterdir() if f.name.startswith("sub-")]
        if len(subj) == 0:
            raise ClinicaDLBIDSError(
                "Your BIDS directory does not contain a single folder whose name "
                "starts with 'sub-'. Check that your folder follows the BIDS standard."
            )

    def __str__(self) -> str:
        """
        Returns a string representation of the BIDS Reader.

        Returns
        -------
        str
            A string representing the BIDS Reader with the path of the input directory.
        """
        return f"BIDS Reader for {self.input_directory}"

    def get_participant_path(self, participant: str) -> Path:
        """
        Returns the path to a specific participant's directory within the BIDS folder.

        Args
        ----------
        participant : str
            The participant's ID (e.g., 'sub-CLNC01').

        Returns
        -------
        Path
            The full path to the participant's folder.
        """
        return self.input_directory / participant

    def get_image_path(
        self, participant: str, session: str, file_type: FileType
    ) -> Path:
        """
        Retrieves the path of an image file for a given participant, session, and file type.

        This method uses a glob pattern to find the file matching the specified participant,
        session, and file type. It raises an error if more than one or no matching files are found.

        Parameters
        ----------
        participant : str
            The participant's ID (e.g., 'sub-CLNC01').
        session : str
            The session ID (e.g., 'ses-M000').
        file_type : FileType
            The file type (e.g., T1w, DWI) used to match the desired file.

        Returns
        -------
        Path
            The full path to the matching image file.

        Raises
        ------
        ClinicaDLBIDSError
            If no file or multiple files matching the pattern are found.
        """
        current_pattern = (
            self.get_session_path(participant, session) / "**" / file_type.pattern
        )
        current_glob_found = glob(str(current_pattern))

        if len(current_glob_found) > 1:
            error_str = f"\t*  ({participant} | {session}): More than 1 file found:\n"
            for found_file in current_glob_found:
                error_str += f"\t\t{found_file}\n"
            raise ClinicaDLBIDSError(error_str)
        elif len(current_glob_found) == 0:
            raise ClinicaDLBIDSError(
                f"\t* ({participant} | {session}): No file found\n"
            )
        else:
            return Path(current_glob_found[0])
