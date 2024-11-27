import json
from logging import getLogger
from pathlib import Path

from clinicadl.dataset.config import FileType
from clinicadl.dataset.utils import insensitive_glob
from clinicadl.utils.exceptions import ClinicaDLBIDSError
from clinicadl.utils.iotools.utils import path_encoder

from .reader import Reader

logger = getLogger("clinicadl.bids_reader")


class BidsReader(Reader):
    def __init__(
        self,
        bids_directory: Path,
    ):
        """CAPS reader class for handling single-cohort CAPS directories.

        Args:
            caps_directory (Path): Path to the CAPS directory.
            from_bids (Optional[Path], optional): Path to BIDS directory, if applicable. Defaults to None.
        """
        super().__init__(bids_directory)
        self.check_bids_folder()

    def check_bids_folder(self) -> None:
        """Check if provided `bids_directory` is a BIDS folder.

        Raises
        ------
        ValueError :
            If `bids_directory` is not a string.

        ClinicaDLBIDSError :
            If the provided path does not exist, or is not a directory.
            If the provided path is a CAPS folder (BIDS and CAPS could
            be swapped by user). We simply check that there is not a folder
            called 'subjects' in the provided path (that exists in CAPS hierarchy).
            If the provided folder is empty.
            If the provided folder does not contain at least one directory whose
            name starts with 'sub-'.
        """
        if (self.input_directory / "subjects").is_dir():
            raise ClinicaDLBIDSError(
                f"The BIDS directory ({self.input_directory}) you provided seems to "
                "be a CAPS directory due to the presence of a 'subjects' folder."
            )

        if len([f for f in self.input_directory.iterdir()]) == 0:
            raise ClinicaDLBIDSError(
                f"The BIDS directory you provided is empty. ({self.input_directory})."
            )

        subj = [f for f in self.input_directory.iterdir() if f.name.startswith("sub-")]
        if len(subj) == 0:
            raise ClinicaDLBIDSError(
                "Your BIDS directory does not contains a single folder whose name "
                "starts with 'sub-'. Check that your folder follow BIDS standard."
            )

    def __str__(self) -> str:
        return f"BIDS Reader for {self.input_directory}"

    def get_participant_path(self, participant: str) -> Path:
        return self.input_directory / participant

    def get_image_path(
        self, participant: str, session: str, file_type: FileType
    ) -> Path:
        """Get the path"""

        current_pattern = (
            self.get_session_path(participant, session) / "**" / file_type.pattern
        )
        current_glob_found = insensitive_glob(str(current_pattern), recursive=True)
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
