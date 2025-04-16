from __future__ import annotations

from pathlib import Path
from typing import Dict

from clinicadl.dictionary.suffixes import NII, NII_GZ, TSV
from clinicadl.dictionary.words import SES, SUB
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType

from ..base import Directory
from ..file_type import FileType
from .file_types.utils import get_file_type


class Bids(Directory):
    """Class to handle BIDS data format.

    This class is a placeholder for handling BIDS data format in the ClinicADL framework.
    It currently does not contain any methods or attributes.
    """

    def __init__(self, path: PathType):
        """Initialize the Bids class."""

        super().__init__(path=path)
        self.subjects: Dict[str, SubjectDir] = {}

    def load(self):
        """Load the BIDS data."""
        if not self.exists() or self.is_empty():
            raise ClinicaDLConfigurationError(
                f"The BIDS at {self.path} doesn't exist or is empty."
            )

        for subject in self.subjects_list:
            subject_dir = SubjectDir.load(parent_path=self.path, subject_id=subject)
            self.subjects[subject] = subject_dir

        print("loading complete")

    @property
    def participants_tsv(self) -> Path:
        """Return the path to the participants.tsv file."""
        return self.path / "participants.tsv"

    @property
    def subjects_list(self) -> list[str]:
        """Return a list of subject IDs."""
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []
        return [
            x.name.split("-")[1]
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(SUB)
        ]


class SubjectDir(Directory):
    """Class to handle subject directories in BIDS format.

    This class represents a subject directory within the BIDS data format.
    It currently does not contain any methods or attributes.
    """

    def __init__(self, parent_path: PathType, subject_id: str):
        """Initialize the SubjectDir class."""
        self.id = subject_id
        self.sessions: Dict[str, SessionDir] = {}

        super().__init__(path=Path(parent_path) / (SUB + "-" + subject_id))

    @classmethod
    def load(cls, parent_path: PathType, subject_id: str) -> SubjectDir:
        """Load a subject directory."""
        subject_dir = cls(parent_path=parent_path, subject_id=subject_id)

        if not subject_dir.exists() or subject_dir.is_empty():
            raise ClinicaDLConfigurationError(
                f"The subject at {subject_dir.path} doesn't exist or is empty."
            )

        for session in subject_dir.sessions_list:
            session_dir = SessionDir.load(
                parent_path=subject_dir.path, session_id=session
            )
            subject_dir.sessions[session] = session_dir

        return subject_dir

    @property
    def bids_dir(self) -> Path:
        """Return the path to the BIDS directory."""
        return self.path.parent

    @property
    def sessions_list(self) -> list[str]:
        """Return a list of session IDs."""
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []
        return [
            x.name.split("-")[1]
            for x in self.path.iterdir()
            if x.is_dir() and x.name.startswith(SES)
        ]


class SessionDir(Directory):
    """Class to handle subject directories in BIDS format.

    This class represents a subject directory within the BIDS data format.
    It currently does not contain any methods or attributes.
    """

    def __init__(self, parent_path: PathType, session_id: str):
        """Initialize the SubjectDir class."""
        self.id = session_id
        self.file_types: Dict[str, FileType] = {}

        super().__init__(path=Path(parent_path) / (SES + "-" + session_id))

    @classmethod
    def load(cls, parent_path: PathType, session_id: str) -> SessionDir:
        """Load a subject directory."""
        session_dir = cls(parent_path=parent_path, session_id=session_id)

        if not session_dir.exists() or session_dir.is_empty():
            raise ClinicaDLConfigurationError(
                f"The session at {session_dir.path} doesn't exist or is empty."
            )

        for filename in session_dir.filename_list:
            print(filename)
            file_type = get_file_type(filename=filename)
            session_dir.file_types[file_type.modality] = file_type

        return session_dir

    @property
    def subject_dir(self) -> Path:
        """Return the path to the subject directory."""
        return self.path.parent

    @property
    def subject(self) -> str:
        """Return the subject ID."""
        return self.path.parent.name.split("-")[1]

    @property
    def bids_dir(self) -> Path:
        """Return the path to the BIDS directory."""
        return self.path.parent.parent

    @property
    def scans_tsv(self) -> Path:
        """Return the path to the scans.tsv file."""
        return (self.path / f"{SUB}-{self.subject}_{SES}-{self.id}_scans").with_suffix(
            TSV
        )

    @property
    def filename_list(self) -> list[str]:
        """Return a list of data IDs."""
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []

        nii_extensions = (NII, NII_GZ)
        return [
            y.stem.removesuffix(NII)
            for x in self.path.iterdir()
            if x.is_dir()
            for y in x.iterdir()
            if y.is_file() and y.name.endswith(nii_extensions)
        ]
