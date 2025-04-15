from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from clinicadl.data.datatypes.modalities.pet import ReconstructionMethod, Tracer
from clinicadl.dictionary.suffixes import TSV
from clinicadl.dictionary.words import ANAT, DWI, PET, SES, SESSION, SUB, SUBJECTS
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLBIDSError,
    ClinicaDLConfigurationError,
)
from clinicadl.utils.typing import PathType

from ..base import Directory


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

        super().__init__(path=Path(parent_path) / subject_id)

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

        # self.anat: Optional[AnatDir] = None
        # self.dwi: Optional[DWIDir] = None
        # self.pet: Optional[PETDir] = None
        # self.custom: Optional[CustomDir] = None

        super().__init__(path=Path(parent_path) / session_id)

    @classmethod
    def load(cls, parent_path: PathType, session_id: str) -> SessionDir:
        """Load a subject directory."""
        session_dir = cls(parent_path=parent_path, session_id=session_id)

        if not session_dir.exists() or session_dir.is_empty():
            raise ClinicaDLConfigurationError(
                f"The session at {session_dir.path} doesn't exist or is empty."
            )

        for data in session_dir.data_list:
            if data == ANAT:
                cls.anat = AnatDir(parent_path=session_dir.path)
                cls.anat.load()
            elif data == PET:
                cls.pet = PETDir(parent_path=session_dir.path)
                cls.pet.load()
            elif data == DWI:
                cls.dwi = DWIDir(parent_path=session_dir.path)
                cls.dwi.load()
            elif data == CUSTOM:
                cls.custom = CustomDir(parent_path=session_dir.path)
                cls.custom.load()
            else:
                raise ClinicaDLArgumentError(f"The data type {data} is not supported.")
            data_dir = DataDir(parent_path=session_dir.path, data_id=data)
            session_dir.data[data] = data_dir

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
        return (self.path / f"sub-{self.subject}_ses-{self.id}_scans").with_suffix(TSV)

    @property
    def data_list(self) -> list[str]:
        """Return a list of data IDs."""
        if not self.exists():
            raise ClinicaDLConfigurationError(f"The MAPS at {self.path} doesn't exist.")
        if self.is_empty():
            return []
        return [
            x.name
            for x in self.path.iterdir()
            if x.is_dir() and x in SupportedRawDataDir
        ]


class DataDir(Directory):
    """Class to handle subject directories in BIDS format.

    This class represents a subject directory within the BIDS data format.
    It currently does not contain any methods or attributes.
    """

    def __init__(self, parent_path: PathType, data_id: str):
        """Initialize the SubjectDir class."""
        self.id = data_id
        super().__init__(path=Path(parent_path) / data_id)


class SupportedRawDataDir(str, Enum):
    """Enum class to represent supported raw data directories in BIDS format.

    This class is used to define the supported raw data directories in the BIDS format.
    """

    ANAT = "anat"
    PET = "pet"
    DWI = "dwi"
    CUSTOM = "custom"
    FMAP = "fmap"
    FUNC = "func"


bids = Bids(path=Path())
bids.subjects[sub].sessions[ses].anat.t1w.json
bids.subjects[sub].sessions[ses].anat.t1w.json
bids.subjects[sub].sessions[ses].anat.t1w.json


class AnatDir(Directory):
    """Class to handle anatomical directories in BIDS format.

    This class represents a subject directory within the BIDS data format.
    It currently does not contain any methods or attributes.
    """

    def __init__(self, parent_path: PathType):
        """Initialize the SubjectDir class."""

        super().__init__(path=Path(parent_path) / ANAT)

        self.t1w: Optional[T1WFiles] = None
        self.flair: Optional[FlairFiles] = None

    def load(self):
        """Load the anatomical data."""
        if not self.exists() or self.is_empty():
            raise ClinicaDLConfigurationError(
                f"The BIDS at {self.path} doesn't exist or is empty."
            )

        self.t1w.load(path=self.path)
        self.flair.load(path=self.path)

        if not self.t1w.exists() and not self.flair.exists():
            raise ClinicaDLConfigurationError(
                f"The anatomical data directory at {self.path} doesn't exist or is empty."
            )


class PETDir(Directory):
    """Class to handle PET directories in BIDS format.

    This class represents a subject directory within the BIDS data format.
    It currently does not contain any methods or attributes.
    """

    def __init__(self, parent_path: PathType):
        """Initialize the SubjectDir class."""
        super().__init__(path=Path(parent_path) / PET)
        self.pets = {}

    def load(self):
        """Load the anatomical data."""
        if not self.exists() or self.is_empty():
            raise ClinicaDLConfigurationError(
                f"The BIDS at {self.path} doesn't exist or is empty."
            )

        patterns = [
            PETFiles(tracer=tracer, reconstruction=recon)
            for tracer in Tracer
            for recon in ReconstructionMethod
        ]
        for tracer in Tracer:
            pet = PETFiles(tracer=tracer)
            pet.load(path=self.path)
            if not pet.exists():
                raise ClinicaDLConfigurationError(
                    f"The anatomical data directory at {self.path} doesn't exist or is empty."
                )


class DWIDir(Directory):
    """Class to handle PET directories in BIDS format.

    This class represents a subject directory within the BIDS data format.
    It currently does not contain any methods or attributes.
    """

    def __init__(self, parent_path: PathType):
        """Initialize the SubjectDir class."""
        super().__init__(path=Path(parent_path) / PET)
        self.pets = {}

    def load(self):
        """Load the anatomical data."""
        if not self.exists() or self.is_empty():
            raise ClinicaDLConfigurationError(
                f"The BIDS at {self.path} doesn't exist or is empty."
            )

        for tracer in Tracer:
            pet = PETFiles(tracer=tracer)
            pet.load(path=self.path)
            if not pet.exists():
                raise ClinicaDLConfigurationError(
                    f"The anatomical data directory at {self.path} doesn't exist or is empty."
                )


class Files(ABC):
    def __init__(self):
        self.json = None
        self.nii = None

    def exists(self):
        """Check if the T1-weighted image and its JSON sidecar file exist."""
        return self.json or self.nii

    @property
    @abstractmethod
    def suffix(self) -> str:
        """Return the suffix of the image."""
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def load(cls, path: Path) -> None:
        """Load the T1-weighted image and its JSON sidecar file."""
        if not path.exists():
            raise ClinicaDLConfigurationError(
                f"The anatomical data at {path} doesn't exist or is empty."
            )
        subject = path.parent.parent.name
        session = path.parent.name
        files = cls()
        for file in path.iterdir():
            if file.name.endswith(f"{subject}_{session}_{files.suffix}.json"):
                files.json = file
            if file.name.endswith(f"{subject}_{session}_{files.suffix}.nii*"):
                files.nii = file


class T1WFiles(Files):
    @property
    def suffix(self) -> str:
        """Return the suffix of the image."""
        return "T1w"


class FlairFiles(Files):
    @property
    def suffix(self) -> str:
        """Return the suffix of the image."""
        return "flair"


class PETFiles(Files):
    def __init__(
        self,
        tracer: Union[str, Tracer],
        reconstruction: Optional[Union[str, ReconstructionMethod]] = None,
    ):
        """Initialize the PETFiles class."""

        if isinstance(tracer, str):
            tracer = Tracer(tracer)
        self.tracer = tracer

        if reconstruction and isinstance(reconstruction, str):
            reconstruction = ReconstructionMethod(reconstruction)
        self.reconstruction = reconstruction

        super().__init__()

    @property
    def suffix(self):
        """Return the suffix of the image."""
        return "pet"

    @property
    def pattern(self) -> str:
        """Return the suffix of the image."""
        if self.reconstruction:
            rec_pattern = f"_rec-{self.reconstruction}"
        else:
            rec_pattern = ""
        return f"trc-{self.tracer}{rec_pattern}_{self.suffix}"


class DWIFiles(Files):
    def __init__(self):
        super().__init__()
        self.bval = None
        self.bvec = None

    def exists(self):
        """Check if the DWI image and its JSON sidecar file exist."""
        return super().exists() or self.bval or self.bvec

    @property
    def suffix(self) -> str:
        """Return the suffix of the image."""
        return "dwi"

    def load(self, path: Path) -> None:
        """Load the DWI image and its JSON sidecar file."""

        super().load(path=path)

        subject = path.parent.parent.name
        session = path.parent.name

        for file in path.iterdir():
            if file.name.endswith(f"{subject}_{session}_{self.suffix}.bval"):
                self.bval = file
            if file.name.endswith(f"{subject}_{session}_{self.suffix}.bvec"):
                self.bvec = file
