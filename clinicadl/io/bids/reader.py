from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Optional

from pydantic import (
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from clinicadl.utils.bids import BidsEntity, Session, Subject
from clinicadl.utils.config import ClinicaDLConfig, ObjectConfig
from clinicadl.utils.enum import BaseEnum
from clinicadl.utils.json import read_json
from clinicadl.utils.objects import HasConfig, equal_if_config_equal
from clinicadl.utils.typing import PathType

from .file_type import BidsFileType

NO_FILE_FOUND = "no file found"


class DatasetType(str, BaseEnum):
    """DatasetTypes allowed by the BIDS specification
    (https://bids-specification.readthedocs.io/en/stable/glossary.html#datasettype-metadata)"""

    RAW = "raw"
    DERIVATIVE = "derivative"
    STUDY = "study"


class BidsConfig(ObjectConfig["Bids"]):
    """
    Config class for ``Bids``.
    """

    path: Path

    @field_validator("path", mode="after")
    def _resolve_path(cls, v: Path) -> Path:
        return v.resolve()

    @classmethod
    def _get_class(cls):
        return Bids


SemVer = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]


class DatasetDescription(ClinicaDLConfig):
    """
    The information stored in a BIDS' dataset description file.
    """

    name: str = Field(alias="Name")
    bids_version: SemVer = Field(alias="BIDSVersion")
    dataset_type: DatasetType = Field(alias="DatasetType")

    model_config = ConfigDict(
        validate_by_name=True,
        validate_by_alias=True,
        extra="allow",
    )

    @property
    def is_caps(self) -> bool:
        return "CAPSVersion" in self.__pydantic_extra__

    @classmethod
    def from_json(cls, json_path, **kwargs):
        return cls(**read_json(json_path))

    @model_validator(mode="after")
    def _validate_caps_version(self) -> Self:
        if self.is_caps:
            assert (
                self.dataset_type == DatasetType.DERIVATIVE
            ), f"If the directory is a CAPS, DatasetType must be 'derivative' in dataset_description.json. Got: '{self.dataset_type}'"

        return self


@equal_if_config_equal
class Bids(HasConfig[BidsConfig]):
    """
    A class to read :term:`BIDS` datasets or :term:`BIDS derivatives` (including :term:`CAPS`).

    The directory is expected to contain the mandatory :bids:`dataset_description.json <modality-agnostic-files/dataset-description.html#dataset_descriptionjson>`
    file, with the key ``"DatasetType"`` (whose value can be either ``"raw"``, `"`derivative"`` or ``"study"``).
    Depending on the value of ``"DatasetType"``, the expected organisation is different:

    - ``"raw"``: default organisation, the subject-specific folders are in the root directory. The tensors saved
      by ``ClinicaDL`` will be in ``derivatives/tensors``.
    - ``"study"``: :bids:`study organisation <common-principles.html#study-dataset>`, the subject-specific folders are in ``sourcedata/raw``. The tensors
      will be saved in ``derivatives/tensors``.
    - ``"derivative"``:

        - if ``"CAPSVersion"`` is in ``dataset_description.json``, the directory will be understood as a :term:`CAPS`. The subject-specific folders are
        expected in ``subjects``, and the tensors will be saved in ``../tensors``.
        - otherwise, it is interpreted as a classical BIDS derivative. The subject-specific folders are
        expected in the root directory, and the tensors will be saved in ``../tensors``.

    Parameters
    ----------
    directory : str | Path
        The path to the :term:`BIDS-like` directory.

    Examples
    --------
    The default BIDS organisation:

    .. code-block:: bash

        bids
        ├── dataset_description.json        <- contains "DatasetType": "raw"
        ├── sub-...
        ...
        └── derivatives
            └── tensors                     <- where tensors will be saved

    The "study" organization:

    .. code-block:: bash

        study
        ├── dataset_description.json        <- contains "DatasetType": "study"
        ├── derivatives
        │   └── tensors
        └── sourcedata
            └── raw
                └── sub-...

    A BIDS derivative:

        ├── derivative                      <- this path is passed to the Bids object
        │   ├── dataset_description.json    <- contains "DatasetType": "derivative"
        │   ├── sub-...
        │   ...
        └── tensors

    A CAPS:

    .. code-block:: bash

        ├── caps                            <- this path is passed to the Bids object
        │   ├── dataset_description.json    <- contains "CAPSVersion"
        │   └── subjects
        │       ├── sub-...
        │       ...
        └── tensors

    """

    _config_type = BidsConfig
    DATASET_DESC_FILENAME = "dataset_description.json"

    def __init__(self, path: PathType):
        self.config = self._config_type(path=path)
        self.path = self.config.path
        self.dataset_desc = self._read_bids_description(self.path)

    @classmethod
    def _read_bids_description(cls, bids_dir: Path) -> DatasetDescription:
        """
        Gets the DatasetType and determines whether the dataset is a CAPS or not.
        """
        data_desc_path = bids_dir / cls.DATASET_DESC_FILENAME
        if not data_desc_path.exists():
            raise FileNotFoundError(
                f"A BIDS (or a derivative) must contain a dataset_description.json. Nothing found at: {data_desc_path}"
            )
        return DatasetDescription.from_json(data_desc_path)

    @property
    def participants_dir(self) -> Path:
        """
        Where the subject-specific directories are stored.
        """
        if self.dataset_desc.is_caps:
            return self.path / "subjects"
        elif self.dataset_desc.dataset_type == DatasetType.STUDY:
            return self.path / "sourcedata" / "raw"
        return self.path

    @property
    def tensors_dir(self) -> Path:
        """
        Where the tensors produced by ``ClinicaDL`` are saved.
        """
        if self.dataset_desc.dataset_type == DatasetType.DERIVATIVE:
            return self.path.parent / "tensors"
        return self.path / "derivatives" / "tensors"

    def get_path(
        self,
        file_type: BidsFileType,
        participant: Optional[str] = None,
        session: Optional[str] = None,
    ) -> Path:
        """
        To get the path of a file in the BIDS-like directory.

        The specifications of the file to found are given via a :py:class:`~clinicadl.io.BidsFileType`.

        The user can also give participant id and session id if the wanted file is
        subject- and session- specific.

        Parameters
        ----------
        file_type : BidsFileType
            The :py:class:`~clinicadl.io.BidsFileType` containing the specifications of the file the user
            is looking for.
        participant : Optional[str], default=None
            The participant id (e.g., ``"sub-xxx"``), if the file is subject-specific.
        session : Optional[str], default=None
            The session id (e.g., ``"ses-xxx"``), if the file is subject-specific.

        Returns
        -------
        Path
            The file matching all the requirements.

        Raises
        ------
        RuntimeError
            If no correspond file is found, or if several corresponding files are found.

        Examples
        --------
        .. code-block:: bash

            bids
            ├── sub-001
            │   ├── ses-M000
            │   │   └── anat
            │   │   │   └── sub-001_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii
            │   │   └── sub-001_ses-M000_scans.tsv
            │   └── sub-001_sessions.tsv
            ...
            └── space-MNI152NLin2009cSym_participants.tsv

        .. code-block:: python

            >>> from clinicadl.io import Bids, BidsFileType
            >>> bids = Bids("bids")
            >>> bids.get_path(
                    file_type=BidsFileType(
                        suffix="T1w",
                        data_type="anat",
                        with_entities={"space": "MNI152NLin2009cSym"},
                    ),
                    participant="sub-001",
                    session="ses-M000",
                )
            Path("bids/sub-001/ses-M000/anat/sub-001_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii")
            >>> bids.get_path(
                    file_type=BidsFileType(
                        suffix="scans",
                        extension=".tsv",
                    ),
                    participant="sub-001",
                    session="ses-M000",
                )
            Path("bids/sub-001/ses-M000/sub-001_ses-M000_scans.tsv")
            >>> bids.get_path(
                    file_type=BidsFileType(
                        suffix="sessions",
                        extension=".tsv",
                    ),
                    participant="sub-001",
                )
            Path("bids/sub-001/sub-001_sessions.tsv")
            >>> bids.get_path(
                    file_type=BidsFileType(
                        suffix="participants",
                        extension=".tsv",
                        with_entities={"space": "MNI152NLin2009cSym"},
                    ),
                )
            Path("bids/space-MNI152NLin2009cSym_participants.tsv")
        """
        dir_ = self._find_root(participant, session)

        selected_files = []
        for root, _, files in os.walk(dir_):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dir_)
                try:
                    if file_type.match(rel_path, participant, session):
                        selected_files.append(Path(full_path))
                except (ValueError, AssertionError):  # not a BIDS file
                    continue

        error_msg = f"For ({participant} | {session}), an error occurred while trying to get {file_type}: "
        if len(selected_files) > 1:
            error_msg += "more than 1 file found:\n"
            for found_file in selected_files:
                error_msg += f"\t * {found_file}\n"
            raise RuntimeError(error_msg)

        elif len(selected_files) == 0:
            error_msg += NO_FILE_FOUND
            raise RuntimeError(error_msg)

        else:
            return selected_files[0]

    def has_file_type(
        self, participant: str, session: str, file_type: BidsFileType
    ) -> bool:
        """
        To check if a participant has a file type for a specified session.

        In practice, it will just check that :py:meth:`get_path` returns a
        file.

        Parameters
        ----------
        participant : str
            The participant id (e.g., ``"sub-xxx"``).
        session : str
            The session id (e.g., ``"ses-xxx"``).
        file_type : BidsFileType
            The :py:class:`~clinicadl.io.BidsFileType` containing the specifications of the file to
            check.

        Returns
        -------
        bool
            Whether the participant has a file type for the specified session.

        Raises
        ------
        RuntimeError
            If several corresponding files are found.
        """
        try:
            self.get_path(file_type, participant, session)
        except RuntimeError as e:
            if NO_FILE_FOUND in str(e):
                return False
            raise

        return True

    def build_path(
        self,
        file_type: BidsFileType,
        participant: Optional[str] = None,
        session: Optional[str] = None,
    ) -> Path:
        """
        Builds the path to the file associated with the input file type,
        and the potential participant and session ids.

        Parameters
        ----------
        file_type : BidsFileType
            The :py:class:`~clinicadl.io.BidsFileType` containing the specifications of the path to create.

            .. note::
                The entities in the ``without_entities`` attribute of the :py:class:`~clinicadl.io.BidsFileType`
                are not used here.
        participant : Optional[str], default=None
            The participant id (e.g., ``"sub-xxx"``), if the file must be subject-specific.
        session : Optional[str], default=None
            The session id (e.g., ``"ses-xxx"``), if the file must be subject-specific.

        Returns
        -------
        Path
            The built path.

        Examples
        --------
        .. code-block:: python

            >>> from clinicadl.io import Bids, BidsFileType
            >>> bids = Bids("bids")
            >>> bids.build_path(
                    file_type=BidsFileType(
                        suffix="T1w",
                        data_type="anat",
                        with_entities={"space": "MNI152NLin2009cSym", "res": "1x1x1},
                        extension="nii",
                    ),
                    participant="sub-001",
                    session="ses-M000",
                )
            Path("bids/sub-001/ses-M000/anat/sub-001_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii")
            >>> bids.build_path(
                    file_type=BidsFileType(
                        suffix="scans",
                        extension=".tsv",
                    ),
                    participant="sub-001",
                    session="ses-M000",
                )
            Path("bids/sub-001/ses-M000/sub-001_ses-M000_scans.tsv")
            >>> bids.build_path(
                    file_type=BidsFileType(
                        suffix="sessions",
                        extension=".tsv",
                    ),
                    participant="sub-001",
                )
            Path("bids/sub-001/sub-001_sessions.tsv")
            >>> bids.build_path(
                    file_type=BidsFileType(
                        suffix="participants",
                        extension=".tsv",
                        with_entities={"space": "MNI152NLin2009cSym"},
                    ),
                )
            Path("bids/space-MNI152NLin2009cSym_participants.tsv")

        """
        dir_ = self._find_root(participant, session)

        filename_components = (
            [
                BidsEntity.from_key_value(key, value.pattern)
                for key, value in file_type.with_entities.items()
            ]
            if file_type.with_entities
            else []
        )
        filename_components.append(file_type.suffix.pattern)
        if session:
            filename_components.insert(0, session)
        if participant:
            filename_components.insert(0, participant)

        filename = "_".join(filename_components)
        folder = Path(file_type.data_type.pattern) if file_type.data_type else Path(".")

        return (dir_ / folder / filename).with_suffix(file_type.extension.pattern)

    def get_participants_sessions(
        self,
        file_type: BidsFileType,
    ) -> set[tuple[str, str]]:
        """
        Finds all the (participant, session) pairs which have a file matching a
        :py:class:`~clinicadl.io.BidsFileType`.

        In practice, it will get all the (participant, session) pairs for which
        :py:meth:`has_file_type` returns ``True``.

        Parameters
        ----------
        file_type : BidsFileType
            The :py:class:`~clinicadl.io.BidsFileType` to match.

        Returns
        -------
        set[tuple[str, str]]
            The (participant, session) pairs that have the specified file type.
        """
        participants_sessions = self.get_all_participants_sessions()
        with_data_type = set()
        for participant, session in participants_sessions:
            if self.has_file_type(participant, session, file_type):
                with_data_type.add((participant, session))

        return with_data_type

    def get_all_participants_sessions(
        self,
    ) -> set[tuple[str, str]]:
        """
        Finds all the (participant, session) pairs in the BIDS-like directory.

        Returns
        -------
        set[tuple[str, str]]
            All the (participant, session) pairs.
        """
        participant_pattern = re.compile(Subject.pattern)
        session_pattern = re.compile(Session.pattern)
        participants_sessions = set()

        for f in os.scandir(self.participants_dir):
            if not f.is_dir():
                continue
            if not participant_pattern.match(f.name):
                continue

            for f_ in os.scandir(f.path):
                if not f_.is_dir():
                    continue
                if not session_pattern.match(f_.name):
                    continue

                participants_sessions.add((f.name, f_.name))

        return participants_sessions

    def _find_root(
        self, participant: Optional[str] = None, session: Optional[str] = None
    ) -> Path:
        """
        Depending on the participant and session specifications, find the root directory
        to consider.
        """
        if participant:
            sub = Subject(participant)
            root = self.participants_dir / sub
            if session:
                ses = Session(session)
                root /= ses

        else:
            assert session is None, "Cannot pass a session without a participant"
            root = self.path

        return root
