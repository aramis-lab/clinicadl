from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Optional, Pattern

from pydantic import StringConstraints, field_serializer

from clinicadl.utils.bids import BidsFile, Session, Subject
from clinicadl.utils.config import ClinicaDLConfig

AlphanumericStr = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9]+$", strip_whitespace=False)
]


class BidsFileType(ClinicaDLConfig):
    """
    To define the files you are interested in within your :term:`BIDS` dataset.

    A ``BidsFileType`` can refer either to data files or metadata files.

    :py:meth:`match` method determines if a file matches the specifications defined in
    the current ``BidsFileType``.

    Parameters
    ----------
    suffix : str | Pattern
        The :bids:`BIDS suffix <common-principles.html#definitions>` of the relevant files.
        Regular expressions are accepted.

    data_type : Optional[str | Pattern], default=None
        The :bids:`BIDS data type <common-principles.html#definitions>`, which is the folder
        where the relevant files are stored (e.g., ``"anat"`` if the relevant files are structural
        imaging files). Regular expressions are accepted.\n
        If ``None``, the files are expected to be at the root of the directory that is being
        explored.

    extension : str | Pattern, default=re.compile(".nii.*")
        The file extension of the relevant files. Regular expressions are accepted.

    with_entities : Optional[dict[AlphanumericStr, str | Pattern]], default=None
        The :bids:`BIDS entities <common-principles.html#entities>` that must contain the relevant files.
        More concretely, if ``with_entities={"trc": "18FFDG"}``, all the files with ``trc-18FFDG`` in their
        filenames are candidate. Regular expressions are accepted for the entity values.

        .. important::
            - No need to mention the entities ``"sub"`` and ``"ses"`` here.

    without_entities : Optional[dict[AlphanumericStr, str | Pattern]], default=None
        The :bids:`BIDS entities <common-principles.html#entities>` that must not contain the relevant files.
        More concretely, if ``without_entities={"trc": "18FFDG"}``, all the files with ``trc-18FFDG`` are excluded.
        Regular expressions are accepted for the entity values.

    description : Union[str], default=None
        A potential description of the files.
    """

    suffix: Pattern
    data_type: Optional[Pattern] = None
    extension: Pattern = re.compile(".nii.*")
    with_entities: Optional[dict[AlphanumericStr, Pattern]] = None
    without_entities: Optional[dict[AlphanumericStr, Pattern]] = None
    description: Optional[str] = None

    def match(
        self,
        path: str | Path,
        participant: Optional[str] = None,
        session: Optional[str] = None,
    ) -> bool:
        """
        Checks whether the input path matches the current ``BidsFileType``, for the
        (participant, session) pair if specified.

        The path is the not the path relative to the term:`BIDS` directory, but the path relative
        to the direct parent of the ``data_type`` folder (see examples).

        Parameters
        ----------
        path : str | Path
            The path to check.
        participant : Optional[str], default=None
            The participant id (e.g., "sub-xxx").
        session : Optional[str], default=None
            The session id (e.g., "ses-xxx").

        Returns
        -------
        bool
            Whether the input path matches the specifications defined in the current ``BidsFileType``,
            and the participant id and session id if specified.

        Examples
        --------
        Looking for data files:

        .. code-block::

            >>> file_type = BidsFileType(
                    data_type="anat",
                    suffix="T1w",
                    extension=".nii.gz",
                    with_entities={"space": "MNI152.*", "res": "1x1x1"},
                    without_entities={"desc": "Crop"},
                )
            >>> file_type.match("anat/sub-000_ses-M000_space-MNI152_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            True
            >>> file_type.match("anat/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            True    # 'space' still matches the pattern
            >>> file_type.match("anat/sub-001_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            False   # not the right subject
            >>> file_type.match("anat/sub-000_ses-M001_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            False   # not the right session
            >>> file_type.match("anat/sub-000_ses-M000_space-MNI152NLin2009cSym_run-1_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            True    # 'run' is not in without_entities, so its presence is not disqualifying
            >>> file_type.match("anat/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii.gz", participant="sub-000", session="ses-M000")
            False   # not the right suffix
            >>> file_type.match("mri/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            False   # not the right data_type
            >>> file_type.match("sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            False   # not the right data_type
            >>> file_type.match("anat/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii", participant="sub-000", session="ses-M000")
            False   # not the right suffix
            >>> file_type.match("anat/sub-000_ses-M000_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            False   # 'space' is missing
            >>> file_type.match("anat/sub-000_ses-M000_space-MNI152NLin2009cSym_res-2x2x2_T1w.nii.gz", participant="sub-000", session="ses-M000")
            False   # not the right value for 'res'
            >>> file_type.match("anat/sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz", participant="sub-000", session="ses-M000")
            False   # contains an entity that is in without_entities

        .. code-block::

            >>> file_type = BidsFileType(
                    data_type="dwi/dti_based_processing/normalized_space",
                    suffix="MD",
                    with_entities={"space": "b0"},
                )
            >>> file_type.match("dwi/dti_based_processing/normalized_space/sub-001_ses-M001_space-b0_MD.nii")
            True

        Looking for metadata files:

        .. code-block::

            >>> file_type = BidsFileType(
                    suffix="sessions",
                    extension=".tsv",
                    with_entities={"space": "MNI152.*"},
                )
            >>> file_type.match("sessions.tsv")
            False
            >>> file_type.match("space-MNI152NLin2009cSym_sessions.tsv")
            True
        """
        file = BidsFile(path)

        if self.data_type and not self.data_type.fullmatch(str(file.path.parent)):
            return False
        elif not self.data_type and file.path.parent != Path("."):
            return False

        if self.extension is not None:
            if not self.extension.fullmatch(file.extension):
                return False

        if not self.suffix.fullmatch(file.suffix):
            return False

        if self.without_entities is not None:
            for key, value in file.entities.items():
                if key in self.without_entities and self.without_entities[
                    key
                ].fullmatch(value):
                    return False

        with_entities = self.with_entities or {}

        if participant is not None:
            sub = Subject(participant)
            with_entities[sub.key] = re.compile(sub.value)

        if session is not None:
            ses = Session(session)
            with_entities[ses.key] = re.compile(ses.value)

        for key, value in with_entities.items():
            if key not in file.entities:
                return False
            if not value.fullmatch(file.entities[key]):
                return False

        return True

    @field_serializer("data_type", "suffix", "extension")
    def _serialize_pattern(self, pattern: Pattern) -> str:
        """
        Serializes a pattern.
        """
        return pattern.pattern

    @field_serializer("without_entities", "with_entities")
    def _serialize_dict_patterns(
        self, patterns: Optional[dict[AlphanumericStr, Pattern]]
    ) -> Optional[dict[str, str]]:
        """
        Serializes a dict of patterns.
        """
        if patterns is None:
            return None
        return {
            key: self._serialize_pattern(pattern) for key, pattern in patterns.items()
        }
