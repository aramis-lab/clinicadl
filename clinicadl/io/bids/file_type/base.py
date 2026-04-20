from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Optional, Pattern

from pydantic import StringConstraints, field_serializer

from clinicadl.utils.bids import BidsEntity, Session, Subject
from clinicadl.utils.config import ClinicaDLConfig

AlphanumericStr = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9]+$", strip_whitespace=False)
]


class BidsFileType(ClinicaDLConfig):
    """
    To define the files you are interested in within your :term:`BIDS` dataset.

    Parameters
    ----------
    datatype : str | Pattern
        The :bids:`BIDS data type <common-principles.html#definitions>`, which is the folder
        where the relevant files are stored. Regular expressions are accepted.

    suffix : str | Pattern
        The :bids:`BIDS data type <common-principles.html#definitions>` of the relevant files.
        Regular expressions are accepted.

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
        A potential description of the data.

    Examples
    --------
    .. code-block::

        BidsFileType(
            datatype="anat",
            suffix="T1w",
            extension=".nii.gz",
            with_entities={"space": "MNI152.*", "res": "1x1x1"},
            without_entities={"desc": "Crop"},
        )
        # will match:
        #   bids/sub-000/ses-M000/anat/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz
        #   bids/sub-000/ses-M000/anat/sub-000_ses-M000_res-1x1x1_space-MNI152NLin2009cSym_T1w.nii.gz
        #   bids/sub-001/ses-M001/anat/sub-001_ses-M001_space-MNI152_res-1x1x1_T1w.nii.gz
        #   bids/sub-010/ses-M000/anat/sub-010_ses-M000_space-MNI152NLin2009cSym_run-1_res-1x1x1_T1w.nii.gz

        # will not match
        #   bids/sub-000/ses-M000/anat/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii.gz
        #   bids/sub-000/ses-M000/anat_/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz
        #   bids/sub-000/ses-M000/anat/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii
        #   bids/sub-000/ses-M000/anat/sub-000_ses-M000_res-1x1x1_T1w.nii.gz
        #   bids/sub-000/ses-M000/anat/sub-000_ses-M000_space-MNI152NLin2009cSym_res-2x2x2_T1w.nii.gz
        #   bids/sub-000/ses-M000/anat/sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz
    """

    datatype: Pattern
    suffix: Pattern
    extension: Pattern = re.compile(".nii.*")
    with_entities: Optional[dict[AlphanumericStr, Pattern]] = None
    without_entities: Optional[dict[AlphanumericStr, Pattern]] = None
    description: Optional[str] = None

    def match(self, path: str | Path, participant: str, session: str) -> bool:
        """
        Checks whether the input path matches the current ``BidsFileType`` for the
        specified (participant, session) pair.

        The path must be relative to the participant-session folder (i.e., ``anat/sub-001_ses-M001_T1w.nii.gz``
        and not ``bids/sub-001/ses-M001/anat/sub-001_ses-M001_T1w.nii.gz``).

        Parameters
        ----------
        path : str | Path
            The path to test.
        participant : str
            The participant id (e.g., "sub-xxx").
        session : str
            The session id (e.g., "ses-xxx").

        Returns
        -------
        bool
            Whether the input path matches is associated with (``participant``, ``session``) and
            matches the specifications defined in the current ``BidsFileType``.
        """
        path = Path(path)
        sub = Subject(participant)
        ses = Session(session)

        if not self.datatype.fullmatch(str(path.parent)):
            return False

        if self.extension is not None:
            extension = self._get_extension(path.name)
            if not self.extension.fullmatch(extension):
                return False

        suffix = self._get_suffix(path.name)
        if not self.suffix.fullmatch(suffix):
            return False

        entities = self._get_entities(path.name)

        if self.without_entities is not None:
            for key, value in entities.items():
                if key in self.without_entities and self.without_entities[
                    key
                ].fullmatch(value):
                    return False

        with_entities = self.with_entities or {}
        with_entities[sub.key], with_entities[ses.key] = (
            re.compile(sub.value),
            re.compile(ses.value),
        )
        for key, value in with_entities.items():
            if key not in entities:
                return False
            if not value.fullmatch(entities[key]):
                return False

        return True

    @staticmethod
    def _get_extension(filename: str) -> str:
        """
        Gets the total extension of a file.
        'file.nii.gz' will return '.nii.gz'.
        """
        return "".join(filename.partition(".")[1:])

    @staticmethod
    def _get_suffix(filename: str) -> str:
        """
        Gets the suffix from a filename.
        """
        return filename.partition(".")[0].split("_")[-1]

    @staticmethod
    def _get_entities(filename: str) -> dict[str, str]:
        """
        Gets all the (key, value) entities from a filename.
        """
        entities = filename.partition(".")[0].split("_")[:-1]
        entities = [BidsEntity(entity) for entity in entities]

        return {entity.key: entity.value for entity in entities}

    @field_serializer("datatype", "suffix", "extension")
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
