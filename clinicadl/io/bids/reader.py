from __future__ import annotations

import os
import re
from pathlib import Path

from clinicadl.data.datatypes import DataType
from clinicadl.utils.dictionary.suffixes import JSON, TSV
from clinicadl.utils.enum import BaseEnum
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.tsvtools import create_participants_sessions_df, df_to_tsv
from clinicadl.utils.typing import PathType

NO_FILE_FOUND = "no file found"


class DatasetType(BaseEnum):
    """DatasetTypes allowed by the BIDS specification
    (https://bids-specification.readthedocs.io/en/stable/glossary.html#datasettype-metadata)"""

    RAW = "raw"
    DERIVATIVE = "derivative"
    STUDY = "study"


class Bids:
    """
    - In a classical :term:`BIDS`, it is in the root directory
    of the dataset.

    - In a :term:`study dataset`, it is in the
    """

    def __init__(self, directory: PathType):
        self.directory = Path(directory)
        self.dataset_type, self.is_caps = self._read_bids_type(self.directory)

    @staticmethod
    def _read_bids_type(bids_dir: Path) -> tuple[DatasetType, bool]:
        """
        Gets the DatasetType and determines whether the dataset is a CAPS or not.
        """
        data_desc_path = (bids_dir / "dataset_description").with_suffix(JSON)
        if not data_desc_path.exists():
            raise FileNotFoundError(
                f"A BIDS (or a derivative) must contain a dataset_description.json. Nothing found at: {data_desc_path}"
            )
        data_desc = read_json(data_desc_path)
        assert (
            "DatasetType" in data_desc
        ), "dataset_description.json must contain 'DatasetType'"
        dataset_type = DatasetType(data_desc["DatasetType"])
        is_caps = True if "CAPSVersion" in data_desc else False
        if is_caps:
            assert (
                dataset_type == DatasetType.DERIVATIVE
            ), "if the directory is a CAPS, DatasetType must be 'derivative' in dataset_description.json"

        return dataset_type, is_caps

    @property
    def participants_dir(self) -> Path:
        """
        Where the participant directories are stored.
        """
        if self.is_caps:
            return self.directory / "subjects"
        elif self.dataset_type == DatasetType.STUDY:
            return self.directory / "sourcedata" / "raw"
        return self.directory

    @property
    def tensors_dir(self) -> Path:
        """
        Where the tensors produced by ``ClinicaDL`` are saved.
        """
        if self.dataset_type == DatasetType.DERIVATIVE:
            return self.directory.parent / "tensors"
        return self.directory / "derivatives" / "tensors"

    def get_image_path(
        self, participant: str, session: str, datatype: DataType
    ) -> Path:
        participant_session_path = self.participants_dir / participant / session

        selected_files = []
        for root, _, files in os.walk(participant_session_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, participant_session_path)
                if datatype.pattern.match(str(rel_path)):
                    selected_files.append(full_path)

        error_msg = f"For ({participant} | {session}), an error occurred while trying to get {datatype}: "
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

    def create_participants_sessions_tsv(self, datatype: DataType) -> None:
        filneame = f"desc-{datatype.name}_participantsXsessions"
        participants_sessions = self.get_participants_sessions(datatype)
        df = create_participants_sessions_df(participants_sessions)
        df_to_tsv((self.directory / filneame).with_suffix(TSV), df)
        write_json(
            (self.directory / filneame).with_suffix(JSON),
            {"DataType": datatype.to_dict()},
        )

    def has_datatype(self, participant: str, session: str, datatype: DataType) -> bool:
        try:
            self.get_image_path(participant, session, datatype)
        except RuntimeError as e:
            if NO_FILE_FOUND in e:
                return False
            raise

        return True

    def get_tensor_path(
        self,
        participant: str,
        session: str,
        tensor_conversion: str,
        check_exists: bool = True,
    ) -> Path:
        path = (
            self.tensors_dir
            / participant
            / session
            / tensor_conversion
            / f"{participant}_{session}_tensors.pt"
        )
        if check_exists and not path.exists():
            raise FileNotFoundError(
                f"No tensors associated to {tensor_conversion} for ({participant}, {session})"
            )

        return path

    def get_participants_sessions(
        self,
        datatype: DataType,
    ) -> set[tuple[str, str]]:
        """
        Finds all the (participant, session) for a specific preprocessing.
        """
        participants_sessions = self.get_all_participants_sessions()
        with_datatype = set()
        for participant, session in participants_sessions:
            if self.has_datatype(participant, session, datatype):
                with_datatype.add((participant, session))

        return with_datatype

    def get_all_participants_sessions(
        self,
    ) -> set[tuple[str, str]]:
        """
        Finds all the (participant, session).
        """
        participant_pattern = re.compile("sub-.*")
        session_pattern = re.compile("ses-.*")
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
