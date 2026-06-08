from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeVar, Union

import pandas as pd
import torch.utils.data
from typing_extensions import Self

from clinicadl.utils.dictionary.words import PARTICIPANT_ID, SESSION_ID
from clinicadl.utils.objects import JsonReaderWriter
from clinicadl.utils.tsvtools import read_df
from clinicadl.utils.typing import DataFrameType

from ..structures import Sample

SampleT = TypeVar("SampleT", Sample, Sequence[Sample], dict[Any, Sample])


class Dataset(JsonReaderWriter, ABC, torch.utils.data.Dataset[SampleT]):
    """
    Abstract class for ``ClinicaDL`` datasets, which inherits from :py:class:`torch.utils.data.Dataset`,
    to work with 3D neuroimaging data.

    To work properly with ``ClinicaDL``, all datasets must inherit from this class.

    See Also
    --------
    :py:class:`~clinicadl.data.datasets.BidsDataset`
        A ``Dataset`` to read data organized in a :term:`BIDS`.
    """

    _df: pd.DataFrame

    @property
    def df(self) -> pd.DataFrame:
        """
        A DataFrame containing metadata on the images present in the dataset.

        Each image must have its associated line in the DataFrame, which must contain at least the columns
        "participant_id" and "session_id", with the ids (strings) of the participant and the session.

        Example
        -------
        .. code-block:: text

            participant_id  session_id   age   sex   diagnosis
            sub-001         ses-M000     55.0  M     CN
            sub-001         ses-M003     55.0  M     AD
            sub-002         ses-M000     62.0  F     MCI
            sub-002         ses-M003     62.0  F     AD
            sub-003         ses-M000     67.0  F     CN
        """
        return self._df

    @abstractmethod
    def eval(self) -> None:
        """
        Sets the dataset to evaluation mode.

        For example, disabling data augmentation in the transformation pipeline.
        """

    @abstractmethod
    def train(self) -> None:
        """
        Sets the dataset to training mode.

        For example, enabling data augmentation in the transformation pipeline.
        """

    def get_participant_session_couples(self) -> set[tuple[str, str]]:
        """
        Retrieves all (participant, session) pairs in the dataset.

        Returns
        -------
        set[tuple[str, str]]
            The set of (participant, session).
        """
        return set(zip(self.df[PARTICIPANT_ID], self.df[SESSION_ID]))

    def subset(
        self, participants_sessions: Union[DataFrameType, Iterable[tuple[str, str]]]
    ) -> Self:
        """
        To get a subset of the dataset from a list of (participant, session) pairs.

        Parameters
        ----------
        data : Union[DataFrameType, Sequence[tuple[str, str]]]
            Can be either:

            - a sequence of (participant, session);
            - a :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of (participant, session)
              pairs to extract. This list must be passed via two columns named ``"participant_id"``
              and ``"session_id"`` (other columns won't be considered).

        Returns
        -------
        Self
            A subset of the original dataset, restricted to the (participant, session) pairs mentioned in ``data``.
        """
        if isinstance(participants_sessions, (str, Path)):
            new_df = read_df(participants_sessions)
        elif isinstance(participants_sessions, pd.DataFrame):
            new_df = participants_sessions[
                [PARTICIPANT_ID, SESSION_ID]
            ].drop_duplicates()
        else:
            new_df = pd.DataFrame(
                participants_sessions, columns=[PARTICIPANT_ID, SESSION_ID]
            ).drop_duplicates()

        new_df = new_df.set_index([PARTICIPANT_ID, SESSION_ID])

        df = self.df.set_index([PARTICIPANT_ID, SESSION_ID])
        subset_df = df.loc[new_df.index.intersection(df.index)].reset_index()

        if len(subset_df) == 0:
            raise RuntimeError(
                "No (participant, session) pairs are in the dataset. This would lead to an empty dataset!"
            )

        dataset = deepcopy(self)
        dataset._df = subset_df

        return dataset

    @abstractmethod
    def get_sample_info(self, idx: int, column: str) -> Any:
        """
        Retrieves information on a given sample in the metadata DataFrame. The information
        corresponds to the information on the image the sample was extracted
        from.

        Parameters
        ----------
        idx : int
            The index of the sample in the dataset.
        column : str
            The information to look for, i.e. a column of :py:attr:`df`.

        Returns
        -------
        Any
            The value of the column for this sample.
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        Computes the total number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples in the dataset, i.e. the number of images
            times the number of samples per image.
        """

    @abstractmethod
    def __getitem__(self, idx: int) -> SampleT:
        """
        Retrieves the sample at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.

        Returns
        -------
        Union[Sample, Sequence[Sample], dict[Any, Sample]]
            A structured output containing the processed data and metadata, as a
            :py:class:`~clinicadl.data.structures.Sample`, or a sequence or dictionary
            of samples.
        """

    def _check_idx(self, idx: int) -> None:
        """
        Checks that a sample index is valid.
        """
        if not isinstance(idx, int) or idx < 0:
            raise IndexError(f"Index must be a non-negative integer, got {idx}.")
        if idx >= len(self):
            raise IndexError(
                f"Index out of range, there are only {len(self)} samples in total in the dataset."
            )

    def _check_column(self, column: str) -> None:
        """
        Checks that the wanted metadata exists.
        """
        if column not in self.df.columns:
            raise KeyError(
                f"No column named '{column}' in the metadata DataFrame. Present columns are: "
                f"{list(self.df.columns)}"
            )
