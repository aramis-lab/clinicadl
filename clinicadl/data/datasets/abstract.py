from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import pandas as pd
import torch.utils.data
from typing_extensions import Self

from clinicadl.utils.objects import JsonReaderWriter
from clinicadl.utils.typing import DataFrameType

from ..structures import Sample


class Dataset(
    JsonReaderWriter, ABC, torch.utils.data.Dataset[Union[Sample, Sequence[Sample]]]
):
    """
    Abstract class for ``ClinicaDL`` datasets, which inherits from :py:class:`torch.utils.data.Dataset`,
    to work with 3D neuroimaging data.

    To work properly with ``ClinicaDL``, all datasets must inherit from this class.

    See Also
    --------
    :py:class:`~clinicadl.data.datasets.BaseDataset`
        A ``Dataset`` with the base logic of all datasets implemented natively in ``ClinicaDL``.
        May be easier to override than the plain ``Dataset``.
    """

    @property
    @abstractmethod
    def df(self) -> pd.DataFrame:
        """
        A DataFrame containing metadata on the images present in the dataset.

        Each image must have its associated line in the DataFrame, which must contain at least the columns
        "participant_id" and "session_id", with respectively the id (a string) of the participant and the session.

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

    @abstractmethod
    def eval(self) -> None:
        """
        Sets the dataset to evaluation mode.

        It disables data augmentation in the transformation pipeline.
        """

    @abstractmethod
    def train(self) -> None:
        """
        Sets the dataset to training mode.

        It enables data augmentation in the transformation pipeline.
        """

    @abstractmethod
    def subset(
        self, particpants_sessions: Union[DataFrameType, Sequence[tuple[str, str]]]
    ) -> Self:
        """
        To get a subset of the dataset from a list of (participant, session) pairs.

        Parameters
        ----------
        data : Union[DataFrameType, Sequence[tuple[str, str]]]
            Can be either:

            - a **sequence of (participant, session)**;
            - a :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of (participant, session)
              pairs to extract. This list must be passed via two columns named ``"participant_id"``
              and ``"session_id"`` (other columns won't be considered).

        Returns
        -------
        Self
            A subset of the original dataset, restricted to the (participant, session) pairs mentioned in ``data``.
        """

    def get_sample_info(self, idx: int, column: str) -> Any:
        """
        Retrieves information on a given sample. The information will
        correspond to the information on the image the sample was extracted
        from.

        Parameters
        ----------
        idx : int
            The index of the sample in the dataset.
        column : str
            The information to look for, i.e. a column of the DataFrame containing
            the metadata.

        Returns
        -------
        Any
            The information (e.g. the age, the sex, etc.)
        """

    def describe(self) -> Any:
        """
        Returns a description of the dataset.

        Returns
        -------
        Any
            The description of the dataset (e.g. a tuple with its length and the (participant, session) inside).
        """
        raise NotImplementedError(
            f"'describe' not implemented in {type(self).__name__}"
        )

    @abstractmethod
    def get_participant_session_couples(self) -> set[tuple[str, str]]:
        """
        Retrieves all (participant, session) pairs in the dataset.

        Returns
        -------
        set[tuple[str, str]]
            The set of (participant, session).
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
    def __getitem__(
        self, idx: int
    ) -> Union[Sample, Sequence[Sample], dict[Any, Sample]]:
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
            :py:class:`~clinicadl.data.datasets.output.Sample`, or a sequence or dictionary
            of such outputs.
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
