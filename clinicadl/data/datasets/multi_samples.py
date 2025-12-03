from bisect import bisect_right
from copy import deepcopy
from typing import Any, Sequence, Union

import pandas as pd
from typing_extensions import Self

from clinicadl.dictionary.words import (
    N_SAMPLES,
    PARTICIPANT_ID,
    SESSION_ID,
)
from clinicadl.tsvtools.utils import read_data
from clinicadl.utils.typing import DataFrameType

from .abstract import ClinicaDLDataset


class MultiSamplesDataset(ClinicaDLDataset):
    """
    An abstract :py:class:`~clinicadl.data.datasets.ClinicaDLDataset` to handle multiple samples per image.

    Here the size of the dataset is not equal to the number of images, since an image can contain multiple samples
    (e.g. patches or slices).

    The key idea of this dataset is that a column of the metadata DataFrame is expected to have a column named "n_samples", with
    the number of samples for each image. If the DataFrame doesn't have this column, ``SamplesDataset`` will try to call
    :py:meth:`_count_samples`.
    """

    _df: pd.DataFrame

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def _has_len(self) -> bool:
        """Whether the number of samples for each image is known."""
        return N_SAMPLES in self.df.columns

    def subset(
        self, particpants_sessions: Union[DataFrameType, Sequence[tuple[str, str]]]
    ) -> Self:
        if isinstance(particpants_sessions, Sequence):
            new_df = pd.DataFrame.from_records(
                particpants_sessions, columns=[PARTICIPANT_ID, SESSION_ID]
            ).drop_duplicates()
        else:
            new_df = read_data(particpants_sessions, check_protected_names=False)

        new_df = new_df.set_index([PARTICIPANT_ID, SESSION_ID])

        df = self._df.set_index([PARTICIPANT_ID, SESSION_ID])
        subset_df = df.loc[new_df.index.intersection(df.index)].reset_index()

        if len(subset_df) == 0:
            raise RuntimeError(
                "No (participant, session) pairs are in the dataset. This would lead to an empty dataset!"
            )

        dataset = deepcopy(self)
        dataset._df = subset_df

        return dataset

    def get_sample_info(self, idx: int, column: str) -> Any:
        self._check_has_len()
        self._check_idx(idx)
        self._check_column(column)

        image_idx = self._get_image_idx(idx)
        row = self.df.iloc[image_idx]

        value = row.at[column]
        try:
            return value.item()  # e.g. convert np.int to int
        except AttributeError:
            return value

    def get_participant_session_couples(self) -> set[tuple[str, str]]:
        return set(zip(self._df[PARTICIPANT_ID], self._df[SESSION_ID]))

    def __len__(self) -> int:
        self._check_has_len()

        return int(self.df[N_SAMPLES].sum())

    def _check_has_len(self) -> None:
        """
        Checks that the length of the dataset has been computed,
        otherwise tries to compute it with :py:meth:`_count_samples`.
        """
        if not self._has_len:
            self._count_samples()

    def _get_index_in_image(self, idx: int) -> int:
        """
        Determines the the index of this sample in its original image.

        E.g.: if every image has 3 samples, then self._get_index_in_image(4)=1,
        because the first image contains samples 0, 1 and 2, and the second
        image contains the samples 3, 4 and 5, so 4 is the 2nd sample
        of its image.
        """
        image_idx = self._get_image_idx(idx)
        if image_idx > 0:
            return int(idx - self.df[N_SAMPLES].cumsum().iat[image_idx - 1])

        return idx

    def _get_image_idx(self, idx: int) -> int:
        """
        Gets the image in the dataset corresponding to the index
        of the sample.
        """
        return bisect_right(self.df[N_SAMPLES].cumsum(), idx)

    def _get_image_info(self, participant: str, session: str, column: str) -> Any:
        """
        Returns the value of a column for a (participant, session).
        """
        self._check_column(column)

        return self.df.set_index([PARTICIPANT_ID, SESSION_ID]).at[
            (participant, session), column
        ]

    def _get_indices_associated_to(
        self, participant: str, session: str
    ) -> tuple[int, int]:
        """
        Returns the value of the range of indices associated to the image.
        """
        cumsum = self.df.set_index([PARTICIPANT_ID, SESSION_ID])[N_SAMPLES].cumsum()
        min_idx = cumsum.shift(1).fillna(0).at[(participant, session)]
        max_idx = max(cumsum.at[(participant, session)] - 1, 0)

        return int(min_idx), int(max_idx)

    def _count_samples(self) -> None:
        """
        Gets the number of samples for each image and puts
        it in the metadata DataFrame in the column 'n_samples'.
        """
        raise NotImplementedError(
            "_count_samples must be implemented if there is no column named 'n_samples' in the metadata DataFrame."
        )
