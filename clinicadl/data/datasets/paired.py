from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from pydantic import field_validator

from clinicadl.utils.dictionary.words import (
    PARTICIPANT_ID,
    SESSION_ID,
)

from ..structures import Sample
from .base import Dataset
from .collection import CollectionDataset, CollectionDatasetConfig


class PairedDatasetConfig(CollectionDatasetConfig):
    """
    Config class for ``PairedDataset``.
    """

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_at_least_two_datasets(
        cls, datasets: tuple[Dataset, ...]
    ) -> tuple[Dataset, ...]:
        return super()._check_at_least_two_datasets(datasets)

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_datasets(cls, datasets: Sequence[Dataset]) -> Sequence[Dataset]:
        """
        Checks datasets consistency.
        """
        for i, dataset in enumerate(datasets):
            df = dataset.df[[PARTICIPANT_ID, SESSION_ID]]

            if df.duplicated().any():
                raise ValueError(
                    "Datasets passed to PairedDataset cannot contain duplicated (participant, session) pairs, "
                    f"but some were founds in dataset {i}:\n {df[df.duplicated(keep=False)]}"
                )

            if i == 0:
                ref_particpants_sessions = set(df.itertuples(index=False, name=None))
                ref_len = len(dataset)
            else:
                particpants_sessions = set(df.itertuples(index=False, name=None))
                difference = particpants_sessions.symmetric_difference(
                    ref_particpants_sessions
                )
                if difference:
                    raise ValueError(
                        "To pair datasets, they must have exactly the same (participant, session) pairs. "
                        f"Differences were found for between dataset 0 and dataset {i}:\n"
                        f"{difference}"
                    )
                assert (
                    (len_ := len(dataset)) == ref_len
                ), f"PairedDataset only accepts datasets of the same length. Dataset 0 is {ref_len} samples long, whereas dataset {i} is {len_}."

        return datasets

    @classmethod
    def _get_class(cls) -> type[PairedDataset]:
        """Returns the class associated to this config class."""
        return PairedDataset


class PairedDataset(CollectionDataset):
    """
    For pairing multiple :py:class:`~clinicadl.data.datasets.Dataset` (e.g., different modalities).

    Pairing datasets means uniquely associating images across the datasets. The keys of this association
    are the (participant, session) pairs present in the underlying datasets. So, **all datasets must
    contain the same (participant, session) pairs**.

    Furthermore, for a (participant, session) pair, **all the datasets must have the same number of samples**:
    if one of your dataset contains whole images and a second one contains a single slice of the images, it's ok;
    but if the second dataset now contains two slices of the images, this will raise an error because the second dataset
    will thus be two times bigger than the first one, and the two datasets cannot be paired.

    A ``PairedDataset`` returns a tuple of :py:class:`~clinicadl.data.structures.Sample` (one for each underlying
    dataset).

    Parameters
    ----------
    datasets : Iterable[Dataset]
        The ``Datasets`` to pair.

    Raises
    ------
    ValueError
        If the datasets contain duplicated (participant, session) pairs. This is an
        issue because it will prevent ``PairedDataset`` from finding a bijective mapping between
        the datasets.
    ValueError
        If there is a mismatch of (participant, session) pairs across the datasets. An error will
        also be raised if the number of samples per image is not the same across datasets.

    Examples
    --------
    .. code-block:: text

        bids
        ├── sub-001
        │   └── ses-M000
        │   │   ├── pet
        │   │   │   └── sub-001_ses-M000_trc-18FAV45_pet.nii.gz
        │   │   └── anat
        │   │       └── sub-001_ses-M000_T1w.nii.gz
            ...
        ...

    .. code-block:: python

        from clinicadl.data.datasets import BidsDataset, PairedDataset
        from clinicadl.io.bids import BidsFileType

        bids_t1 = BidsDataset("bids", file_type=BidsFileType(data_type="anat", suffix="T1w"))
        bids_pet = BidsDataset("bids", file_type=BidsFileType(data_type="pet", suffix="pet"))

        multimodal_dataset = PairedDataset([bids_t1, bids_pet])

    .. code-block:: python

        >>> len(bids_t1)
        4
        >>> len(bids_pet)
        4
        >>> len(multimodal_dataset)
        4
        >>> sample = multimodal_dataset[0]
        >>> len(sample)
        2
        >>> sample[0].file_type
        BidsFileType(suffix=re.compile('T1w'), data_type=re.compile('anat'), extension=re.compile('.nii.*'), with_entities=None, without_entities=None, description=None)
        >>> sample[1].file_type
        BidsFileType(suffix=re.compile('pet'), data_type=re.compile('pet'), extension=re.compile('.nii.*'), with_entities=None, without_entities=None, description=None)
    """

    _config_type = PairedDatasetConfig

    def __init__(
        self,
        datasets: Iterable[Dataset],
    ):
        super().__init__(datasets=datasets)
        self._check_datasets_conistency()

    @property
    def df(self):
        "The output of the merger of the metadata DataFrames of the underlying datasets."
        return super().df

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int) -> tuple[Sample, ...]:
        """
        Retrieves the collection of samples at a given index.

        Parameters
        ----------
        idx : int
            Index of the samples in the dataset.

        Returns
        -------
        tuple[Sample, ...]
            A structured output containing the processed data and metadata
            from each dataset of the ``PairedDataset``, in a ``tuple`` of
            :py:class:`~clinicadl.data.structures.Sample`.
        """
        return tuple(dataset[idx] for dataset in self.datasets)

    def get_sample_info(self, idx: int, column: str) -> Any:
        """
        Retrieves information on a given sample.

        It will look for ``column`` in the DataFrame of each underlying
        dataset. If several values are found, it will raise an error.

        Parameters
        ----------
        idx : int
            The index of the sample in the dataset.
        column : str
            The information to look for, i.e. a column of :py:attr:`df`.

        Returns
        -------
        Any
            The value for this sample.

        Raises
        ------
        RuntimeError
            If different values are found across the datasets.
        """
        values = set()
        for dataset in self.datasets:
            try:
                values.add(dataset.get_sample_info(idx, column))
            except KeyError:
                continue

        if len(values) == 0:
            raise KeyError(
                f"No column named '{column}' in any dataset of the PairedDataset."
            )
        elif len(values) > 1:
            raise RuntimeError(
                f"Multiple values found for '{column}' for sample {idx} in the datasets. Got {values.pop()} and {values.pop()}"
            )

        return values.pop()

    def _check_datasets_conistency(
        self,
    ) -> None:
        """
        Checks that the (participant, session) pairs match between datasets.
        """
        for idx in range(len(self)):
            participants_sessions = set(
                (
                    dataset.get_sample_info(idx, PARTICIPANT_ID),
                    dataset.get_sample_info(idx, SESSION_ID),
                )
                for dataset in self.datasets
            )
            if len(participants_sessions) > 1:
                raise RuntimeError(
                    f"Sample {idx} is associated to {participants_sessions.pop()} in one dataset, but {participants_sessions.pop()} in another. "
                    "Make sure that the (participant, session) are consistent across your datasets."
                )

    @staticmethod
    def _merge_dfs(datasets: Sequence[Dataset]) -> pd.DataFrame:
        key_cols = [PARTICIPANT_ID, SESSION_ID]
        all_cols = set().union(*(df.df.columns for df in datasets))
        extra_cols = list(all_cols - set(key_cols))

        if not extra_cols:
            return (
                pd.concat([dataset.df[key_cols] for dataset in datasets])
                .drop_duplicates()
                .reset_index(drop=True)
            )

        concat_df = pd.concat(
            [dataset.df.set_index(key_cols) for dataset in datasets],
            keys=range(len(datasets)),
            axis=1,
        )

        def _resolve(group: pd.Series) -> Any:
            values = group.dropna().unique()
            if len(values) == 0:
                return np.nan
            elif len(values) == 1:
                return values[0]
            participant, session, column = group.name
            raise RuntimeError(
                f"For ({participant}, {session}), "
                f"different values found for '{column}' across the datasets forming the PairedDataset: {values}"
            )

        merged = (
            concat_df.stack(dropna=False)
            .apply(_resolve, axis=1)
            .unstack(2)
            .reset_index()
        )

        # correct column types
        for column in merged:
            for dataset in datasets:
                try:
                    t = dataset.df[column].dtype
                except KeyError:
                    continue
                merged[column] = merged[column].astype(t)

        return merged
