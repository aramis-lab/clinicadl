from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd
from pydantic import field_validator
from torch.nn import UninitializedParameter

from clinicadl.utils.dictionary.words import DATASET_ID

from ..structures import Sample
from .base import Dataset
from .collection import CollectionDataset, CollectionDatasetConfig


class UnpairedDatasetConfig(CollectionDatasetConfig):
    """
    Config class for ``UnpairedDataset``.
    """

    oversample: bool

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_dataset_id_column(
        cls, datasets: tuple[Dataset, ...]
    ) -> tuple[Dataset, ...]:
        return super()._check_dataset_id_column(datasets)

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_at_least_two_datasets(
        cls, datasets: tuple[Dataset, ...]
    ) -> tuple[Dataset, ...]:
        return super()._check_at_least_two_datasets(datasets)

    @classmethod
    def _get_class(cls) -> type[UnpairedDataset]:
        """Returns the class associated to this config class."""
        return UnpairedDataset


class UnpairedDataset(CollectionDataset):
    """
    For stacking multiple :py:class:`~clinicadl.data.datasets.Dataset`
    (e.g. different modalities from different datasets). By "stacking", we mean **randomly** associating images across datasets.

    So, ``UnpairedDataset`` differs from :py:class:`~clinicadl.data.datasets.PairedDataset` in that ``PairedDataset``
    associates images across datasets via a unique mapping. Therefore, as opposed to ``PairedDataset``, there is no need for
    the datasets forming the ``UnpairedDataset`` to contain the same (participant, session) pairs.

    The randomness of the mapping between datasets can be controlled via :py:meth:`~UnpairedDataset.set_epoch`.
    This enables to have different associations for each epoch.

    The size of an ``UnpairedDataset`` is set to **the size of its biggest underlying dataset** if ``oversample=True``,
    or to **the size of its smallest underlying dataset** if ``oversample=False``: to handle datasets
    with different sizes, ``UnpairedDataset`` will randomly replicate some of their samples so that they reach the
    size of the biggest dataset if ``oversample=True``, or will randomly drop some of their samples so that they reach the
    size of the smallest dataset if ``oversample=False``. This randomness is also controlled via
    :py:meth:`~UnpairedDataset.set_epoch`.

    An ``UnpairedDataset`` will return a tuple of :py:class:`~clinicadl.data.structures.Sample` (one for each underlying
    dataset).

    Parameters
    ----------
    datasets : Iterable[Dataset]
        The ``Datasets`` to stack.

    oversample: bool, default=False
        Strategy to adopt when the datasets have different sizes:

        - ``oversample=True``: randomly replicate samples in smaller datasets so that they reach the
          size of the biggest dataset.
        - ``oversample=False``: randomly drop samples in bigger datasets so that all datasets reach the
          size of the smallest dataset.

    Examples
    --------
    .. code-block:: text

        bids_t1
        ├── sub-001
        │   └── ses-M000
        │   │   └── anat
        │   │       └── sub-001_ses-M000_T1w.nii.gz
            ...
        ...

        bids_pet
        ├── sub-A
        │   └── ses-M003
        │   │   └── pet
        │   │       └── sub-A_ses-M000_trc-18FAV45_pet.nii.gz
            ...
        ...

    .. code-block:: python

        from clinicadl.data.datasets import BidsDataset, UnpairedDataset
        from clinicadl.io.bids import BidsFileType

        bids_t1 = BidsDataset("bids_t1", file_type=BidsFileType(data_type="anat", suffix="T1w"))
        bids_pet = BidsDataset("bids_pet", file_type=BidsFileType(data_type="pet", suffix="pet"))

        multimodal_dataset = UnpairedDataset([bids_t1, bids_pet], oversample=True)

    .. code-block:: python

        >>> len(bids_t1)
        4
        >>> len(bids_pet)
        2
        >>> len(stacked)
        4   # length of the biggest dataset

    We can access the random mapping made between the datasets via ``.mapping``:

    .. code-block:: python

        >>> stacked.mapping
        dataset_id	0	1
               idx
                0	2	0
                1	3	0
                2	1	0
                3	0	1

    ``idx`` is the index of the sample in the ``UnpairedDataset``. In column ``0``, you have the
    associated sample in the first dataset (``bids_t1``), and in column ``1``, the associated
    sample in the second dataset (``bids_pet``).

    .. code-block:: python

        >>> bids_t1[2].participant, bids_t1[2].session,
        ('sub-002', 'ses-M000')

        >>> bids_pet[0].participant, bids_pet[0].session
        ('sub-A', 'ses-M000')

        >>> sample = stacked[0]
        >>> len(sample)
        2
        >>> sample[0].participant, sample[0].session
        ('sub-002', 'ses-M000')
        >>> sample[1].participant, sample[1].session
        ('sub-A', 'ses-M000')

    Now we can change the random mapping with :py:meth:`~UnpairedDataset.set_epoch`:

    .. code-block:: python

        >>> stacked.set_epoch(7)
        >>> stacked.mapping
        dataset_id	0	1
               idx
                0	2	1
                1	1	1
                2	0	0
                3	3	0

        >>> sample = stacked[0]
        >>> sample[1].participant, sample[1].session
        ('sub-B', 'ses-M000')

    Finally, if ``oversample=False``:

    .. code-block:: python

        >>> stacked = UnpairedDataset([bids_t1, bids_pet], oversample=False)
        >>> len(stacked)
        2   # = length of the smallest dataset
        >>> stacked.mapping
        dataset_id	0	1
               idx
                0	2	0
                1	3	1
    """

    _config_type = UnpairedDatasetConfig
    config: type[UnpairedDatasetConfig]

    def __init__(
        self,
        datasets: Iterable[Dataset],
        oversample: bool = False,
    ):
        super().__init__(datasets=datasets, oversample=oversample)
        self.epoch = 0
        self._mapping = self._map_datasets()

    @property
    def df(self):
        "The output of the merger of the metadata DataFrames of the underlying datasets."
        return super().df

    @property
    def mapping(self) -> pd.DataFrame:
        """The random mapping between the samples of the underlying datasets."""
        return self._mapping

    def get_sample_info(self, idx: int, column: str) -> tuple[Any, ...]:
        """
        Retrieves information on a given sample.

        In an ``UnpairedDataset``, a sample is a tuple of "sub-samples" from the underlying datasets. Therefore,
        ``get_sample_info`` will also return a tuple, containing the information on all the sub-samples
        forming the sample.

        If the information cannot be found for a sub-sample (because all the underlying datasets don't
        necessarily contain the same information), ``get_sample_info`` will return ``None`` for this sub-sample.

        See :py:meth:`Dataset.get_sample_info <clinicadl.data.datasets.Dataset.get_sample_info>`
        for more details.

        Parameters
        ----------
        idx : int
            The index of the sample in the ``UnpairedDataset``.
        column : str
            The information to look for, i.e. a column present in the metadata DataFrame of at least one of the
            dataset forming the ``UnpairedDataset``.

        Returns
        -------
        tuple[Any, ...]
            The information (e.g. the age, the sex, etc.) found for each sub-sample.

        Raises
        ------
        KeyError
            If ``column`` is not in any DataFrame of the datasets forming the ``UnpairedDataset``.
        """
        self._check_idx(idx)

        indices = self._mapping.iloc[idx]
        list_info = []
        for dataset, idx_in_dataset in zip(self.datasets, indices):
            try:
                info = dataset.get_sample_info(idx_in_dataset, column)
            except KeyError:
                info = None
            list_info.append(info)

        if all(v is None for v in list_info):
            raise KeyError(
                f"No column named '{column}' in any DataFrame of the datasets forming the UnpairedDataset."
            )

        return tuple(list_info)

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch.

        This ensures that the random mapping between the datasets is
        different for each epoch.

        Parameters
        ----------
        epoch : int
            Epoch number.
        """
        self.epoch = epoch
        self._mapping = self._map_datasets()

    def __len__(self) -> int:
        """
        The length of an ``UnpairedDataset`` is the length of its biggest dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self._mapping)

    def __getitem__(self, idx: int) -> tuple[Sample, ...]:
        """
        Retrieves the collection of samples at a given index.

        The random mapping between datasets (in ``self.mapping``) is
        used to determine which samples to retrieve for each underlying
        dataset.

        Parameters
        ----------
        idx : int
            Index of the samples in the dataset.

        Returns
        -------
        tuple[Sample, ...]
            A structured output containing the processed data and metadata
            from each dataset of the ``UnpairedDataset``, as a ``tuple`` of
            :py:class:`~clinicadl.data.structures.Sample`.
        """
        self._check_idx(idx)
        indices = self._mapping.iloc[idx]
        return tuple(
            dataset[idx_in_dataset]
            for dataset, idx_in_dataset in zip(self.datasets, indices)
        )

    @staticmethod
    def _merge_dfs(datasets: Sequence[Dataset]) -> pd.DataFrame:
        df: pd.DataFrame = pd.concat(
            [dataset.df for dataset in datasets],
            keys=range(len(datasets)),
            names=[DATASET_ID],
        )

        return df.reset_index(
            drop=False,
            level=DATASET_ID,
        ).reset_index(drop=True)

    def _map_datasets(self) -> pd.DataFrame:
        """
        Randomly associates the samples of the datasets forming
        the ``UnpairedDataset``.

        As the datasets don't necessarily have the same length, some
        data of the small datasets are replicated so that they match
        the length of the biggest one.

        The randomness of the mapping is entirely controlled by ``self.epoch``.
        """
        max_len = max(len(dataset) for dataset in self.datasets)
        shuffled_indices = []
        for i, dataset in enumerate(self.datasets):
            indices = pd.Series(range(len(dataset)))

            if self.config.oversample:
                indices = indices.reindex(range(max_len))  # nans appear

            indices = indices.sample(
                frac=1.0,
                random_state=self.epoch
                + i * 1000,  # different shuffling for every dataset
                ignore_index=True,
            )
            indices = (
                indices.ffill().bfill()  # fill nans, i.e. duplicate some data to reach len(self) when oversample=True
            )

            shuffled_indices.append(indices)

        mapping: pd.DataFrame = pd.concat(
            shuffled_indices,
            axis=1,
            keys=range(len(self.datasets)),
            names=[DATASET_ID],
        )  # nans appear only if oversample=False
        mapping = mapping.dropna()

        return mapping.astype(int).rename_axis(index="idx")
