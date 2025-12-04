from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd
from pydantic import field_validator

from clinicadl.utils.dictionary.words import (
    DATASET_ID,
    PARTICIPANT_ID,
    SESSION_ID,
)

from .collection import CollectionDataset, CollectionDatasetConfig
from .multi_samples import MultiSamplesDataset
from .output import Sample


class UnpairedDatasetConfig(CollectionDatasetConfig):
    """
    Config class for ``UnpairedDataset``.
    """

    oversample: bool

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_n_datasets(
        cls, datasets: Sequence[MultiSamplesDataset]
    ) -> Sequence[MultiSamplesDataset]:
        assert (
            len(datasets) >= 2
        ), f"{cls._get_name()} needs at least 2 datasets to join!"

        return datasets

    @classmethod
    def _get_class(cls) -> type[UnpairedDataset]:
        """Returns the class associated to this config class."""
        return UnpairedDataset


class UnpairedDataset(CollectionDataset):
    """
    ``UnpairedDataset`` is a useful class to stack multiple :py:class:`~clinicadl.data.datasets.MultiSamplesDataset`
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

    An ``UnpairedDataset`` will return a tuple of :py:class:`~clinicadl.data.structures.DataPoint` (one for each underlying
    dataset).

    .. note::
        ``UnpairedDataset`` also accepts :py:class:`~clinicadl.data.datasets.ConcatDataset`.

    Parameters
    ----------
    datasets : Iterable[MultiSamplesDataset]
        List of :py:class:`~clinicadl.data.datasets.MultiSamplesDataset` to be stacked.
    oversample: bool, default=False
        Strategy to adopt when the datasets have different sizes:

        - ``oversample=True``: randomly replicate samples in smaller datasets so that they reach the
          size of the biggest dataset.
        - ``oversample=False``: randomly drop samples in bigger datasets so that all datasets reach the
          size of the smallest dataset.

    Examples
    --------
    .. code-block:: text

        Data look like:

        caps_t1
        ├── tensor_conversion
        │   └── default_t1-linear.json
        └── subjects
            ├── sub-001
            │   └── ses-M000
            │       └── t1_linear
            │           ├── sub-001_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz
            │           └── tensors
            │               └── default
            │                   └── sub-001_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt
                ...
            ...

        caps_pet
        ├── tensor_conversion
        │   └── default_pet-linear_18FAV45_pons2.json
        └── subjects
            ├── sub-A
            │   └── ses-M000
            │       ├── pet_linear
            │       │   ├── sub-A_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
            │       │   └── tensors
            │       │       └── default
            │       │           └── sub-A_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt
                ...
            ...

    .. code-block:: python

        from clinicadl.data.datasets import CapsDataset, UnpairedDataset
        from clinicadl.data.datatypes import PETLinear, T1Linear

        caps_t1 = CapsDataset("caps_t1", datatype=T1Linear(use_uncropped_image=True))
        caps_pet = CapsDataset(
            "caps_pet",
            datatype=PETLinear(
                use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
            ),
        )

        caps_pet.read_tensor_conversion()
        caps_t1.read_tensor_conversion()

        stacked = UnpairedDataset([caps_t1, caps_pet], oversample=True)

    .. code-block:: python

        >>> len(caps_t1)
        4
        >>> len(caps_pet)
        2
        >>> len(stacked)
        4   # = length of the biggest dataset

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
    associated sample in the first dataset (``caps_t1``), and in column ``1``, the associated
    sample in the second dataset (``caps_pet``).

    .. code-block:: python

        >>> caps_t1[2].participant, caps_t1[2].session,
        ('sub-002', 'ses-M000')

        >>> caps_pet[0].participant, caps_pet[0].session
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

        >>> stacked = UnpairedDataset([caps_t1, caps_pet], oversample=False)
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
        datasets: Iterable[MultiSamplesDataset],
        oversample: bool = False,
    ):
        super().__init__(datasets=datasets, oversample=oversample)
        self.epoch = 0
        self._mapping = self._map_datasets()

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

        See :py:meth:`ClinicaDLDataset.get_sample_info <clinicadl.data.datasets.ClinicaDLDataset.get_sample_info>`
        for more details.

        Parameters
        ----------
        idx : int
            The index of the sample in the ``UnpairedDataset``.
        column : str
            The information to look for, i.e. a column present in the DataFrame of at least one of the
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
                f"No column named {column} in any DataFrame of the datasets forming the UnpairedDataset."
            )

        return tuple(list_info)

    def get_participant_session_couples(self) -> set[tuple[str, str]]:
        return set(zip(self._df[PARTICIPANT_ID], self._df[SESSION_ID]))

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
            :py:class:`~clinicadl.transforms.extraction.Sample`.
        """
        self._check_idx(idx)
        indices = self._mapping.iloc[idx]
        return tuple(
            dataset[idx_in_dataset]
            for dataset, idx_in_dataset in zip(self.datasets, indices)
        )

    @staticmethod
    def _merge_dfs(datasets: Sequence[MultiSamplesDataset]) -> pd.DataFrame:
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
