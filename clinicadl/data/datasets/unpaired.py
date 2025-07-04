# coding: utf8
from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Iterable, Tuple

import pandas as pd
from torch.utils.data import Dataset

from clinicadl.dictionary.words import (
    DATASET_ID,
    FIRST_INDEX,
    LAST_INDEX,
    PARTICIPANT_ID,
    SESSION_ID,
)
from clinicadl.transforms.extraction import Sample
from clinicadl.tsvtools.utils import read_data
from clinicadl.utils.exceptions import ClinicaDLCAPSError
from clinicadl.utils.typing import DataType

from .caps_dataset import CapsDataset

logger = getLogger("clinicadl.data.datasets.unpaired")


class UnpairedDataset(Dataset):
    """
    ``UnpairedDataset`` is a useful class to stack multiple :py:class:`~clinicadl.data.datasets.CapsDataset`
    (e.g. different modalities from different datasets). By "stacking", we mean **randomly** associating images across datasets.

    So, ``UnpairedDataset`` differs from :py:class:`~clinicadl.data.datasets.PairedDataset` in that ``PairedDataset``
    associates images across datasets via a unique mapping. Therefore, as opposed to ``PairedDataset``, there is no need for
    the ``CapsDatasets`` forming the ``UnpairedDataset`` to contain the same (participant, session) pairs.

    The randomness of the mapping between datasets can be controlled via :py:meth:`~UnpairedDataset.set_epoch`.
    This enables to have different associations for each epoch.

    The size of an ``UnpairedDataset`` is set to **the size of its biggest underlying CapsDataset** if ``oversample=True``,
    or to **the size of its smallest underlying CapsDataset** if ``oversample=False``: to handle datasets
    with different sizes, ``UnpairedDataset`` will randomly replicate some of their samples so that they reach the
    size of the biggest dataset if ``oversample=True``, or will randomly drop some of their samples so that they reach the
    size of the smallest dataset if ``oversample=False``. This randomness is also controlled via
    :py:meth:`~UnpairedDataset.set_epoch`.

    An ``UnpairedDataset`` will return a tuple of :py:class:`~clinicadl.data.structures.DataPoint` (one for each underlying
    dataset).

    To use ``UnpairedDataset``, you must **previously perform**
    :py:func:`tensor conversion <clinicadl.data.datasets.CapsDataset.to_tensors>`.

    .. note::
        ``UnpairedDataset`` also accepts :py:class:`~clinicadl.data.datasets.ConcatDataset`.

    Parameters
    ----------
    datasets : Iterable[CapsDataset]
        List of :py:class:`~clinicadl.data.datasets.CapsDataset` to be stacked.
    oversample: bool, default=False
        Strategy to adopt when the datasets have different sizes:

        - ``oversample=True``: randomly replicate samples in smaller datasets so that they reach the
          size of the biggest dataset.
        - ``oversample=False``: randomly drop samples in bigger datasets so that all datasets reach the
          size of the smallest dataset.


    Raises
    ------
    ClinicaDLCAPSError
        If tensor conversion has not been performed for all the datasets before stacking.

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

        caps_t1 = CapsDataset("caps_t1", preprocessing=T1Linear(use_uncropped_image=True))
        caps_pet = CapsDataset(
            "caps_pet",
            preprocessing=PETLinear(
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

    def __init__(self, datasets: Iterable[CapsDataset], oversample: bool = False):
        super().__init__()
        assert len(datasets) >= 2, "UnpairedDataset needs at least 2 datasets to stack!"
        self._check_conversion(datasets)
        self._df = self._concat_dfs(list(datasets))
        self.datasets = datasets
        self.epoch = 0
        self.oversample = oversample
        self._mapping = self._map_datasets()

    @property
    def df(self) -> pd.DataFrame:
        """The result of the merger of the DataFrames of the underlying ``CapsDatasets``."""
        return self._df

    @property
    def mapping(self) -> pd.DataFrame:
        """The random mapping between the samples of the underlying ``CapsDatasets``."""
        return self._mapping

    def eval(self) -> None:
        """
        Sets the datasets to evaluation mode.

        It disables data augmentation in the transformation pipeline.
        """
        for dataset in self.datasets:
            dataset.eval_mode = True

    def train(self) -> None:
        """
        Sets the datasets to training mode.

        It enables data augmentation in the transformation pipeline.
        """
        for dataset in self.datasets:
            dataset.eval_mode = False

    def subset(self, data: DataType) -> UnpairedDataset:
        """
        To get a subset of the ``UnpairedDataset`` from a list of (participant, session) pairs.

        In practice, it will call :py:meth:`CapsDataset.subset <clinicadl.data.datasets.CapsDataset.subset>`
        for all the datasets forming the ``UnpairedDataset``.

        Parameters
        ----------
        data : DataType
            A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of participant-session
            pairs to extract. Please note that this list must be passed via two columns named ``"participant_id"``
            and ``"session_id"`` (other columns won't be considered).

        Returns
        -------
        UnpairedDataset
            A subset of the original ``UnpairedDataset``, restricted to the (participant, session) pairs mentioned in ``data``.

        Raises
        ------
        ClinicaDLTSVError
            If the DataFrame associated to ``data`` does not contain the columns ``"participant_id"``
            and ``"session_id"``.
        ClinicaDLCAPSError
            If the subset of one of the datasets forming the ``UnpairedDataset`` is empty.
        """
        return UnpairedDataset([dataset.subset(data) for dataset in self.datasets])

    def describe(self) -> tuple[Dict[str, Any], ...]:
        """
        Returns a description of the ``CapsDatasets`` forming the ``UnpairedDataset``.

        Returns
        -------
        tuple[Dict[str, Any], ...]
            The descriptions returned by :py:meth:`CapsDataset.describe
            <clinicadl.data.datasets.CapsDataset.describe>` for each
            dataset forming the ``UnpairedDataset``.

        Raises
        ------
        ClinicaDLCAPSError
            See :py:meth:`CapsDataset.describe <clinicadl.data.datasets.CapsDataset.describe>`.
        """
        return tuple([dataset.describe() for dataset in self.datasets])

    def get_sample_info(self, idx: int, column: str) -> tuple[Any, ...]:
        """
        Retrieves information on a given sample.

        In an ``UnpairedDataset``, a sample is a tuple of "sub-samples" from the underlying datasets. Therefore,
        ``get_sample_info`` will also return a tuple, containing the information on all the sub-samples
        forming the sample.

        If the information cannot be found for a sub-sample (because all the underlying datasets don't
        necessarily contain the same information), ``get_sample_info`` will return ``None`` for this sub-sample.

        See :py:meth:`CapsDataset.get_sample_info <clinicadl.data.datasets.CapsDataset.get_sample_info>`
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
        IndexError
            If ``idx`` is not a non-negative integer, greater or equal to
            the length of the dataset.
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
                f"No column named {column} in any DataFrame of the datasets forming the UnairedDataset."
            )

        return tuple(list_info)

    def get_participant_session_couples(self) -> list[Tuple[str, str]]:
        """
        Retrieves all participant-session pairs in the dataset.

        Returns
        -------
        List[Tuple[str, str]]
            The list of (participant, session).
        """
        stacked = self._df.stack(DATASET_ID)
        return list(set(zip(stacked[PARTICIPANT_ID], stacked[SESSION_ID])))

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
        """
        return len(self._mapping)

    def __getitem__(self, idx: int) -> tuple[Sample, ...]:
        """
        Retrieves the samples at a given index.

        The random mapping between datasets (in ``self.mapping``) is
        used to determine which sample to retrieve for each underlying
        dataset.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.

        Returns
        -------
        tuple[Sample, ...]
            A structured output containing the processed data and metadata
            for each dataset of the UnpairedDataset, as
            :py:class:`~clinicadl.transforms.extraction.Sample`.

        Raises
        ------
        IndexError
            If ``idx`` is not a non-negative integer, greater or equal to
            the length of the UnpairedDataset (i.e. the length of the biggest
            dataset).
        """
        self._check_idx(idx)
        indices = self._mapping.iloc[idx]
        return tuple(
            dataset[idx_in_dataset]
            for dataset, idx_in_dataset in zip(self.datasets, indices)
        )

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

    @staticmethod
    def _check_conversion(datasets: Iterable[CapsDataset]) -> None:
        """
        Checks that tensor conversion has been performed before stacking.
        """
        for dataset in datasets:
            if not dataset.converted:
                raise ClinicaDLCAPSError(
                    "Tensor conversion must be performed BEFORE stacking. Please call "
                    "'to_tensors' or 'read_tensor_conversion' for each dataset."
                )

    @staticmethod
    def _concat_dfs(datasets: list[CapsDataset]) -> pd.DataFrame:
        """
        Concatenate (along column axis) the dataframes.
        """
        df = pd.concat(
            [
                dataset.df.drop(columns=[FIRST_INDEX, LAST_INDEX])
                for dataset in datasets
            ],
            axis=1,
            keys=range(len(datasets)),
            names=[DATASET_ID],
        )

        return df

    def _map_datasets(self) -> pd.DataFrame:
        """
        Randomly associates the samples of the datasets forming
        the UnpairedDataset.

        As the datasets don't necessarily have the same length, some
        data of the small datasets are replicated so that they match
        the length of the biggest one.

        The randomness of the mapping is entirely controlled by ``self.epoch``.
        """
        max_len = max(len(dataset) for dataset in self.datasets)
        shuffled_indices = []
        for i, dataset in enumerate(self.datasets):
            indices = pd.Series(range(len(dataset)))

            if self.oversample:
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
