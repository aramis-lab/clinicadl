# coding: utf8
from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import StackDataset

from clinicadl.dictionary.words import (
    FIRST_INDEX,
    LAST_INDEX,
    PARTICIPANT_ID,
    SESSION_ID,
)
from clinicadl.transforms.extraction import Sample
from clinicadl.utils.exceptions import ClinicaDLCAPSError
from clinicadl.utils.typing import DataType

from .caps_dataset import CapsDataset

logger = getLogger("clinicadl.data.datasets.paired")


class PairedDataset(StackDataset):
    """
    ``PairedDataset`` is a useful class to pair multiple :py:class:`~clinicadl.data.datasets.CapsDataset`
    (e.g. different modalities). Pairing datasets means uniquely associating images across the datasets.

    The keys of this association are the (participant, session) pairs present in the underlying ``CapsDatasets``. So, **all
    datasets must contain the same (participant, session) pairs**.

    Furthermore, for a (participant, session) pair, **all the datasets must have the same number of samples**:
    if one of your dataset contains whole images and a second one contains a single slice of the images, it's ok;
    but if the second dataset now contains two slices of the images, this will raise an error because the second dataset
    will thus be two times bigger than the first one, and the two datasets cannot be paired.

    A ``PairedDataset`` will return a tuple of :py:class:`~clinicadl.data.structures.DataPoint` (one for each underlying
    dataset).

    To pair ``CapsDatasets``, you must **previously perform**
    :py:func:`tensor conversion <clinicadl.data.datasets.CapsDataset.to_tensors>`.

    .. note::
        ``PairedDataset`` also accepts :py:class:`~clinicadl.data.datasets.ConcatDataset`.

    Parameters
    ----------
    datasets : Iterable[CapsDataset]
        List of :py:class:`~clinicadl.data.datasets.CapsDataset` to be paired.

    Raises
    ------
    ClinicaDLCAPSError
        If tensor conversion has not been performed for all the datasets before pairing.
    ClinicaDLCAPSError
        If the datasets contain duplicated (participant, session) pairs. This is an
        issue because it will prevent ``PairedDataset`` from finding a bijective mapping between
        the datasets.
    ClinicaDLCAPSError
        If there is a mismatch of (participant, session) pairs across the datasets. An error will
        also be raised if the number of samples per image is not the same across datasets.

    Examples
    --------
    .. code-block:: text

        Data look like:

        mycaps
        ├── tensor_conversion
        │   ├── default_pet-linear_18FAV45_pons2.json
        │   └── default_t1-linear.json
        └── subjects
            ├── sub-001
            │   └── ses-M000
            │       ├── pet_linear
            │       │   ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
            │       │   └── tensors
            │       │       └── default
            │       │           └── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt
            │       └── t1_linear
            │           ├── sub-001_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz
            │           └── tensors
            │               └── default
            │                   └── sub-001_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt
                ...
            ...

    .. code-block:: python

        from clinicadl.data.datasets import CapsDataset, PairedDataset
        from clinicadl.data.datatypes import PETLinear, T1Linear

        caps_t1 = CapsDataset(
            "mycaps", preprocessing=T1Linear(use_uncropped_image=True), data=participants_sessions
        )
        caps_pet = CapsDataset(
            "mycaps",
            preprocessing=PETLinear(
                use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
            ),
        )

        caps_t1.read_tensor_conversion()
        caps_pet.read_tensor_conversion()

        paired_dataset = PairedDataset([caps_t1, caps_pet])

    .. code-block:: python

        >>> len(caps_t1)
        4
        >>> len(caps_pet)
        4
        >>> len(paired_dataset)
        4
        >>> sample = paired_dataset[0]
        >>> len(sample)
        2
        >>> sample[0].participant, sample[0].session
        ('sub-001', 'ses-M000')
        >>> sample[1].participant, sample[1].session
        ('sub-001', 'ses-M000')
    """

    def __init__(
        self,
        datasets: Iterable[CapsDataset],
    ):
        assert len(datasets) >= 2, "PairedDataset needs at least 2 datasets to pair!"
        self._check_conversion(datasets)
        self._df = self._merge_dfs(list(datasets))
        super().__init__(*datasets)
        self.datasets: tuple[CapsDataset, ...]

    @property
    def df(self) -> pd.DataFrame:
        """The result of the merger of the DataFrames of the underlying ``CapsDatasets``."""
        return self._df

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

    def subset(self, data: DataType) -> PairedDataset:
        """
        To get a subset of the ``PairedDataset`` from a list of (participant, session) pairs.

        In practice, it will call :py:meth:`CapsDataset.subset <clinicadl.data.datasets.CapsDataset.subset>`
        for all the datasets forming the ``PairedDataset``.

        Parameters
        ----------
        data : DataType
            A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of (participant, session)
            pairs to extract. Please note that this list must be passed via two columns named ``"participant_id"``
            and ``"session_id"`` (other columns won't be considered).

        Returns
        -------
        PairedDataset
            A subset of the original ``PairedDataset``, restricted to the (participant, session) pairs mentioned in ``data``.

        Raises
        ------
        ClinicaDLTSVError
            If the DataFrame associated to ``data`` does not contain the columns ``"participant_id"``
            and ``"session_id"``.
        ClinicaDLCAPSError
            If no (participant, session) pairs mentioned in ``data`` are in the current ``PairedDataset``
            (this would lead to an empty dataset).
        """
        return PairedDataset([dataset.subset(data) for dataset in self.datasets])

    def describe(self) -> tuple[Dict[str, Any], ...]:
        """
        Returns a description of the ``CapsDatasets`` forming the ``PairedDataset``.

        Returns
        -------
        tuple[Dict[str, Any], ...]
            The descriptions returned by :py:meth:`CapsDataset.describe
            <clinicadl.data.datasets.CapsDataset.describe>` for each
            dataset forming the ``PairedDataset``.

        Raises
        ------
        ClinicaDLCAPSError
            See :py:meth:`CapsDataset.describe <clinicadl.data.datasets.CapsDataset.describe>`.
        """
        return tuple([dataset.describe() for dataset in self.datasets])

    def get_sample_info(self, idx: int, column: str) -> Any:
        """
        Retrieves information on a given sample.

        See :py:meth:`CapsDataset.get_sample_info <clinicadl.data.datasets.CapsDataset.get_sample_info>`
        for more details.

        Parameters
        ----------
        idx : int
            The index of the sample in the ``PairedDataset``.
        column : str
            The information to look for, i.e. a column present in the DataFrame of at least one of the
            dataset forming the ``PairedDataset``.

        Returns
        -------
        Any
            The information (e.g. the age, the sex, etc.)

        Raises
        ------
        IndexError
            If ``idx`` is not a non-negative integer, greater or equal to
            the length of the dataset.
        KeyError
            If ``column`` is not in any DataFrame of the datasets forming the ``PairedDataset``.
        """
        self._check_idx(idx)

        if column not in self._df.columns:
            raise KeyError(
                f"No column named '{column}' in any dataset of the PairedDataset. Present columns are: "
                f"{list(self._df.columns)}"
            )

        row = self._df[(self._df[FIRST_INDEX] <= idx) & (idx <= self._df[LAST_INDEX])]

        return row[column].iloc[0]

    def get_participant_session_couples(self) -> list[Tuple[str, str]]:
        """
        Retrieves all (participant, session) pairs in the dataset.

        Returns
        -------
        List[Tuple[str, str]]
            The list of (participant, session).
        """
        return list(zip(self._df[PARTICIPANT_ID], self._df[SESSION_ID]))

    def __getitem__(self, idx: int) -> tuple[Sample, ...]:
        """
        Retrieves the samples at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.

        Returns
        -------
        tuple[Sample, ...]
            A structured output containing the processed data and metadata
            for each dataset of the PairedDataset, as
            :py:class:`~clinicadl.transforms.extraction.Sample`.

        Raises
        ------
        IndexError
            If ``idx`` is not a non-negative integer, greater or equal to
            the length of the dataset.
        """
        self._check_idx(idx)
        return super().__getitem__(idx)

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
        Checks that tensor conversion has been performed before pairing.
        """
        for dataset in datasets:
            if not dataset.converted:
                raise ClinicaDLCAPSError(
                    "Tensor conversion must be performed BEFORE pairing. Please call "
                    "'to_tensors' or 'read_tensor_conversion' for each dataset."
                )

    @classmethod
    def _merge_dfs(cls, datasets: list[CapsDataset]) -> pd.DataFrame:
        """
        Checks that consistency between dataframes and merge them.
        """
        # reorder all datasets
        for i, dataset in enumerate(datasets):
            df = dataset.df[[PARTICIPANT_ID, SESSION_ID]]
            if df.duplicated().any():
                raise ClinicaDLCAPSError(
                    "Datasets passed to 'PairedDataset' cannot contain duplicated (participant, session) pairs, "
                    f"but some were founds in dataset {i}:\n {df[df.duplicated(keep=False)]}"
                )
            dataset._df = dataset.df.sort_values(
                [PARTICIPANT_ID, SESSION_ID]
            ).reset_index(drop=True)
            CapsDataset._map_indices_to_images(dataset.df)

        # check (participant, session) pairs consistency
        particpants_sessions = [
            set(
                dataset.df[[PARTICIPANT_ID, SESSION_ID]].itertuples(
                    index=False, name=None
                )
            )
            for dataset in datasets
        ]
        for i in range(len(datasets))[1:]:
            if particpants_sessions[i] != particpants_sessions[0]:
                difference = particpants_sessions[0].symmetric_difference(
                    particpants_sessions[i]
                )
                raise ClinicaDLCAPSError(
                    "To pair datasets, they must have exactly the same (participant, session) pairs. "
                    f"Differences were found for between dataset 0 and dataset {i}:\n"
                    f"{difference}"
                )

        # check consistency on other columns and merge
        stacked: pd.DataFrame = pd.concat(
            [dataset.df for dataset in datasets], keys=range(len(datasets))
        )

        def _resolve(group: pd.Series) -> Any:
            values = group.dropna().unique()
            if len(values) == 0:
                return np.nan
            elif len(values) == 1:
                return values[0]
            else:
                idx, column = group.name
                raise ClinicaDLCAPSError(
                    f"For ({datasets[0].df.loc[idx, PARTICIPANT_ID]}, {datasets[0].df.loc[idx, SESSION_ID]}), "
                    f"different values found for '{column}' across the datasets forming the PairedDataset: {values}"
                )

        return (
            stacked.stack(dropna=False)
            .groupby(level=[1, 2], sort=False)
            .apply(_resolve)
            .unstack()
        )
