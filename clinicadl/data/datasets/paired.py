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
from .collection import CollectionDataset, CollectionDatasetConfig
from .multi_samples import MultiSamplesDataset


class PairedDatasetConfig(CollectionDatasetConfig):
    """
    Config class for ``PairedDataset``.
    """

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_n_datasets(
        cls, datasets: Sequence[MultiSamplesDataset]
    ) -> Sequence[MultiSamplesDataset]:
        assert (
            len(datasets) >= 2
        ), f"{cls._get_name()} needs at least 2 datasets to join!"

        return datasets

    @field_validator("datasets", mode="after")
    @classmethod
    def _check_datasets(
        cls, datasets: Sequence[MultiSamplesDataset]
    ) -> Sequence[MultiSamplesDataset]:
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

        return datasets

    @classmethod
    def _get_class(cls) -> type[PairedDataset]:
        """Returns the class associated to this config class."""
        return PairedDataset


class PairedDataset(CollectionDataset, MultiSamplesDataset):
    """
    A useful class to pair multiple :py:class:`~clinicadl.data.datasets.MultiSamplesDataset`
    (e.g. different modalities). Pairing datasets means uniquely associating images across the datasets.

    The keys of this association are the (participant, session) pairs present in the underlying datasets. So, **all
    datasets must contain the same (participant, session) pairs**.

    Furthermore, for a (participant, session) pair, **all the datasets must have the same number of samples**:
    if one of your dataset contains whole images and a second one contains a single slice of the images, it's ok;
    but if the second dataset now contains two slices of the images, this will raise an error because the second dataset
    will thus be two times bigger than the first one, and the two datasets cannot be paired.

    A ``PairedDataset`` will return a tuple of :py:class:`~clinicadl.data.structures.DataPoint` (one for each underlying
    dataset).

    .. note::
        ``PairedDataset`` also accepts :py:class:`~clinicadl.data.datasets.ConcatDataset`.

    Parameters
    ----------
    datasets : Iterable[Union[MultiSamplesDataset, ConcatDataset]]
        List of :py:class:`~clinicadl.data.datasets.MultiSamplesDataset` to be paired.

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
            "mycaps", datatype=T1Linear(use_uncropped_image=True), data=participants_sessions
        )
        caps_pet = CapsDataset(
            "mycaps",
            datatype=PETLinear(
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

    _config_type = PairedDatasetConfig

    def __init__(
        self,
        datasets: Iterable[MultiSamplesDataset],
    ):
        super().__init__(datasets=datasets)
        self._mapping = self._map_datasets()

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
            from each dataset of the ``PairedDataset``, as a ``tuple`` of
            :py:class:`~clinicadl.transforms.extraction.Sample`.
        """
        participant = self.get_sample_info(idx, PARTICIPANT_ID)
        session = self.get_sample_info(idx, SESSION_ID)
        idx_in_image = self._get_index_in_image(idx)

        return tuple(
            dataset[map_[(participant, session)][idx_in_image]]
            for dataset, map_ in zip(self.datasets, self._mapping)
        )

    def _map_datasets(self) -> tuple[dict[tuple[str, str], tuple[int, ...]], ...]:
        """
        For each dataset, indicates the indices associated to the (participant, session).
        """
        mapping = [dict() for _ in self.datasets]
        for participant, session in self.get_participant_session_couples():
            for dataset, map_ in zip(self.datasets, mapping):
                min_idx, max_idx = dataset._get_indices_associated_to(
                    participant, session
                )
                map_[(participant, session)] = tuple(range(min_idx, max_idx + 1))

        return tuple(mapping)

    @staticmethod
    def _merge_dfs(datasets: Sequence[MultiSamplesDataset]) -> pd.DataFrame:
        # get a unique value per column
        concat_df: pd.DataFrame = pd.concat(
            [
                dataset.df.set_index([PARTICIPANT_ID, SESSION_ID])
                for dataset in datasets
            ],
            keys=range(len(datasets)),
            axis=1,
        )

        def _resolve(group: pd.Series) -> Any:
            values = group.dropna().unique()
            if len(values) == 0:
                return np.nan
            elif len(values) == 1:
                return values[0]
            else:
                participant, session, column = group.name
                raise RuntimeError(
                    f"For ({participant}, {session}), "
                    f"different values found for '{column}' across the datasets forming the PairedDataset: {values}"
                )

        merged = (
            concat_df.stack(dropna=False).T.apply(_resolve).unstack(2).reset_index()
        )

        # correct column types
        for column in merged:
            for dataset in datasets:
                try:
                    t = dataset.df[column].dtype
                except KeyError:
                    continue
                merged[column] = merged[column].astype(t)

        return merged.sort_values([PARTICIPANT_ID, SESSION_ID])
