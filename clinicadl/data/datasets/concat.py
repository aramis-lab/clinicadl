from __future__ import annotations

import warnings
from bisect import bisect_right
from typing import Any, Iterable, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import model_validator
from typing_extensions import Self

from clinicadl.dictionary.words import DATASET_ID
from clinicadl.utils.typing import DataFrameType
from clinicadl.utils.variables import SPACING_RTOL

from .collection import CollectionDataset, CollectionDatasetConfig
from .multi_samples import MultiSamplesDataset
from .output import Sample, Sample2D


class ConcatDatasetConfig(CollectionDatasetConfig):
    """
    Config class for ``ConcatDataset``.
    """

    raise_warnings: bool

    @model_validator(mode="after")
    def _check_dimensionality_and_spacing(self) -> Self:
        if self.raise_warnings:
            self._check_consistency(self.datasets)

        return self

    @staticmethod
    def _check_consistency(
        datasets: Sequence[MultiSamplesDataset],
    ) -> Sequence[MultiSamplesDataset]:
        """
        Checks if all datasets have images of the same dimensionality (2D or 3D).
        """
        _2d = False
        _3d = False
        shapes = set()
        spacings = set()
        for dataset in datasets:
            sample = dataset[0]
            if isinstance(sample, Sample2D) and sample.squeeze:
                _2d = True
            else:
                _3d = True
            shapes.add(tuple(sample.image.shape))
            if not any(
                np.allclose(sample.spacing, spacing, rtol=SPACING_RTOL)
                for spacing in spacings
            ):
                spacings.add(sample.image.spacing)

        if _2d and _3d:
            warnings.warn(
                "You are trying to concatenate datasets with different dimensionalities: at least one of your dataset contains 2D slices (a Sample2D is returned "
                "with 'squeeze=True'), whereas at least one other contains 3D elements. To disable this warning set 'raise_warnings' to False."
            )
        elif len(shapes) > 1:
            warnings.warn(
                f"You are trying to concatenate datasets with different image shapes: found an image of shape {shapes.pop()} in one dataset, "
                f"and one with shape {shapes.pop()} in another. To disable this warning set 'raise_warnings' to False."
            )
        if len(spacings) > 1:
            warnings.warn(
                f"You are trying to concatenate datasets with different voxel spacings: found an image with spacing {spacings.pop()} in one dataset, "
                f"and one with spacing {spacings.pop()} in another. To disable this warning set 'raise_warnings' to False."
            )

        return datasets

    @classmethod
    def _get_class(cls) -> type[ConcatDataset]:
        """Returns the class associated to this config class."""
        return ConcatDataset


class ConcatDataset(CollectionDataset, MultiSamplesDataset):
    """
    A useful class to assemble multiple :py:class:`~clinicadl.data.datasets.MultiSamplesDataset`
    (e.g. from different datasets).

    ``ConcatDataset`` concatenates the input datasets, so the length of the new dataset will be equal to the sum
    of the lengths of each individual dataset.

    Parameters
    ----------
    datasets : Iterable[MultiSamplesDataset]
        List of :py:class:`~clinicadl.data.datasets.MultiSamplesDataset` to be concatenated.
    raise_warnings : bool, default=True
        Whether to raise warnings during concatenation, related to different kinds of issues ``ClinicaDL`` thinks
        the user should be aware of (e.g. datasets of different dimensionality, datasets with different spacings).

    Notes
    -----
    If ``raise_warnings=False``, raises a warning if:
    - the dimensionalities of the datasets are not the same, e.g. a dataset contains 2D slices but
      another one contains 3D images;
    - image shape is not consistent across the datasets;
    - voxel spacing is not consistent across the datasets.

    Examples
    --------
    .. code-block:: text

        Data look like:

        caps_1
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

        caps_2
        ├── tensor_conversion
        │   └── default_t1-linear.json
        └── subjects
            ├── sub-A
            │   └── ses-M000
            │       └── t1_linear
            │           ├── sub-A_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz
            │           └── tensors
            │               └── default
            │                   └── sub-A_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt
                ...
            ...

    .. code-block:: python

        from clinicadl.data.datasets import CapsDataset, ConcatDataset
        from clinicadl.data.datatypes import T1Linear

        caps_1 = CapsDataset("caps_1", datatype=T1Linear(use_uncropped_image=True))
        caps_2 = CapsDataset("caps_2", datatype=T1Linear(use_uncropped_image=True))

        caps_1.read_tensor_conversion()
        caps_2.read_tensor_conversion()

        full_dataset = ConcatDataset([caps_1, caps_2])

    .. code-block:: python

        >>> len(caps_1)
        4
        >>> len(caps_2)
        8
        >>> len(full_dataset)
        12
        >>> full_dataset[0].participant, full_dataset[0].session
        ('sub-001', 'ses-M000')
        >>> full_dataset[4].participant, full_dataset[4].session
        ('sub-A', 'ses-M000')
    """

    _config_type = ConcatDatasetConfig

    def __init__(
        self,
        datasets: Iterable[MultiSamplesDataset],
        raise_warnings: bool = True,
    ):
        super().__init__(datasets=datasets, raise_warnings=raise_warnings)

    def subset(
        self, particpants_sessions: Union[DataFrameType, Sequence[tuple[str, str]]]
    ) -> Self:
        sub_datasets = []
        not_empty = False
        for dataset in self.datasets:
            try:
                sub_datasets.append(dataset.subset(particpants_sessions))
            except RuntimeError:  # empty dataset
                continue
            else:
                not_empty = True

        if not not_empty:
            raise RuntimeError(
                "No (participant, session) pairs are in the dataset. This would lead to an empty dataset!"
            )

        return type(self)(
            sub_datasets,
            raise_warnings=False,
        )

    def get_sample_info(self, idx: int, column: str) -> Any:
        dataset_idx, idx_in_dataset = self._get_dataset_and_rank(idx)
        try:
            return self.datasets[dataset_idx].get_sample_info(idx_in_dataset, column)
        except KeyError as e:
            raise KeyError(
                f"No column named '{column}' in the metadata DataFrame of the dataset from which the sample is taken."
            ) from e

    def __getitem__(self, idx: int) -> Sample:
        dataset_idx, idx_in_dataset = self._get_dataset_and_rank(idx)

        return self.datasets[dataset_idx][idx_in_dataset]

    def _get_dataset_and_rank(self, idx: int) -> tuple[int, int]:
        """
        Gets the dataset from which is the sample.
        """
        self._check_idx(idx)
        cum_len = np.cumsum([len(dataset) for dataset in self.datasets])
        dataset_idx = bisect_right(cum_len, idx)
        if dataset_idx > 0:
            idx_in_dataset = int(idx - cum_len[dataset_idx - 1])
        else:
            idx_in_dataset = idx

        return dataset_idx, idx_in_dataset

    def _count_samples(self) -> None:
        pass

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
