# coding: utf8
from __future__ import annotations

import bisect
import warnings
from logging import getLogger
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset as TorchConcatDataset

from clinicadl.dictionary.words import DATASET_ID, PARTICIPANT_ID, SESSION_ID
from clinicadl.transforms.extraction import Sample
from clinicadl.transforms.extraction.slice import Slice
from clinicadl.utils.exceptions import ClinicaDLCAPSError
from clinicadl.utils.typing import DataType

from .caps_dataset import CapsDataset

logger = getLogger("clinicadl.data.datasets.concat")


class ConcatDataset(TorchConcatDataset):
    """
    ``ConcatDataset`` is a useful class to assemble multiple :py:class:`~clinicadl.data.datasets.CapsDataset`
    (e.g. from different datasets).

    ``ConcatDataset`` concatenates the input datasets, so the length of the new dataset will be equal to the sum
    of the lengths of each individual dataset.

    To assemble ``CapsDatasets``, you must **previously perform**
    :py:func:`tensor conversion <clinicadl.data.datasets.CapsDataset.to_tensors>`.

    Parameters
    ----------
    datasets : Iterable[CapsDataset]
        List of :py:class:`~clinicadl.data.datasets.CapsDataset` to be concatenated.
    ignore_spacing : bool, default=False
        Whether to ignore checks made on voxel spacing. If ``False``, ``ConcatDataset`` will check that the voxel spacing
        is consistent across all the datasets (if the information is provided in the ``.json`` file of the tensor
        conversion).
    raise_warnings : bool, default=True
        Whether to raise warnings during concatenation, related to different kinds of issues ``ClinicaDL`` thinks
        the user should be aware of (e.g. datasets of different dimensionality).

    Raises
    ------
    ClinicaDLCAPSError
        If tensor conversion has not been performed for all the datasets before concatenating.
    ClinicaDLCAPSError
        If ``ignore_spacing=False`` and some datasets don't have the same voxel spacing.

    Notes
    --------
    Also raises a warning if the dimensionalities of the datasets are not the same, e.g. a dataset contains 2D slices but
    another one contains 3D images (unless ``raise_warnings=False``).

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

        caps_1 = CapsDataset("caps_1", preprocessing=T1Linear(use_uncropped_image=True))
        caps_2 = CapsDataset("caps_2", preprocessing=T1Linear(use_uncropped_image=True))

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

    def __init__(
        self,
        datasets: Iterable[CapsDataset],
        ignore_spacing: bool = False,
        raise_warnings: bool = True,
    ):
        self._check_conversion(datasets)
        super().__init__(datasets)
        self.datasets: list[CapsDataset]
        self.converted = True  # for compatibility with CapsDataset

        if raise_warnings:
            self._check_dimensionality()
        if not ignore_spacing:
            self._check_spacing()
        self._df = self._concat_dfs()

    @property
    def df(self) -> pd.DataFrame:
        """The result of the concatenation of the DataFrames of the underlying ``CapsDatasets``."""
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

    def subset(self, data: DataType) -> ConcatDataset:
        """
        To get a subset of the ``ConcatDataset`` from a list of (participant, session) pairs.

        In practice, it will call :py:meth:`CapsDataset.subset <clinicadl.data.datasets.CapsDataset.subset>`
        for all the datasets forming the ``ConcatDataset``.

        Parameters
        ----------
        data : DataType
            A :py:class:`pandas.DataFrame` (or a path to a ``TSV`` file containing the dataframe) with the list of participant-session
            pairs to extract. Please note that this list must be passed via two columns named ``"participant_id"``
            and ``"session_id"`` (other columns won't be considered).

        Returns
        -------
        ConcatDataset
            A subset of the original ``ConcatDataset``, restricted to the (participant, session) pairs mentioned in ``data``.

        Raises
        ------
        ClinicaDLTSVError
            If the DataFrame associated to ``data`` does not contain the columns ``"participant_id"``
            and ``"session_id"``.
        ClinicaDLCAPSError
            If no (participant, session) pairs mentioned in ``data`` are at least in one of the underlying datasets.
            This would lead to an empty ``ConcatDataset``.
        """
        sub_datasets = []
        not_empty = False
        for dataset in self.datasets:
            try:
                sub_datasets.append(dataset.subset(data))
                not_empty = True
            except ClinicaDLCAPSError:  # empty dataset
                continue

        if not not_empty:
            raise ClinicaDLCAPSError(
                "No (participant, session) pairs mentioned in 'data' are in the ConcatDataset. This would lead to an empty dataset!"
            )

        return ConcatDataset(
            sub_datasets,
            ignore_spacing=True,
            raise_warnings=False,
        )

    def describe(self) -> tuple[Dict[str, Any], ...]:
        """
        Returns a description of the ``CapsDatasets`` forming the ``ConcatDataset``.

        Returns
        -------
        tuple[Dict[str, Any], ...]
            The descriptions returned by :py:meth:`CapsDataset.describe
            <clinicadl.data.datasets.CapsDataset.describe>` for each
            dataset forming the ConcatDataset.

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
            The index of the sample in the ConcatDataset.
        column : str
            The information to look for, i.e. a column of the DataFrame containing
            the metadata. The DataFrame is the one from the ``CapsDataset`` associated to
            ``idx``. So, the information accessible depends on ``idx``.

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
            If ``column`` is not in the associated metadata DataFrame.
        """
        self._check_idx(idx)

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx].get_sample_info(sample_idx, column)

    def get_participant_session_couples(self) -> List[Tuple[str, str]]:
        """
        Retrieves all participant-session pairs in the dataset.

        Returns
        -------
        List[Tuple[str, str]]
            The list of (participant, session).
        """
        return list(set(zip(self._df[PARTICIPANT_ID], self._df[SESSION_ID])))

    def __getitem__(self, idx: int) -> Sample:
        """
        Retrieves the sample at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample in the dataset.

        Returns
        -------
        Sample
            A structured output containing the processed data and metadata, as a
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
        Checks that tensor conversion has been performed before concatenation.
        """
        for dataset in datasets:
            if not dataset.converted:
                raise ClinicaDLCAPSError(
                    "Tensor conversion must be performed BEFORE concatenation. Please call "
                    "'to_tensors' or 'read_tensor_conversion' for each dataset."
                )

    def _check_dimensionality(self) -> None:
        """
        Checks if all datasets have images of the same dimensionality (2D or 3D).
        """
        _2d = False
        _3d = False
        for dataset in self.datasets:
            extraction = dataset.extraction
            if isinstance(extraction, Slice) and extraction.squeeze:
                _2d = True
            else:
                _3d = True

        if _2d and _3d:
            warnings.warn(
                "You are trying to concatenate datasets with different dimensionalities: at least one of your dataset contains 2D slices ('Slice' extraction "
                "with 'squeeze=True'), whereas at least one other contains 3D elements. This can be problematic if your network only accepts a specific "
                "dimensionality. To disable this warning set 'raise_warnings' to False."
            )

    def _check_spacing(self) -> None:
        """
        Checks that voxel spacing is consistent across datasets.
        """
        ref_spacing = None
        ref_idx = None
        for i, dataset in enumerate(self.datasets):
            spacing = dataset.tensor_conversion.get_info().spacing
            if spacing is not None:
                if ref_spacing is None:
                    ref_spacing = spacing
                    ref_idx = i
                else:
                    if not np.isclose(spacing, ref_spacing, rtol=1e-2).all():
                        raise ClinicaDLCAPSError(
                            "Different voxel spacings found across datasets: "
                            f"voxel spacing is '{ref_spacing}' in dataset {ref_idx}, "
                            f"but '{spacing}' in dataset {i}.\n"
                            "If you don't care about voxel spacing, set 'ignore_spacing' "
                            "to True to ignore this error."
                        )

    def _concat_dfs(self) -> pd.DataFrame:
        """
        Concatenates the dataframes from all the datasets.
        """
        df: pd.DataFrame = pd.concat(
            [dataset.df for dataset in self.datasets],
            keys=range(len(self.datasets)),
            names=[DATASET_ID],
        )
        CapsDataset._map_indices_to_images(df)

        return df.reset_index(
            drop=False,
            level=DATASET_ID,
        ).reset_index(drop=True)
