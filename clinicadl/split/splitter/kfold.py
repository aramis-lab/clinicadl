from pathlib import Path
from typing import Generator, Optional, Sequence

from pydantic import PositiveInt, field_validator

from clinicadl.data.datasets import ClinicaDLDataset
from clinicadl.split.split import Split
from clinicadl.split.splitter.splitter import (
    Splitter,
    SplitterConfig,
    SubjectsSessionsSplit,
)
from clinicadl.utils.dictionary.words import SPLIT


class KFoldConfig(SplitterConfig):
    """
    Configuration for K-Fold cross-validation splits.
    """

    _json_name: str = "kfold_config"

    n_splits: PositiveInt
    stratification: Optional[str]

    @field_validator("n_splits", mode="after")
    @classmethod
    def _n_splits_validator(cls, v: int) -> int:
        """Checks that 'n_splits' is greater than 2."""
        assert v >= 2, "'n_splits' must be at least 2."
        return v

    def get_split_subdir(self, split: int, create: bool = False) -> Path:
        """
        Returns the subdirectory of a split of a K-Fold, and creates this directory if it does not
        exist yet.

        Parameters
        ----------
        split : int
            The index of the split.
        create : bool (optional, default=False)
            Create the directory if it doesn't exist.

        Returns
        -------
        Path
            The path to the split subdirectory.
        """
        split_dir = self.split_dir / f"{SPLIT}-{split}"
        if create:
            split_dir.mkdir(parents=True, exist_ok=True)

        return split_dir

    def _check_split_dirs(self) -> None:
        """Checks all the splits directories."""
        for i in range(self.n_splits):
            self._check_split_dir(self.get_split_subdir(i))


class KFold(Splitter):
    """
    To handle a K-Fold cross-validator.

    This object will read a split directory returned by :py:func:`~clinicadl.split.make_kfold`,
    and can then be used to split any :py:class:`~clinicadl.data.datasets.ClinicaDLDataset` using :py:meth:`~KFold.get_splits`,
    provided that all the (participant, session) pairs in the dataset are mentioned in the split directory.

    Parameters
    ----------
    split_dir : PathType
        The split directory, returned by :py:func:`~clinicadl.split.make_kfold`.

    Raises
    ------
    FileNotFoundError
        If ``split_dir`` does not exist or if a required file is missing in this directory.

    See Also
    --------
    :py:class:`~clinicadl.split.SingleSplit`
    """

    _config_type = KFoldConfig

    def get_splits(
        self,
        dataset: ClinicaDLDataset,
        eval_dataset: Optional[ClinicaDLDataset] = None,
        splits: Optional[Sequence[int]] = None,
    ) -> Generator[Split, None, None]:
        """
        Splits a dataset according to the splits found in the K-Fold directory, and
        yields the splits by their indices.

        Parameters
        ----------
        dataset : ClinicaDLDataset
            The :py:class:`~clinicadl.data.datasets.CapsDataset` to split.
        eval_dataset : Optional[ClinicaDLDataset], default=None
            If not ``None``, it will be understood as the dataset from which the validation dataset should be created, and
            ``dataset`` will be the dataset from which the training dataset will be created (see examples). If ``None``, both
            training and validation datasets are built from ``dataset``.
        splits : Optional[Sequence[int]], (optional, default=None)
            Indices of the splits to get. If ``None``, will return all the splits.

        Yields
        ------
        Split
            The train and validation datasets for each requested split, in a :py:class:`~clinicadl.split.Split`
            object.

        Raises
        ------
        IndexError
            If one of the requested split indices is out of range.

        Examples
        --------
        .. code-block::

            >>> df  # quick look at the data
                participant_id	session_id
            0	sub-000	        ses-M000
            1	sub-000	        ses-M003
            2	sub-100	        ses-M000
            3	sub-100	        ses-M012
            4	sub-999	        ses-M099
            5	sub-999	        ses-M999

        .. code-block::

            from clinicadl.split import KFold
            from clinicadl.data import datasets, datatypes
            from clinicadl.transforms import Transforms, extraction

            dataset = datasets.CapsDataset(
                "caps_dir",
                data=df,
                datatype=datatypes.PETLinear(
                    tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
                ),
                transforms=Transforms(extraction=extraction.Patch()),
            )
            splitter = KFold("split_dir/3_fold")

            splits_iterator = splitter.get_splits(dataset)
            split = next(iter(splits_iterator))

        .. code-block::

            >>> split.train_dataset.df
                participant_id	session_id
            0	sub-000	        ses-M000
            1	sub-000	        ses-M003
            2	sub-100	        ses-M000
            3	sub-100	        ses-M012
            >>> split.val_dataset.df
                participant_id	session_id
            0	sub-999	        ses-M099
            1	sub-999	        ses-M999


        Now, let's say you want to train your model on patches, but evaluate it on images:

        .. code-block::

            eval_dataset = datasets.CapsDataset(
                "caps_dir",
                data=df,
                datatype=datatypes.PETLinear(
                    tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
                ),
            )

            splits_iterator = splitter.get_splits(dataset, eval_dataset=eval_dataset)
            split = next(iter(splits_iterator))

        .. code-block::

            >>> split.train_dataset.extraction
                Patch(patch_size=(50, 50, 50), stride=(50, 50, 50), extract_method='patch')
            >>> split.val_dataset.extraction
                Image(extract_method='image')

        """
        if splits is None:
            splits = list(range(self.config.n_splits))

        for split in splits:
            if split not in range(self.config.n_splits):
                raise IndexError(
                    f"Split '{split}' doesn't exist. There are {self.config.n_splits} splits, numbered from 0 to {self.config.n_splits - 1}."
                )
            yield self._get_split(
                split_id=split, dataset=dataset, eval_dataset=eval_dataset
            )

    def _read_splits(self) -> list[SubjectsSessionsSplit]:
        """
        Load all splits in 'split_dir' from the tsv files.
        """
        self.config: KFoldConfig
        return [
            self._read_split(self.config.get_split_subdir(i))
            for i in range(self.config.n_splits)
        ]
