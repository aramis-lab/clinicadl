from typing import List, Optional, Union

from pydantic import NonNegativeFloat, field_validator

from clinicadl.data.datasets.types import Dataset
from clinicadl.split.split import Split
from clinicadl.split.splitter.splitter import (
    Splitter,
    SplitterConfig,
    SubjectsSessionsSplit,
)


class SingleSplitConfig(SplitterConfig):
    """
    Configuration for simple split.
    """

    _json_name: str = "single_split_config"

    n_test: NonNegativeFloat
    stratification: List[str]
    p_categorical_threshold: NonNegativeFloat
    p_continuous_threshold: NonNegativeFloat

    @field_validator("p_categorical_threshold", "p_continuous_threshold", mode="after")
    @classmethod
    def validate_thresholds(cls, value: Union[float, int], ctx) -> float:
        if not (0 <= value <= 1):
            raise ValueError(f"'{ctx.field_name}' must be between 0 and 1, got {value}")
        return value

    def _check_split_dirs(self) -> None:
        """Checks the split directory."""
        self._check_split_dir(self.split_dir)


class SingleSplit(Splitter):
    """
    To handle a single training-validation split, as opposed to :py:class:`~clinicadl.split.KFold`
    that can handle several splits.

    This object will read a split directory returned by :py:func:`~clinicadl.split.make_split`,
    and can then be used to split any :py:class:`~clinicadl.data.datasets.CapsDataset` (or
    :py:class:`~clinicadl.data.datasets.ConcatDataset`, :py:class:`~clinicadl.data.datasets.PairedDataset`,
    :py:class:`~clinicadl.data.datasets.UnpairedDataset`) using :py:meth:`~SingleSplit.get_split`,
    provided that all the (participant, session) pairs in the dataset are mentioned in the split directory.

    Parameters
    ----------
    split_dir : Path
        The split directory, returned by :py:func:`~clinicadl.split.make_split`.

    Raises
    ------
    FileNotFoundError
        If ``split_dir`` does not exist or if a required file is missing in this directory.
    """

    @property
    def _associated_config(self) -> type[SingleSplitConfig]:
        """The config class associated to the splitter."""
        return SingleSplitConfig

    def get_split(
        self, dataset: Dataset, eval_dataset: Optional[Dataset] = None
    ) -> Split:
        """
        Splits a dataset according to the split found
        in the split directory.

        Parameters
        ----------
        dataset : Dataset
            The dataset to split. Can be a :py:class:`~clinicadl.data.datasets.CapsDataset`, :py:class:`~clinicadl.data.datasets.ConcatDataset`,
            :py:class:`~clinicadl.data.datasets.PairedDataset`, or :py:class:`~clinicadl.data.datasets.UnpairedDataset`.
        eval_dataset : Optional[Dataset], default=None
            If not ``None``, it will be understood as the dataset from which the validation dataset should be created, and
            ``dataset`` will be the dataset from which the training dataset will be created (see examples). If ``None``, both
            training and validation datasets are built from ``dataset``.

        Returns
        -------
        Split
            A :py:class:`~clinicadl.split.Split` object, with the training and validation datasets for
            the requested split.

        Examples
        --------
        .. code-block::

            >>> df  # quick look at the data
                participant_id	session_id
            0	sub-000	        ses-M000
            1	sub-000	        ses-M003
            2	sub-010	        ses-M003
            3	sub-010	        ses-M012
            4	sub-100	        ses-M000
            5	sub-100	        ses-M012
            6	sub-999	        ses-M099
            7	sub-999	        ses-M999

        .. code-block::

            from clinicadl.split import SingleSplit
            from clinicadl.data import datasets, datatypes
            from clinicadl.transforms import Transforms, extraction

            dataset = datasets.CapsDataset(
                "caps_dir",
                data=df,
                preprocessing=datatypes.PETLinear(
                    tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
                ),
                transforms=Transforms(extraction=extraction.Patch()),
            )
            splitter = SingleSplit("split_dir")
            split = splitter.get_split(dataset)

        .. code-block::

            >>> split.train_dataset.df
                participant_id	session_id
            0	sub-000	        ses-M000
            1	sub-000	        ses-M003
            2	sub-100	        ses-M000
            3	sub-100	        ses-M012
            4	sub-999	        ses-M099
            5	sub-999	        ses-M999
            >>> split.val_dataset.df
                participant_id	session_id
            0	sub-010	        ses-M003

        Now, let's say you want to train your model on patches, but evaluate it on images:

        .. code-block::

            eval_dataset = datasets.CapsDataset(
                "caps_dir",
                data=df,
                preprocessing=datatypes.PETLinear(
                    tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
                ),
            )

            split = splitter.get_split(dataset, eval_dataset=eval_dataset)

        .. code-block::

            >>> split.train_dataset.extraction
                Patch(patch_size=(50, 50, 50), stride=(50, 50, 50), extract_method='patch')
            >>> split.val_dataset.extraction
                Image(extract_method='image')

        """
        return self._get_split(dataset, eval_dataset=eval_dataset)

    def _read_splits(self) -> List[SubjectsSessionsSplit]:
        """
        Load the split from the tsv files in 'split_dir'.
        """
        return [self._read_split(self.split_dir)]
