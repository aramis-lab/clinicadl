from pathlib import Path
from typing import Any, Optional

from pydantic import NonNegativeInt, PositiveInt

from clinicadl.data.dataloader import DataLoader, DataLoaderConfig
from clinicadl.data.datasets.types import Dataset
from clinicadl.utils.config import ClinicaDLConfig


class Split(ClinicaDLConfig):
    """
    Dataclass that contains a split.

    More precisely, the dataclass will first contain the training and validation datasets, as well as
    the split index and the split directory used to split the dataset.

    Then, when :py:meth:`~Split.build_train_loader` and :py:meth:`~Split.build_val_loader` will be called,
    the dataclass will also contain the training and validation :py:class:`~torch.utils.data.DataLoader`.

    Finally, to instantiate Data Parallelism, that will distribute the training and validation sets
    across devices, the user can use :py:meth:`~Split.parallelism`.

    Attributes
    ----------
    index : NonNegativeInt
        The index of the split.
    split_dir : Path
        Directory from which the split was built.
    train_dataset : Dataset
        The training set.
    val_dataset : Dataset
        The validation set.
    train_loader : Optional[DataLoader]
        The training PyTorch DataLoader. Will be None until :py:meth:`~Split.build_train_loader`
        is called.
    val_loader : Optional[DataLoader]
        The validation PyTorch DataLoader. Will be None until :py:meth:`~Split.build_val_loader`
        is called.

    ..
        train_loader_config : Optional[DataLoaderConfig]
            A dataclass saving the parameters used when calling :py:meth:`~Split.build_train_loader`.
            For reproducibility.
        val_loader_config : Optional[DataLoaderConfig]
            A dataclass saving the parameters used when calling :py:meth:`~Split.build_val_loader`.
            For reproducibility.
    """

    index: NonNegativeInt
    split_dir: Path
    train_dataset: Dataset
    val_dataset: Dataset
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    train_loader_config: Optional[DataLoaderConfig] = None
    val_loader_config: Optional[DataLoaderConfig] = None
    _dp_degree: Optional[PositiveInt] = None
    _rank: Optional[NonNegativeInt] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Customized version of 'model_dump'.

        Returns the serialized config class.
        """
        dict_ = super().model_dump(
            exclude={"train_loader", "val_loader", "train_dataset", "val_dataset"}
        )

        dict_["val_dataset"] = self.val_dataset.describe()
        dict_["train_dataset"] = self.train_dataset.describe()

        return dict_

    def reset(self) -> None:
        """
        Resets the computed fields of the Split object
        (``train_loader``, ``val_loader``, etc.).
        """
        self.train_loader = None
        self.val_loader = None
        self.train_loader_config = None
        self.val_loader_config = None
        self._dp_degree = None
        self._rank = None

    def parallelism(self, dp_degree: int, rank: int) -> None:
        """
        Instantiates data parallelism. Training and validation sets will then be distributed
        across devices.

        Parameters
        ----------
        dp_degree : int
           The degree of data parallelism.
        rank : int
            Process id within the data parallelism communicator.

        Raises
        ------
        ValueError
            If ``rank`` is greater than ``dp_degree``.
        """
        self._dp_degree = dp_degree
        self._rank = rank
        if self.train_loader_config:
            self.train_loader = self.train_loader_config.get_object(
                dataset=self.train_dataset,
                dp_degree=self._dp_degree,
                rank=self._rank,
            )
        if self.val_loader_config:
            self.val_loader = self.val_loader_config.get_object(
                dataset=self.val_dataset,
                dp_degree=self._dp_degree,
                rank=self._rank,
            )

    def build_train_loader(
        self,
        dataloader_config: Optional[DataLoaderConfig] = None,
        *,
        batch_size: int = 1,
        sampling_weights: Optional[str] = None,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
    ) -> None:
        """
        Builds a :py:class:`~torch.utils.data.DataLoader` for the training set of the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
            A pre-configured :py:class:`~clinicadl.data.dataloader.DataLoaderConfig`.
            If passed, the arguments in this configuration object will prevail, otherwise
            the following arguments will be used.
        batch_size : int (optional, default=1)
            Batch size for the DataLoader. Used if ``dataloader_config`` is not provided.
        sampling_weights : Optional[str] (optional, default=None)
            Name of the column in the dataframe of the dataset where to find the sampling
            weights. The column must contain ``float`` values.

            The probability of sampling a certain sample is proportional to the associated value
            in this column of the dataframe.

            Used if ``dataloader_config`` is not provided.
        shuffle : bool (optional, default=True)
            Whether to shuffle the data.

            .. note::

                If ``sampling_weights`` is passed, the data will be fetched randomly with
                replacement, no matter the value of ``shuffle``.

            Used if ``dataloader_config`` is not provided.
        num_workers : int (optional, default=0)
            Number of workers for data loading. Used if ``dataloader_config`` is not provided.
        pin_memory : bool (optional, default=True)
            Whether to copy Tensors into device/CUDA pinned memory before returning them.
            Used if ``dataloader_config`` is not provided.
        drop_last : bool (optional, default=False)
            Whether to drop the last incomplete batch. Used if ``dataloader_config`` is not provided.
        prefetch_factor : Optional[int] (optional, default=None)
            Number of batches loaded in advance by each worker. Can't be passed if ``num_workers=0``.
            Used if ``dataloader_config`` is not provided.
        persistent_workers : bool (optional, default=False)
            Whether to maintain the worker processes alive at the end of an epoch.
            Can't be passed if ``num_workers=0``. Used if ``dataloader_config`` is not provided.

        Raises
        ------
        ValueError
            If ``prefetch_factor`` or ``persistent_workers`` is passed, but ``num_workers=0``.
        ValueError
            If the dataset is an :py:class:`~clinicadl.data.datasets.UnpairedDataset`,
            and ``sampling_weights`` is not ``None``.
        KeyError
            If ``sampling_weights`` is not ``None``, but there is no column named like
            ``sampling_weights`` in the dataframe of the dataset.
        ValueError
            If ``sampling_weights`` is not ``None`` and the associated column cannot
            be converted to float values.
        """
        if dataloader_config:
            self.train_loader_config = dataloader_config
        else:
            self.train_loader_config = DataLoaderConfig(
                batch_size=batch_size,
                sampling_weights=sampling_weights,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        self.train_loader = self.train_loader_config.get_object(
            dataset=self.train_dataset,
            dp_degree=self._dp_degree,
            rank=self._rank,
        )

    def build_val_loader(
        self,
        dataloader_config: Optional[DataLoaderConfig] = None,
        *,
        batch_size: int = 1,
        sampling_weights: Optional[str] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
    ) -> None:
        """
        Builds a :py:class:`~torch.utils.data.DataLoader` for the validation set of the split.

        Parameters
        ----------
        dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
            A pre-configured :py:class:`~clinicadl.data.dataloader.DataLoaderConfig`.
            If passed, the arguments in this configuration object will prevail, otherwise
            the following arguments will be used.
        batch_size : int (optional, default=1)
            Batch size for the DataLoader. Used if ``dataloader_config`` is not provided.
        sampling_weights : Optional[str] (optional, default=None)
            Name of the column in the dataframe of the dataset where to find the sampling
            weights. The column must contain ``float`` values.

            The probability of sampling a certain sample is proportional to the associated value
            in this column of the dataframe.

            Used if ``dataloader_config`` is not provided.
        shuffle : bool (optional, default=False)
            Whether to shuffle the data.

            .. note::

                If ``sampling_weights`` is passed, the data will be fetched randomly with
                replacement, no matter the value of ``shuffle``.

            Used if ``dataloader_config`` is not provided.
        num_workers : int (optional, default=0)
            Number of workers for data loading. Used if ``dataloader_config`` is not provided.
        pin_memory : bool (optional, default=True)
            Whether to copy Tensors into device/CUDA pinned memory before returning them.
            Used if ``dataloader_config`` is not provided.
        drop_last : bool (optional, default=False)
            Whether to drop the last incomplete batch. Used if ``dataloader_config`` is not provided.
        prefetch_factor : Optional[int] (optional, default=None)
            Number of batches loaded in advance by each worker. Can't be passed if ``num_workers=0``.
            Used if ``dataloader_config`` is not provided.
        persistent_workers : bool (optional, default=False)
            Whether to maintain the worker processes alive at the end of an epoch.
            Can't be passed if ``num_workers=0``. Used if ``dataloader_config`` is not provided.

        Raises
        ------
        ValueError
            If ``prefetch_factor`` or ``persistent_workers`` is passed, but ``num_workers=0``.
        ValueError
            If the dataset is an :py:class:`~clinicadl.data.datasets.UnpairedDataset`,
            and ``sampling_weights`` is not ``None``.
        KeyError
            If ``sampling_weights`` is not ``None``, but there is no column named like
            ``sampling_weights`` in the dataframe of the dataset.
        ValueError
            If ``sampling_weights`` is not ``None`` and the associated column cannot
            be converted to float values.
        """
        if dataloader_config:
            self.val_loader_config = dataloader_config
        else:
            self.val_loader_config = DataLoaderConfig(
                batch_size=batch_size,
                sampling_weights=sampling_weights,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        self.val_loader = self.val_loader_config.get_object(
            dataset=self.val_dataset,
            dp_degree=self._dp_degree,
            rank=self._rank,
        )
