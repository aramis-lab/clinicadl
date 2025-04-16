from typing import Iterator, Optional, Union, overload

from pydantic import NonNegativeInt, PositiveInt, model_validator
from torch.utils.data import DataLoader as TorchDataLoaader
from torch.utils.data import DistributedSampler, Sampler, WeightedRandomSampler

from clinicadl.data.datasets import (
    CapsDataset,
    ConcatDataset,
    PairedDataset,
    UnpairedDataset,
)
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.seed import pl_worker_init_function

from .batch import SimpleBatch, simple_collate_fn, tuple_collate_fn

SimpleDataset = Union[CapsDataset, ConcatDataset]
TupleDataset = Union[PairedDataset, UnpairedDataset]
Dataset = Union[SimpleDataset, TupleDataset]


class _SimpleDataLoader(TorchDataLoaader):
    """To type the iterator."""

    def __iter__(
        self,
    ) -> Iterator[SimpleBatch]:
        return super().__iter__()


class _TupleDataLoader(TorchDataLoaader):
    """To type the iterator."""

    def __iter__(
        self,
    ) -> Iterator[tuple[SimpleBatch, ...]]:
        return super().__iter__()


class DataLoader(TorchDataLoaader):
    """
    Overwrites :py:class:`torch.utils.data.DataLoader` only to add a `set_epoch` method.
    """

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch.

        This ensures a different random ordering for :py:class:`torch.utils.data.distributed.DistributedSampler`
        and a different random mapping for :py:class:`clinicadl.data.datasets.UnpairedDataset` for each epoch.

        Parameters
        ----------
        epoch : int
            Epoch number.
        """
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
        if isinstance(self.dataset, UnpairedDataset):
            self.dataset.set_epoch(epoch)


class DataLoaderConfig(ClinicaDLConfig):
    """
    Configuration class for the DataLoader.

    The DataLoader can then be accessed with :py:meth:`~DataLoaderConfig.get_object`.
    The DataLoader obtained will be a :py:class:`torch.utils.data.DataLoader`.

    Parameters
    ----------
    batch_size : PositiveInt (optional, default=1)
        Batch size for the DataLoader.
    sampling_weights : Optional[str] (optional, default=None)
        Name of the column in the dataframe of the dataset where to find the sampling
        weights. The column must contain float values.
    shuffle : bool (optional, default=True)
        Whether to shuffle the data.
        .. note::
            If ``sampling_weights`` is passed, the data will be fetched randomly with
            replacement, no matter the argument ``shuffle``.
    num_workers : NonNegativeInt (optional, default=0)
        Number of workers for data loading.
    pin_memory : bool (optional, default=True)
        Whether to copy Tensors into device/CUDA pinned memory before returning them.
    drop_last : bool (optional, default=False)
        Whether to drop the last incomplete batch.
    prefetch_factor : Optional[int] (optional, default=None)
        Number of batches loaded in advance by each worker. Can't be passed if ``num_workers`` is 0.
    persistent_workers : bool (optional, default=False)
        Whether to maintain the worker processes alive at the end of an epoch.
        Can't be passed if ``num_workers`` is 0.

    Raises
    ------
    ValueError
        If ``prefetch_factor`` or ``persistent_workers`` is passed, but ``num_workers`` is 0.

    Examples
    --------
    """

    batch_size: PositiveInt = 1
    sampling_weights: Optional[str] = None
    shuffle: bool = True
    num_workers: NonNegativeInt = 0
    pin_memory: bool = True
    drop_last: bool = False
    prefetch_factor: Optional[NonNegativeInt] = None
    persistent_workers: bool = False

    @model_validator(mode="after")
    def validate_worker_parameters(self):
        """Checks that 'prefetch_factor' is None if 'num_workers' = 0."""
        if self.num_workers == 0 and self.prefetch_factor:
            raise ValueError(
                "'prefetch_factor' option can only be specified num_workers > 0. Got "
                f"prefetch_factor={self.prefetch_factor} and num_workers={self.num_workers}"
            )
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError(
                "'persistent_workers' option can only be specified num_workers > 0. Got "
                f"persistent_workers={self.persistent_workers} and num_workers={self.num_workers}"
            )
        return self

    @overload
    def get_object(
        self,
        dataset: SimpleDataset,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> _SimpleDataLoader:
        ...

    @overload
    def get_object(
        self,
        dataset: TupleDataset,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> _TupleDataLoader:
        ...

    def get_object(
        self,
        dataset: Dataset,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        """
        To get a dataloader from a dataset (:py:class:`~clinicadl.data.datasets.CapsDataset`,
        :py:class:`~clinicadl.data.datasets.ConcatDataset`, :py:class:`~clinicadl.data.datasets.PairedDataset` or
        :py:class:`~clinicadl.data.datasets.UnpairedDataset`). The dataloader is parametrized
        with the options stored in this configuration class.

        Parameters
        ----------
        dataset : Dataset
            The ClinicaDL dataset to put in a DataLoader.
        dp_degree : Optional[int] (optional, default=None)
            The degree of data parallelism. ``None`` if no data parallelism.
        rank : Optional[int] (optional, default=None)
            Process id within the data parallelism communicator.
            ``None`` if no data parallelism.

        Returns
        -------
        DataLoader
            The dataloader that wraps the dataset.

        Raises
        ------
        ValueError
            If only one of ``dp_degree`` and ``rank`` is not ``None``.
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
        return DataLoader(
            dataset=dataset,
            sampler=self._generate_sampler(dataset, dp_degree, rank),
            worker_init_fn=pl_worker_init_function,
            collate_fn=tuple_collate_fn
            if isinstance(dataset, TupleDataset)
            else simple_collate_fn,
            **self.model_dump(exclude={"sampling_weights", "shuffle"}),
        )

    def _generate_sampler(
        self,
        dataset: CapsDataset,
        dp_degree: Optional[int],
        rank: Optional[int],
    ) -> Sampler:
        """
        Returns a WeightedRandomSampler if self.sampling_weights is not None, otherwise a
        a DistributedSampler, even when data parallelism is not performed (in this case
        the degree of data parallelism is set to 1, so it is equivalent to a simple PyTorch
        RandomSampler if self.shuffle is True or no sampler if self.shuffle is False).
        """
        if (rank is not None and dp_degree is None) or (
            dp_degree is not None and rank is None
        ):
            raise ValueError(
                "For data parallelism, none of 'dp_degree' and 'rank' can be None. "
                f"Got rank={rank} and dp_degree={dp_degree}"
            )
        if dp_degree is None:
            dp_degree = 1
            rank = 0

        if self.sampling_weights and rank is not None:
            weights = self._get_weights(dataset, self.sampling_weights)
            length = len(weights) // dp_degree + int(rank < len(weights) % dp_degree)
            sampler = WeightedRandomSampler(weights, num_samples=length)  # type: ignore
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dp_degree,
                rank=rank,
                shuffle=self.shuffle,
                drop_last=False,  # not the same as self.drop_last
            )

        return sampler

    @staticmethod
    def _get_weights(dataset: Dataset, weights_name: str) -> list[float]:
        """
        Gets the list of weights from the column of the dataframe.
        """
        if isinstance(dataset, UnpairedDataset):
            raise ValueError("Can't use 'sampling_weights' with UnpairedDataset.")
        try:
            weights = [
                dataset.get_sample_info(idx, weights_name)
                for idx in range(len(dataset))
            ]
        except KeyError as exc:
            raise KeyError(
                f"Failed to get the column '{weights_name}' in the dataframe of the dataset."
            ) from exc
        try:
            weights = [float(weight) for weight in weights]
        except ValueError as exc:
            raise ValueError(
                f"Got '{weights_name}' for 'sampling_weights' but cannot convert "
                "this column to float values."
            ) from exc

        return weights
