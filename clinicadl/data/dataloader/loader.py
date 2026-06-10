from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import DistributedSampler, Sampler, WeightedRandomSampler

from clinicadl.data.datasets import (
    UnpairedDataset,
)
from clinicadl.utils.config import ObjectConfig
from clinicadl.utils.factories import safe_factory_from_json
from clinicadl.utils.objects import HasConfig
from clinicadl.utils.seed import pl_worker_init_function
from clinicadl.utils.typing import PathType

from ..datasets.base import Dataset, SampleT
from .collate import CollateFn, ToBatchCollate, ToBatchesCollate
from .collate.factory import get_collate_from_dict


def _read_collate(serialized_collate: Any) -> Any:
    """
    To read the field 'collate_fn'.
    """
    if isinstance(serialized_collate, dict):
        return get_collate_from_dict(serialized_collate)
    return serialized_collate


class DataLoaderConfig(ObjectConfig):
    """Config class for ``DataLoader``."""

    batch_size: PositiveInt
    sampling_weights: Optional[str]
    shuffle: bool
    num_workers: NonNegativeInt
    pin_memory: bool
    drop_last: bool
    prefetch_factor: Optional[NonNegativeInt]
    persistent_workers: bool
    collate_fn: Optional[CollateFn] = Field(json_schema_extra={"reader": _read_collate})

    @model_validator(mode="after")
    def _validate_worker_parameters(self):
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

    @classmethod
    def _get_class(cls) -> Any:
        return DataLoader

    def get_object(self, dataset: Dataset) -> DataLoader:
        """
        Returns the dataloader associated to this configuration,
        parametrized with the parameters passed by the user.

        Parameters
        ----------
        dataset : Dataset
            The dataset from which data are loaded.

        Returns
        -------
        DataLoader
            The ``DataLoader``.
        """
        return self._get_class()(dataset, **self.to_raw_dict())


class DataLoader(HasConfig["DataLoaderConfig"], TorchDataLoader[SampleT]):
    """
    To load data in batches.

    It inherits from :py:class:`torch.utils.data.DataLoader`. **Only this dataloader is guaranteed to work with ClinicaDL.**

    The format of the output of the iterator can be specified via ``collate_fn``.

    Parameters
    ----------
    dataset : Dataset
        Dataset from which to load the data.
    batch_size : PositiveInt, default=1
        Batch size..
    sampling_weights : Optional[str], default=None
        Name of the column in the :py:attr:`Dataset.df <clinicadl.data.datasets.Dataset.df>` where to find the sampling
        weights. The column must contain ``float`` values.

        The probability of sampling a certain sample is proportional to the associated value
        in this column.

        .. warning::
            ``sampling_weights`` doesn't work with an :py:class:`~clinicadl.data.datasets.UnpairedDataset`.

    shuffle : bool, default=True
        Whether to shuffle the data.

        .. note::

            If ``sampling_weights`` is passed, the data will be fetched randomly with
            replacement, no matter the value of ``shuffle``.

    num_workers : NonNegativeInt, default=0
        Number of workers for data loading.
    pin_memory : bool, default=True
        Whether to copy tensors into device/CUDA pinned memory before returning them.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.
    prefetch_factor : Optional[int], default=None
        Number of batches loaded in advance by each worker. Can't be passed if ``num_workers=0``.
    persistent_workers : bool, default=False
        Whether to maintain the worker processes alive at the end of an epoch.
        Can't be passed if ``num_workers=0``.
    collate_fn : Optional[CollateFn], default=None
        To customize the way samples are collated into batches. See :py:mod:`clinicadl.data.dataloader.collate`.

    Raises
    ------
    ValueError
        If ``prefetch_factor`` or ``persistent_workers`` is passed, but ``num_workers=0``.
    ValueError
        If the dataset is an :py:class:`~clinicadl.data.datasets.UnpairedDataset`
        and ``sampling_weights`` is not ``None``.
    KeyError
        If ``sampling_weights`` is not ``None``, but there is no column named like
        ``sampling_weights`` in the dataframe of the dataset.
    ValueError
        If ``sampling_weights`` is not ``None`` and the associated column cannot
        be converted to float values.

    Examples
    --------
    .. code-block:: text

        Data look like:

        bids
        ├── metadata.tsv
        ├── sub-001
        │   ├── ses-M000
        │   │   ├── pet
        │   │       └── sub-001_ses-M000_trc-18FAV45_pet.nii.gz
            ...
        ...

        The "metadata.tsv" file looks like:

        participant_id  session_id   age   sex
        sub-001         ses-M000     55.0  M
        ...

    .. code-block:: python

        from clinicadl.data.datasets import BidsDataset, PairedDataset
        from clinicadl.io.bids import BidsFileType
        from clinicadl.data.dataloader import DataLoader

        bids = BidsDataset(
            "bids",
            file_type=BidsFileType(data_type="pet", suffix="pet"),
            data="bids/metadata.tsv",
        )
        dataloader = DataLoader(bids, batch_size=3, shuffle=False)

    .. code-block:: python

        >>> batch = next(iter(dataloader))
        >>> batch
        [Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1),
            Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1),
            Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1)]

    Now, let's see what happens with a :py:class:`~clinicadl.data.datasets.PairedDataset`:

    .. code-block:: python

        paired_dataset = PairedDataset([bids, bids])
        dataloader = DataLoader(paired_dataset, batch_size=3, shuffle=False)

    .. code-block:: python

        >>> batch = next(iter(dataloader))
        >>> batch
        ([Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1),
            Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1),
            Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1)],
            [Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1),
            Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1),
            Sample(Keys: ('file_type', 'image_path', 'sample_type', 'sample_position', 'image', 'participant_id', 'session_id'); images: 1)])

    Because, the default behavior is to use :py:class:`~clinicadl.data.dataloader.ToBatchesCollate` to collate batches,
    we obtain here a tuple of :math:`n` batches, where :math:`n` is the number of datasets that we paired.

    See Also
    --------
    :py:class:`torch.utils.data.DataLoader`
        For more details on the parameters.

    """

    _config_type = DataLoaderConfig
    dataset: Dataset[SampleT]

    def __init__(
        self,
        dataset: Dataset[SampleT],
        *,
        batch_size: int = 1,
        sampling_weights: Optional[str] = None,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        collate_fn: Optional[CollateFn] = None,
    ):
        self.config = self._config_type(
            batch_size=batch_size,
            sampling_weights=sampling_weights,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

        if self.config.collate_fn:
            collate_fn = self.config.collate_fn
        else:
            if isinstance(dataset[0], Sequence):
                collate_fn = ToBatchesCollate()
            else:
                collate_fn = ToBatchCollate()

        super().__init__(
            dataset=dataset,
            sampler=self._generate_sampler(dataset, dp_degree=1, rank=0),
            worker_init_fn=pl_worker_init_function,
            collate_fn=collate_fn,
            **self.config.to_dict(
                exclude={"sampling_weights", "shuffle", "collate_fn", "name_"}
            ),
        )

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
        if callable(set_epoch := getattr(self.sampler, "set_epoch", None)):
            set_epoch(epoch)
        if callable(set_epoch := getattr(self.dataset, "set_epoch", None)):
            set_epoch(epoch)

    def _generate_sampler(
        self,
        dataset: Dataset,
        dp_degree: int,
        rank: int,
    ) -> Sampler:
        """
        Returns a WeightedRandomSampler if self.sampling_weights is not None, otherwise a
        a DistributedSampler, even when data parallelism is not performed (in this case
        the degree of data parallelism is set to 1, so it is equivalent to a simple PyTorch
        RandomSampler if self.shuffle is True or no sampler if self.shuffle is False).
        """
        if self.config.sampling_weights:
            weights = self._get_weights(dataset, self.config.sampling_weights)
            length = len(weights) // dp_degree + int(rank < len(weights) % dp_degree)
            sampler = WeightedRandomSampler(weights, num_samples=length)  # type: ignore
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dp_degree,
                rank=rank,
                shuffle=self.config.shuffle,
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

    @classmethod
    def from_json(
        cls, json_path: PathType, dataset: Dataset[SampleT], **kwargs: Any
    ) -> DataLoader:
        """
        To create the object from a ``json`` file saved with
        :py:meth:`to_json`.

        Parameters
        ----------
        json_path : PathType
            Path to the ``json`` file.
        dataset : Dataset
            The dataset from which the data are loaded.
        kwargs : Any
            Any field of the ``json`` to override.

        Returns
        -------
        Self
            The object instantiated from the file.
        """
        config = cls._config_type.from_json(json_path, **kwargs)
        return config.get_object(dataset)

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        dataset: Dataset[SampleT],
        **kwargs: Any,
    ) -> DataLoader:
        """
        To create the object from a dictionary returned by
        :py:meth:`to_dict`.

        Parameters
        ----------
        config_dict : dict[str, Any]
            The input dictionary.
        dataset : Dataset
            The dataset from which the data are loaded.
        kwargs : Any
            Any field of the dictionary to override.

        Returns
        -------
        Self
            The object instantiated from the dictionary.
        """
        config = cls._config_type.from_dict(config_dict, **kwargs)
        return config.get_object(dataset)


@safe_factory_from_json(factory=DataLoaderConfig.from_json)
def get_dataloader_from_json_safely(
    json_path: PathType, default: Optional[DataLoaderConfig] = None
) -> tuple[Optional[DataLoaderConfig], list[str]]:
    """
    Factory function to get a :py:class:`DataLoaderConfig` from the
    file saved with :py:meth:`DataLoaderConfig.to_json`, which will not raised errors.

    If some fields of the serialized dataloader cannot be read, they will be reported, and
    the field of ``default`` will be used to override them (if not ``None``).

    If it was impossible to read the serialized dataloader, the factory returns ``None``.

    Parameters
    ----------
    json_path : PathType
        The path to the serialized dataloader.
    default : Optional[DataLoaderConfig], default=None
        The :py:class:`DataLoaderConfig` from which to take the default arguments.

    Returns
    -------
    Optional[DataLoaderConfig]
        The deserialized DataLoaderConfig. ``None`` if deserialization was impossible.
    list[str]
        The list of fields that could not be read in the serialized dataloader.
    """
