from pathlib import Path
from typing import Iterator, Optional, overload

from pydantic import NonNegativeInt, PositiveInt, model_validator
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import DistributedSampler, Sampler, WeightedRandomSampler

from clinicadl.data.datasets import (
    PairedDataset,
    UnpairedDataset,
)
from clinicadl.data.datasets.types import Dataset, SimpleDataset, TupleDataset
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.json import read_json, update_json, write_json
from clinicadl.utils.seed import pl_worker_init_function

from .batch import SimpleBatch, simple_collate_fn, tuple_collate_fn


class DataLoader(TorchDataLoader):
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


class _SimpleDataLoader(DataLoader):
    """To type the iterator."""

    def __iter__(
        self,
    ) -> Iterator[SimpleBatch]:
        return super().__iter__()


class _TupleDataLoader(DataLoader):
    """To type the iterator."""

    def __iter__(
        self,
    ) -> Iterator[tuple[SimpleBatch, ...]]:
        return super().__iter__()


class DataLoaderConfig(ClinicaDLConfig):
    """
    Configuration class to create a DataLoader.

    **This object is the only type of DataLoader that will be accepted by other ClinicaDL objects.**
    So, ``ClinicaDL`` won't work with a raw :py:class:`PyTorch DataLoader <torch.utils.data.DataLoader>`.

    Nevertheless, if you want to access the underlying :py:class:`PyTorch DataLoader <torch.utils.data.DataLoader>`,
    you can use :py:meth:`~DataLoaderConfig.get_object`.

    Parameters
    ----------
    batch_size : PositiveInt, default=1
        Batch size for the DataLoader.
    sampling_weights : Optional[str], default=None
        Name of the column in the DataFrame of the :py:mod:`ClinicaDL dataset <clinicadl.data.datasets>` where to find the sampling
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

    Raises
    ------
    ValueError
        If ``prefetch_factor`` or ``persistent_workers`` is passed, but ``num_workers=0``.

    See Also
    --------
    :py:class:`torch.utils.data.DataLoader`
        For more details on the parameters.

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
        """:noindex:"""

    @overload
    def get_object(
        self,
        dataset: TupleDataset,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> _TupleDataLoader:
        """:noindex:"""

    def get_object(
        self,
        dataset: Dataset,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        """
        To get a :py:class:`PyTorch DataLoader <torch.utils.data.DataLoader>` from a dataset
        :py:mod:`ClinicaDL dataset <clinicadl.data.datasets>`. The dataloader is parametrized
        with the options stored in this configuration class.

        The output of the iterator is a list of :py:class:`~clinicadl.data.structures.DataPoint`
        returned by the underlying :py:class:`~clinicadl.data.datasets.CapsDataset` (or a tuple
        of lists of :py:class:`~clinicadl.data.structures.DataPoint` if the dataset is a
        :py:class:`~clinicadl.data.datasets.PairedDataset` or an :py:class:`~clinicadl.data.datasets.UnpairedDataset`
        - see examples).

        This list has special methods ``get_images`` and ``get_labels`` to get :py:class:`torch.Tensor` instead
        of ``DataPoint``:

        .. code-block:: python

            dataloader = dataloader_config.get_object(caps_dataset)
            for batch in dataloader:
                images = batch.get_images()
                labels = batch.get_labels()

        .. note::
            - ``batch.get_images()`` will return the batch of images as a unique :py:class:`torch.Tensor` if all the images
              in the batch have the **same size**, otherwise it will return a list of :py:class:`torch.Tensor`.
            - ``batch.get_labels()`` will return the batch of labels as a unique :py:class:`torch.Tensor` if all
              the labels in the batch are **homogeneous** (e.g. all scalars, or all images of same size).

        Parameters
        ----------
        dataset : Dataset
            The :py:mod:`ClinicaDL dataset <clinicadl.data.datasets>` to put in the DataLoader.
        dp_degree : Optional[int], default=None
            The degree of data parallelism. ``None`` if no data parallelism.
        rank : Optional[int], default=None
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
            If ``rank`` is greater than ``dp_degree``.
        ValueError
            If the dataset is an :py:class:`~clinicadl.data.datasets.UnpairedDataset`,
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

            mycaps
            ├── data.tsv
            ├── tensor_conversion
            │   └── default_pet-linear_18FAV45_pons2.json
            └── subjects
                ├── sub-001
                │   └── ses-M000
                │       └── pet_linear
                │           ├── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz
                │           └── tensors
                │               └── default
                │                   └── sub-001_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.pt
                    ...
                ...

            The "data.tsv" file looks like:

            participant_id  session_id   age   sex   diagnosis
            sub-001         ses-M000     55.0  M     CN
            sub-001         ses-M003     55.0  M     AD
            sub-002         ses-M000     62.0  F     MCI
            sub-002         ses-M003     62.0  F     AD
            sub-003         ses-M000     67.0  F     CN
            ...

        .. code-block:: python

            from clinicadl.data.datasets import CapsDataset, PairedDataset
            from clinicadl.data.datatypes import PETLinear
            from clinicadl.data.dataloader import DataLoaderConfig

            caps_dataset = CapsDataset(
                caps_directory="mycaps",
                preprocessing=PETLinear(
                    tracer="18FAV45", use_uncropped_image=True, suvr_reference_region="pons2"
                ),
                data="mycaps/data.tsv",
                label="age",
                columns=["age"],
            )
            caps_dataset.read_tensor_conversion()

            dataloader_config = DataLoaderConfig(batch_size=3, shuffle=False)
            dataloader = dataloader_config.get_object(caps_dataset)

        .. code-block:: python

            >>> batch = next(iter(dataloader))
            >>> batch
            [DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1),
             DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1),
             DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1)]
            >>> batch[0]
            DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1)

        We have a list of three ``DataPoints`` (``batch_size=3``). However, if you want to pass your images to a neural network, you need tensors.
        To get them, you can call ``get_images`` and ``get_labels``:

        .. code-block:: python

            >>> images = batch.get_images()
            >>> type(images)
            list    # here, the images don't have the same shape, so the list of images cannot be converted to a single tensor
            >>> images[0].shape
            torch.Size([1, 222, 312, 234])
            >>> images[1].shape
            torch.Size([1, 220, 312, 234])

            >>> batch.get_labels()
            tensor([55., 55., 62.])

        Now, let's see what happens with a :py:class:`~clinicadl.data.datasets.PairedDataset`:

        .. code-block:: python

            caps_dataset_no_label = CapsDataset(
                caps_directory="mycaps",
                preprocessing=PETLinear(
                    tracer="18FAV45", use_uncropped_image=True, suvr_reference_region="pons2"
                ),
                data="mycaps/data.tsv",
            )
            caps_dataset_no_label.read_tensor_conversion()

            paired_dataset = PairedDataset([caps_dataset, caps_dataset_no_label]

            dataloader = dataloader_config.get_object(paired_dataset)

        .. code-block:: python

            >>> batch = next(iter(dataloader))
            >>> batch
            ([DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1),
              DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1),
              DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1)],
             [DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1),
              DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1),
              DataPoint(Keys: ('image', 'label', 'participant', 'session', 'image_path', 'preprocessing', 'extraction'); images: 1)])

        We have a tuple of :math:`n` batches, where :math:`n` is the number of datasets that we paired.
        We can still call ``get_images`` and ``get_labels`` on these batches:

        .. code-block:: python

            >>> batch[0].get_labels()
            tensor([55., 55., 62.])     # from caps_dataset
            >>> batch[1].get_labels()
            [None, None, None]          # from caps_dataset_no_label
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

        if rank >= dp_degree:
            raise ValueError(
                "'rank' must be strictly smaller than 'dp_degree'. Got "
                f"dp_degree={dp_degree} and rank={rank}"
            )

        return DataLoader(
            dataset=dataset,
            sampler=self._generate_sampler(dataset, dp_degree, rank),
            worker_init_fn=pl_worker_init_function,
            collate_fn=tuple_collate_fn
            if isinstance(dataset, (PairedDataset, UnpairedDataset))
            else simple_collate_fn,
            **self.to_dict(exclude={"sampling_weights", "shuffle"}),
        )

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
        if self.sampling_weights:
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

    def write_json(self, json_path: Path, name: str) -> None:
        if json_path.is_file():
            update_json(json_path=json_path, new_data={name: self.to_dict()})
        else:
            write_json(json_path=json_path, data={name: self.to_dict()})
