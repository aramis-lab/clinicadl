from typing import Optional

from pydantic import NonNegativeInt, PositiveInt, model_validator
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torch.utils.data import WeightedRandomSampler as BaseWeightedRandomSampler

from clinicadl.data.dataloader import BatchLoader
from clinicadl.data.datasets import CapsDataset
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.seed import pl_worker_init_function


class WeightedRandomSampler(BaseWeightedRandomSampler):
    """
    Modifies PyTorch's WeightedRandomSampler to have a similar behavior to
    PyTorch's DistributedSampler.
    """

    def set_epoch(self, epoch: int) -> None:
        """
        Fake method to simulate 'set_epoch' of PyTorch's DistributedSampler.
        To be able to always call sampler.set_epoch(), no matter the sampler.
        """


class DataLoaderConfig(ClinicaDLConfig):
    """
    Class to configure a PyTorch DataLoader from a CapsDataset.

    ..sealso::https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    Parameters
    ----------
    dataloader_config : Optional[DataLoaderConfig] (optional, default=None)
        Pre-configured DataLoader configuration.
    batch_size : PositiveInt (optional, default=1)
        Batch size for the DataLoader.
    sampling_weights : Optional[str] (optional, default=None)
        Name of the column in the dataframe of the CapsDataset where to find the sampling
        weights. The column must contain float values.
    shuffle : bool (optional, default=True)
        Whether to shuffle the data.
        .. note:: If `sampling_weights` is passed, the data will be fetched randomly with
        replacement. So, data are shuffled, no matter the argument `shuffle`.
    num_workers : NonNegativeInt (optional, default=0)
        Number of workers for data loading.
    pin_memory : bool (optional, default=True)
        whether to copy Tensors into device/CUDA pinned memory before returning them.
    drop_last : bool (optional, default=False)
        Whether to drop the last incomplete batch.
    prefetch_factor : Optional[int] (optional, default=None)
        Number of batches loaded in advance by each worker. Can't be passed if `num_workers` is 0.
    persistent_workers : bool (optional, default=False)
        Whether to maintain the worker processes alive at the end of an epoch.
        Can't be passed if `num_workers` is 0.

    Raises
    ------
    ValueError
        If `prefetch_factor` or `persistent_workers` is passed, but `num_workers` is 0.
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

    def get_dataloader(
        self,
        dataset: CapsDataset,
        dp_degree: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> DataLoader:
        """
        To get a dataloader from a dataset. The dataloader is parametrized
        with the options stored in this configuration class.

        Parameters
        ----------
        dataset : CapsDataset
            The dataset to put in a Dataloader.
        dp_degree : Optional[int] (optional, default=None)
            The degree of data parallelism. None if no data parallelism.
        rank : Optional[int] (optional, default=None)
            Process id within the data parallelism communicator.
            None if no data parallelism.

        Returns
        -------
        DataLoader
            The dataloader that wraps the dataset.
        """
        loader = DataLoader(
            dataset=dataset,
            sampler=self._generate_sampler(dataset, dp_degree, rank),
            worker_init_fn=pl_worker_init_function,
            collate_fn=lambda x: BatchLoader(
                x
            ),  # TODO: check if we want to maybe return something else in the dataloader ?
            **self.model_dump(exclude=set(["sampling_weights", "shuffle"])),
        )

        return loader

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
    def _get_weights(dataset: CapsDataset, weights_name: str) -> list[float]:
        """
        Gets the list of weights from the column of the dataframe.
        """
        try:
            weights = [
                dataset.get_sample_info(idx, weights_name)
                for idx in range(len(dataset))
            ]
        except KeyError as exc:
            raise KeyError(
                f"Got '{weights_name}' for 'sampling_weights' but there is no "
                "such column in the metadata dataframe of the dataset."
            ) from exc
        try:
            weights = [float(weight) for weight in weights]
        except ValueError as exc:
            raise ValueError(
                f"Got '{weights_name}' for 'sampling_weights' but cannot convert "
                "this column to float values."
            ) from exc

        return weights
