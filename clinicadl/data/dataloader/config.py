from typing import Optional

from pydantic import NonNegativeInt, PositiveInt
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torch.utils.data import WeightedRandomSampler as BaseWeightedRandomSampler

from clinicadl.data.datasets import CapsDataset
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.seed import pl_worker_init_function

from .defaults import (
    BATCH_SIZE,
    DP_DEGREE,
    DROP_LAST,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    RANK,
    SAMPLING_WEIGHTS,
    SHUFFLE,
)


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
    """Config class to parametrize a PyTorch DataLoader."""

    batch_size: PositiveInt = BATCH_SIZE
    sampling_weights: Optional[str] = SAMPLING_WEIGHTS
    shuffle: bool = SHUFFLE
    drop_last: bool = DROP_LAST
    num_workers: NonNegativeInt = NUM_WORKERS
    prefetch_factor: Optional[NonNegativeInt] = PREFETCH_FACTOR
    pin_memory: bool = PIN_MEMORY

    def _generate_sampler(
        self,
        dataset: CapsDataset,
        dp_degree: Optional[int] = DP_DEGREE,
        rank: Optional[int] = RANK,
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
        distributed = dp_degree is not None

        if self.sampling_weights:
            try:
                weights = dataset.df[self.sampling_weights].values.astype(float)
            except KeyError as exc:
                raise KeyError(
                    f"Got {self.sampling_weights} for 'sampling_weights' but there is no "
                    "column named like that in the dataframe of the dataset."
                ) from exc
            length = (
                len(weights) // dp_degree + int(rank < len(weights) % dp_degree)
                if distributed
                else len(weights)
            )
            sampler = WeightedRandomSampler(weights, num_samples=length)  # type: ignore
        else:
            if not distributed:
                dp_degree = 1
                rank = 0
            sampler = DistributedSampler(
                dataset,
                num_replicas=dp_degree,
                rank=rank,
                shuffle=self.shuffle,
                drop_last=False,  # not the same as self.drop_last
            )

        return sampler

    def get_dataloader(
        self,
        dataset: CapsDataset,
        dp_degree: Optional[int] = DP_DEGREE,
        rank: Optional[int] = RANK,
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
            **self.model_dump(),
        )

        return loader
