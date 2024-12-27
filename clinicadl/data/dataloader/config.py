from typing import Optional

from pydantic import NonNegativeInt, PositiveInt, model_validator
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

    @model_validator(mode="after")
    def validate_prefetch_factor(self):
        """Checks that 'prefetch_factor' is None if 'num_workers' = 0."""
        if self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError(
                "'prefetch_factor' option can only be specified num_workers > 0. Got "
                f"prefetch_factor={self.prefetch_factor} and num_workers={self.num_workers}"
            )
        return self

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
            **self.model_dump(exclude=["sampling_weights", "shuffle"]),
        )

        return loader

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
        if dp_degree is None:
            dp_degree = 1
            rank = 0

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
    def _get_weights(dataset: CapsDataset, weights_name: str) -> list[float]:
        try:
            weights = [
                dataset.get_sample_info(idx, weights_name)
                for idx in range(len(dataset))
            ]
        except KeyError as exc:
            raise KeyError(
                f"Got {weights_name} for 'sampling_weights' but there is no "
                "such column the metadata dataframe of the dataset."
            ) from exc
        try:
            weights = [float(weight) for weight in weights]
        except ValueError as exc:
            raise ValueError(
                f"Got {weights_name} for 'sampling_weights' but cannot convert "
                "this column to float values."
            ) from exc

        return weights
