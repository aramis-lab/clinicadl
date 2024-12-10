from typing import Optional

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torch.utils.data import WeightedRandomSampler as BaseWeightedRandomSampler

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.utils.seed import pl_worker_init_function


class WeightedRandomSampler(BaseWeightedRandomSampler):
    def set_epoch(self, epoch: int) -> None:
        """
        Fake method to simulate 'set_epoch' of pytorch's DistributedSampler.
        To be able to always call sampler.set_epoch(), no matter the sampler.
        """
        pass


class DataLoaderConfig(BaseModel):
    batch_size: PositiveInt = 10
    # sampling_weights: Optional[str] = None
    shuffle: bool = False
    num_workers: int = 0
    drop_last: bool = False
    prefetch_factor: Optional[NonNegativeInt] = None

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    def _generate_sampler(
        self,
        dataset: CapsDataset,
        sampling_weights: Optional[str] = None,
        dp_degree: PositiveInt = 1,
        rank: int = 0,
    ) -> Sampler:
        """
        Returns sampler according to the wanted options.

        Args:
            dataset: the dataset.
            sampling_weights: the column of sample weights in the tsv. If None, no weights
            will be used.
            shuffle: if no sampling_weights are passed, whether to shuffle or not.
            dp_degree: the degree of data parallelism.
            rank: process id within the data parallelism communicator.
        Returns:
            callable given to the training data loader.
        """
        distributed = rank == 0 or dp_degree == 1

        if sampling_weights:
            weights = dataset.df[sampling_weights].values.astype(float)
            length = (
                len(weights) // dp_degree + int(rank < len(weights) % dp_degree)
                if distributed
                else len(weights)
            )
            sampler = WeightedRandomSampler(weights, num_samples=length)  # type: ignore
        else:
            if distributed:
                rank = 0
                dp_degree = 1
            sampler = DistributedSampler(
                dataset,
                num_replicas=dp_degree,
                rank=rank,
                shuffle=self.shuffle,
                drop_last=False,
            )

        return sampler

    def get_dataloader(
        self,
        dataset: CapsDataset,
        sampling_weights: Optional[str] = None,
        dp_degree: PositiveInt = 1,
        rank: int = 0,
    ):
        loader = DataLoader(
            dataset=dataset,
            sampler=self._generate_sampler(dataset, sampling_weights, dp_degree, rank),
            worker_init_fn=pl_worker_init_function,
            **self.model_dump(),
        )

        return loader
