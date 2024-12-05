from typing import Optional

from pydantic import NonNegativeInt
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


def _generate_sampler(
    dataset: CapsDataset,
    sampling_weights: Optional[str] = None,
    shuffle: bool = True,
    dp_degree: Optional[int] = None,
    rank: Optional[int] = None,
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
    distributed = rank is not None and dp_degree is not None

    if sampling_weights is not None:
        weights = dataset.df[sampling_weights].values.astype(float)
        length = (
            len(weights) // dp_degree + int(rank < len(weights) % dp_degree)
            if distributed
            else len(weights)
        )
        sampler = WeightedRandomSampler(weights, num_samples=length)
    else:
        if not distributed:
            dp_degree = 1
            rank = 0
        sampler = DistributedSampler(
            dataset, num_replicas=dp_degree, rank=rank, shuffle=shuffle, drop_last=False
        )

    return sampler


def get_dataloader(
    dataset: CapsDataset,
    batch_size: int,
    sampling_weights: Optional[str],
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
    prefetch_factor: Optional[NonNegativeInt],
    dp_degree: Optional[int],
    rank: Optional[int],
):
    sampler = _generate_sampler(
        dataset,
        sampling_weights=sampling_weights,
        shuffle=shuffle,
        dp_degree=dp_degree,
        rank=rank,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=pl_worker_init_function,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
    )

    return loader
