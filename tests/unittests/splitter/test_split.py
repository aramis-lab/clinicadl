from pathlib import Path

import pandas as pd
import pytest
from torch.utils.data import WeightedRandomSampler

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatype import PETLinear
from clinicadl.splitter.split import Split

caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
data = pd.read_csv(caps_dir / "labels.tsv", sep="\t")


def test_build_loaders():
    train_dataset = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            tracer="18FAV45",
            suvr_reference_region="pons2",
            use_uncropped_image=True,
        ),
        data=data.iloc[:6],
    )
    val_dataset = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            tracer="18FAV45",
            suvr_reference_region="pons2",
            use_uncropped_image=True,
        ),
        data=data.iloc[6:],
    )
    split = Split(
        index=0, split_dir="abc", train_dataset=train_dataset, val_dataset=val_dataset
    )

    config = DataLoaderConfig(batch_size=2)
    split.build_train_loader(config, batch_size=1)
    assert split.train_loader.batch_size == 2
    assert split.train_loader.sampler.num_replicas == 1

    config = DataLoaderConfig(num_workers=1)
    split.build_val_loader(config, num_workers=0)
    assert split.val_loader.num_workers == 1
    assert split.val_loader.sampler.num_replicas == 1

    split.parallelism(dp_degree=2, rank=0)
    config = DataLoaderConfig(batch_size=2)
    split.build_train_loader(config)
    assert split.train_loader.batch_size == 2
    assert split.train_loader.sampler.num_replicas == 2

    config = DataLoaderConfig(num_workers=1)
    split.build_val_loader(config)
    assert split.val_loader.num_workers == 1
    assert split.val_loader.sampler.num_replicas == 2

    #####
    split.reset()

    split.build_train_loader(
        batch_size=2,
        sampling_weights="age",
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert split.train_loader.batch_size == 2
    assert split.train_loader.num_workers == 1
    assert not split.train_loader.pin_memory
    assert split.train_loader.drop_last
    assert split.train_loader.prefetch_factor == 2
    assert split.train_loader.persistent_workers
    assert isinstance(split.train_loader.sampler, WeightedRandomSampler)

    split.build_val_loader(
        batch_size=2,
        sampling_weights="age",
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert split.val_loader.batch_size == 2
    assert split.val_loader.num_workers == 1
    assert not split.val_loader.pin_memory
    assert split.val_loader.drop_last
    assert split.val_loader.prefetch_factor == 2
    assert split.val_loader.persistent_workers
    assert isinstance(split.val_loader.sampler, WeightedRandomSampler)

    split.parallelism(dp_degree=2, rank=0)
    split.build_train_loader(shuffle=True)
    assert split.train_loader.sampler.shuffle
    assert split.train_loader.sampler.num_replicas == 2

    config = DataLoaderConfig(num_workers=1)
    split.build_val_loader(shuffle=True)
    assert split.train_loader.sampler.shuffle
    assert split.val_loader.sampler.num_replicas == 2

    # error
    with pytest.raises(ValueError):
        split.parallelism(dp_degree=2, rank=2)
