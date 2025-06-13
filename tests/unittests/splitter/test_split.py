import platform
from pathlib import Path

import pandas as pd
import pytest
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear
from clinicadl.splitter.split import Split

CAPS_DIR = Path(__file__).parents[1] / "resources" / "caps_example"
DATA = pd.read_csv(CAPS_DIR / "tsv" / "labels.tsv", sep="\t")
TRAIN_DATASET = CapsDataset(
    CAPS_DIR,
    preprocessing=PETLinear(
        tracer="18FAV45",
        suvr_reference_region="pons2",
        use_uncropped_image=True,
    ),
    data=DATA.iloc[:6],
)
VAL_DATASET = CapsDataset(
    CAPS_DIR,
    preprocessing=PETLinear(
        tracer="18FAV45",
        suvr_reference_region="pons2",
        use_uncropped_image=True,
    ),
    data=DATA.iloc[6:],
)


def test_build_loaders():
    split = Split(
        index=0, split_dir="abc", train_dataset=TRAIN_DATASET, val_dataset=VAL_DATASET
    )

    config = DataLoaderConfig(batch_size=2)
    split.build_train_loader(config, batch_size=1)
    assert split.train_loader.batch_size == 2
    assert split.train_loader.sampler.num_replicas == 1

    config = DataLoaderConfig()
    split.build_val_loader(config)
    assert split.val_loader.sampler.num_replicas == 1

    split.parallelism(dp_degree=2, rank=0)
    assert split.train_loader.sampler.num_replicas == 2
    assert split.val_loader.sampler.num_replicas == 2

    #####
    split.reset()
    split.parallelism(dp_degree=2, rank=1)

    split.build_train_loader(
        batch_size=2,
        sampling_weights="age",
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )
    assert split.train_loader.batch_size == 2
    assert not split.train_loader.pin_memory
    assert split.train_loader.drop_last
    assert isinstance(split.train_loader.sampler, WeightedRandomSampler)

    split.build_val_loader(
        batch_size=2,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )
    assert split.val_loader.batch_size == 2
    assert not split.val_loader.pin_memory
    assert split.val_loader.drop_last
    assert isinstance(split.val_loader.sampler, DistributedSampler)
    assert split.val_loader.sampler.num_replicas == 2

    # error
    with pytest.raises(ValueError):
        split.parallelism(dp_degree=2, rank=2)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="Avoid persistent_workers on macOS"
)
def test_workers():
    split = Split(
        index=0, split_dir="abc", train_dataset=TRAIN_DATASET, val_dataset=VAL_DATASET
    )
    split.build_train_loader(
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert split.train_loader.num_workers == 1
    assert split.train_loader.prefetch_factor == 2
    assert split.train_loader.persistent_workers

    split.build_val_loader(
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
    )
    assert split.train_loader.num_workers == 1
    assert split.train_loader.prefetch_factor == 2
    assert split.train_loader.persistent_workers


def test_to_dict():
    split = Split(
        index=0, split_dir="abc", train_dataset=TRAIN_DATASET, val_dataset=VAL_DATASET
    )
    split.build_train_loader(batch_size=2)
    split.build_val_loader(num_workers=1)
    dict_ = split.to_dict()
    assert sorted(list(dict_.keys())) == sorted(
        [
            "index",
            "split_dir",
            "train_dataset",
            "val_dataset",
            "train_loader_config",
            "val_loader_config",
        ]
    )
    assert dict_["index"] == 0
    assert dict_["split_dir"] == Path("abc")
    assert dict_["train_dataset"]["total_samples"] == 6
    assert dict_["val_dataset"]["total_samples"] == 2
    assert dict_["train_loader_config"]["batch_size"] == 2
    assert dict_["val_loader_config"]["num_workers"] == 1
