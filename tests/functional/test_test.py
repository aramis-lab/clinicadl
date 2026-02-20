import time
from pathlib import Path

import pytest

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.train import ComputationalConfig, Trainer

from .utils import build_dataset

CAPS_DIR = Path("/Users/thibault.devarax/Desktop/code/clinicadl_data_ci/new_data/caps")
SPLIT_DIR = CAPS_DIR / "splits" / "split"
TRAIN_MAPS_DIR = Path(
    "/Users/thibault.devarax/Desktop/code/clinicadl_data_ci/new_data/maps_test_train"
)
REF_MAPS_DIR = Path(
    "/Users/thibault.devarax/Desktop/code/clinicadl_data_ci/new_data/maps_test_test"
)


@pytest.fixture
def maps_path(tmp_path):
    import shutil

    shutil.copytree(TRAIN_MAPS_DIR, tmp_path / "maps")
    return tmp_path / "maps"


def _test(maps: Path):
    trainer = Trainer.from_maps(maps)
    test_dataset = build_dataset(CAPS_DIR).subset(SPLIT_DIR / "test_baseline.tsv")
    test_loader = DataLoaderConfig().get_object(test_dataset)

    trainer.test(
        model_checkpoint="split-0_best-f1",
        dataloader=test_loader,
        group_name="oasis",
        computational=ComputationalConfig(
            gpu=False, amp=False, channels_last=False, seed=0, deterministic=True
        ),
    )
    time.sleep(1)  # so that the exec files don't have the same name
    trainer.test(
        model_checkpoint="split-0_final",
        dataloader=test_loader,
        group_name="oasis",
        computational=ComputationalConfig(
            gpu=False, amp=False, channels_last=False, seed=0, deterministic=True
        ),
    )


def test_test(maps_path):
    from ..utils import compare_maps_dir

    _test(maps_path)

    compare_maps_dir(maps_path, REF_MAPS_DIR)
