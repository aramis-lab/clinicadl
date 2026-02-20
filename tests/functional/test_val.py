from pathlib import Path

import pytest

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.metrics.config import ConfusionMatrixMetricConfig
from clinicadl.train import ComputationalConfig, Trainer
from clinicadl.transforms import TransformsHandler

from .utils import RandomMasking, ResampleMask

TRAIN_MAPS_DIR = Path(
    "/Users/thibault.devarax/Desktop/code/clinicadl_data_ci/new_data/maps_test_train"
)
REF_MAPS_DIR = Path(
    "/Users/thibault.devarax/Desktop/code/clinicadl_data_ci/new_data/maps_test_val"
)


@pytest.fixture
def maps_path(tmp_path):
    import shutil

    shutil.copytree(TRAIN_MAPS_DIR, tmp_path / "maps")
    return tmp_path / "maps"


def _validate(maps: Path):
    trainer = Trainer.from_maps(maps)

    dataset = CapsDataset.from_json(
        trainer.maps.training.data.validation.splits[0].dataset_json,
        transforms=TransformsHandler(
            sample_transforms=[ResampleMask(), RandomMasking()]
        ),
    )
    dataloader_conf = DataLoaderConfig.from_json(
        trainer.maps.training.data.validation.splits[0].dataloader_json
    )
    dataloder = dataloader_conf.get_object(dataset)

    trainer.add_metrics(recall=ConfusionMatrixMetricConfig(metric_name="recall"))

    trainer.validate(
        split_idx=0,
        dataloader=dataloder,
        metrics=["recall"],
        model_checkpoint="best-f1",
        computational=ComputationalConfig(
            gpu=False, amp=True, channels_last=True, seed=0, deterministic=True
        ),
    )


def test_validate(maps_path):
    from ..utils import compare_maps_dir

    _validate(maps_path)

    compare_maps_dir(maps_path, REF_MAPS_DIR)
