from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from clinicadl.callbacks import (
    EarlyStoppingCallback,
    LoggerCallback,
    LRSchedulerCallback,
    ModelCheckpointCallback,
)
from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.losses.config import BCEWithLogitsLossConfig
from clinicadl.metrics.config import ConfusionMatrixMetricConfig, LossMetricConfig
from clinicadl.models import SupervisedModel
from clinicadl.networks.config import CNNConfig
from clinicadl.optim import OptimizationConfig
from clinicadl.optim.lr_schedulers.config import OneCycleLRConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.split import KFold
from clinicadl.train import ComputationalConfig, Trainer
from clinicadl.transforms.config import (
    ActivationsConfig,
    AsDiscreteConfig,
)

from .utils import build_dataset

if TYPE_CHECKING:
    from clinicadl.data.datasets import Dataset


def _setup(caps_dir: Path, maps_path: Path) -> None:
    # dataset
    dataset = build_dataset(caps_dir)

    # model
    model = SupervisedModel(
        network=CNNConfig(
            in_shape=(1, 16, 20, 17),
            num_outputs=6,
            conv_args={"channels": [1, 1]},
        ),
        loss=BCEWithLogitsLossConfig(),
        optimizer=AdamConfig(),
    )

    # trainer
    optim_config = OptimizationConfig(
        num_epochs=100, evaluation_interval=5, accumulation_steps=2
    )

    trainer = Trainer(
        maps=maps_path,
        model=model,
        metrics={
            "loss": LossMetricConfig(loss_name="loss"),
            "f1": ConfusionMatrixMetricConfig(
                metric_name="f1 score",
                postprocessing=[
                    ActivationsConfig(sigmoid=True, include=["output"]),
                    AsDiscreteConfig(threshold=0.5, include=["output"]),
                ],
            ),
        },
        callbacks=[
            LRSchedulerCallback(
                OneCycleLRConfig(
                    max_lr=1e-3, epochs=optim_config.num_epochs, steps_per_epoch=2
                )
            ),
            EarlyStoppingCallback(metric="loss", patience=3, min_delta=0.1),
            ModelCheckpointCallback(metric="f1"),
            LoggerCallback(progress_bar=False, debug=False),
        ],
        optimization=optim_config,
        overwrite=True,
    )

    return dataset, trainer


def _train(kfold_dir: Path, dataset: Dataset, trainer: Trainer, gpu: bool) -> None:
    kfold = KFold(kfold_dir)

    for split in kfold.get_splits(dataset, splits=[0, 1]):
        split.build_train_loader(sampling_weights="age", batch_size=2, drop_last=True)
        split.build_val_loader()

        trainer.train(
            split,
            computational=ComputationalConfig(
                gpu=gpu, amp=True, channels_last=True, seed=0, deterministic=True
            ),
        )


def _validate(kfold_dir: Path, dataset: Dataset, trainer: Trainer, gpu: bool) -> None:
    kfold = KFold(kfold_dir)
    split = next(iter(kfold.get_splits(dataset, splits=[0])))
    split.build_val_loader()

    trainer.add_metrics(recall=ConfusionMatrixMetricConfig(metric_name="recall"))

    trainer.validate(
        split_idx=0,
        dataloader=split.val_loader,
        metrics=["recall"],
        model_checkpoint="best-f1",
        computational=ComputationalConfig(
            gpu=gpu, amp=True, channels_last=False, seed=0, deterministic=True
        ),
    )


def _test(split_dir: Path, dataset: Dataset, trainer: Trainer, gpu: bool):
    test_dataset = dataset.subset(split_dir / "test_baseline.tsv")
    test_loader = DataLoaderConfig().get_object(test_dataset)

    trainer.test(
        model_checkpoint="split-0_best-f1",
        dataloader=test_loader,
        group_name="oasis",
        computational=ComputationalConfig(
            gpu=gpu, amp=False, channels_last=False, seed=0, deterministic=True
        ),
    )
    time.sleep(1)  # so that the exec files don't have the same name
    trainer.test(
        model_checkpoint="split-0_final",
        dataloader=test_loader,
        group_name="oasis",
        computational=ComputationalConfig(
            gpu=gpu, amp=True, channels_last=True, seed=0, deterministic=True
        ),
    )


def _test_train(
    tmp_path: Path,
    ref: Path,
    caps_dir: Path,
    split_dir: Path,
    kfold_dir: Path,
    gpu: bool,
) -> None:
    from ..utils import compare_maps_dir

    maps_path = tmp_path / "maps"

    dataset, trainer = _setup(caps_dir, maps_path)
    _train(kfold_dir, dataset, trainer, gpu=gpu)
    _validate(kfold_dir, dataset, trainer, gpu=gpu)
    _test(split_dir, dataset, trainer, gpu=gpu)

    compare_maps_dir(maps_path, ref)


def test_train(tmp_path, ref_data, caps_dir, split_dir, kfold_dir):
    ref_maps = ref_data / "maps_test_basics"
    _test_train(tmp_path, ref_maps, caps_dir, split_dir, kfold_dir, gpu=False)


@pytest.mark.gpu
def test_train_gpu(tmp_path, ref_data, caps_dir, split_dir, kfold_dir):
    ref_maps = ref_data / "maps_test_basics_gpu"
    _test_train(tmp_path, ref_maps, caps_dir, split_dir, kfold_dir, gpu=True)
