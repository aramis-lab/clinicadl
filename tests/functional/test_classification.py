"""
A multi-class, multi-label classification task trained on 2 splits (KFold splitting) with:
- a BidsDataset with data augmentation and custom transforms
  involving individual and common masks;
- a DataLoader with weighted sampling;
- a SupervisedModel;
- metrics with postprocessing (computed on CPU);
- accumulation steps and evaluation interval more than 1;
- amp and channels last memory format (and without);
- lr scheduling;
- early stopping;
- model checkpointing (with metric tracking);
- logging (without debug and progress bar);
- validation on new metrics;
- test (two times on the same group).

The data location (i.e. the device) across the workflow is tested.
Model resetting (or no resetting) is tested.

An error is raised to interrupt the training. Training is then resumed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from clinicadl.callbacks import (
    EarlyStoppingCallback,
    LoggerCallback,
    LRSchedulerCallback,
    ModelCheckpointCallback,
)
from clinicadl.data.dataloader import DataLoader
from clinicadl.data.datasets import BidsDataset
from clinicadl.io.bids import BidsFileType
from clinicadl.losses.config import BCEWithLogitsLossConfig
from clinicadl.metrics.config import ConfusionMatrixMetricConfig, LossMetricConfig
from clinicadl.models import SupervisedModel
from clinicadl.networks.config import CNNConfig
from clinicadl.optim import OptimizationConfig
from clinicadl.optim.lr_schedulers.config import OneCycleLRConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.split import KFold
from clinicadl.train import ComputationalConfig, Trainer
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import (
    ActivationsConfig,
    AsDiscreteConfig,
    OneOfConfig,
    RandomAffineConfig,
    RandomBiasFieldConfig,
    RandomBlurConfig,
    RandomElasticDeformationConfig,
    RandomGammaConfig,
    RandomGhostingConfig,
    RandomMotionConfig,
    RandomNoiseConfig,
    RandomSpikeConfig,
)
from clinicadl.utils.seed import seed_everything_context

from .utils import (
    ErrorCallback,
    RandomMasking,
    ResampleMask,
    TestDeviceCallback,
    TestModelReset,
)

if TYPE_CHECKING:
    from clinicadl.data.datasets import Dataset


def _setup(
    bids_dir: Path, metadata: Path, maps_path: Path, reset_model: bool, gpu: bool
) -> None:
    # dataset
    dataset = BidsDataset(
        bids=bids_dir,
        file_type=BidsFileType(data_type="anat", suffix="T1w"),
        data=metadata,
        masks={
            "head": (
                bids_dir / "derivatives" / "masks",
                BidsFileType(
                    data_type="anat", suffix="mask", with_entities={"label": "head"}
                ),
            ),
            "left_hemisphere": bids_dir
            / "derivatives"
            / "masks"
            / "leftHemisphere.nii.gz",
        },
        columns=["age"],
        transforms=TransformsHandler(
            sample_transforms=[ResampleMask(), RandomMasking()],
            augmentations=[
                OneOfConfig(
                    transforms=[
                        RandomAffineConfig(),
                        RandomElasticDeformationConfig(),
                        RandomMotionConfig(),
                        RandomGhostingConfig(),
                        RandomGammaConfig(),
                        RandomSpikeConfig(),
                        RandomBiasFieldConfig(),
                        RandomBlurConfig(),
                        RandomNoiseConfig(),
                    ]
                ),
            ],
        ),
    )

    # model
    with seed_everything_context(seed=0):
        model = SupervisedModel(
            network=CNNConfig(
                in_shape=(1, 16, 20, 17),
                num_outputs=6,
                conv_args={"channels": [4, 8]},
            ),
            loss=BCEWithLogitsLossConfig(),
            optimizer=AdamConfig(),
        )

    # trainer
    optim_config = OptimizationConfig(
        num_epochs=100,
        evaluation_interval=5,
        accumulation_steps=2,
        reset_model=reset_model,
    )
    callbacks = [
        LRSchedulerCallback(
            OneCycleLRConfig(
                max_lr=1e-3, epochs=optim_config.num_epochs, steps_per_epoch=2
            )
        ),
        EarlyStoppingCallback(metric="loss", patience=3, min_delta=0.1),
        LoggerCallback(progress_bar=False, debug=False),
    ]
    if reset_model:
        callbacks.append(TestModelReset(assert_equal=False))
    else:
        callbacks.append(TestModelReset(assert_equal=True))
    if gpu:
        callbacks.append(
            TestDeviceCallback(
                model_on_gpu=True,
                post_processing_on_gpu=True,
                metrics_on_gpu=False,
            )
        )  # no postprocessing so it should stay on GPU

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
        callbacks=callbacks,
        optimization=optim_config,
        overwrite=True,
    )

    return dataset, trainer


def _train(kfold_dir: Path, dataset: Dataset, trainer: Trainer, gpu: bool) -> None:
    kfold = KFold(kfold_dir)

    for split in kfold.get_splits(dataset, splits=[0, 1]):
        split.build_train_loader(sampling_weights="age", batch_size=2, drop_last=True)
        split.build_val_loader()

        if split.index == 1:
            trainer.add_callbacks(
                [
                    ModelCheckpointCallback(metric="f1", epochs=[30]),
                    ErrorCallback(error_epoch=30),
                ]
            )

        try:
            trainer.train(
                split,
                computational=ComputationalConfig(
                    gpu=gpu,
                    amp=True,
                    channels_last=True,
                    seed=split.index,
                    deterministic=True,
                ),
            )
        except torch.cuda.OutOfMemoryError:
            time.sleep(1)
            trainer.resume(split_idx=split.index, split=split)


def _validate(kfold_dir: Path, dataset: Dataset, trainer: Trainer) -> None:
    kfold = KFold(kfold_dir)
    split = next(iter(kfold.get_splits(dataset, splits=[1])))
    split.build_val_loader()

    trainer.add_metrics(recall=ConfusionMatrixMetricConfig(metric_name="recall"))

    trainer.validate(
        split_idx=1,
        dataloader=split.val_loader,
        metrics=["recall"],
        model_checkpoint="best-f1",
    )


def _test(split_dir: Path, dataset: Dataset, trainer: Trainer, gpu: bool):
    test_dataset = dataset.subset(split_dir / "test_baseline.tsv")
    test_loader = DataLoader(test_dataset)

    trainer.test(
        model_checkpoint="split-1_best-f1",
        dataloader=test_loader,
        group_name="oasis",
        computational=ComputationalConfig(
            gpu=gpu, amp=False, channels_last=False, seed=0, deterministic=True
        ),
    )
    time.sleep(1)  # so that the exec files don't have the same name
    trainer.test(
        model_checkpoint="split-1_final",
        dataloader=test_loader,
        group_name="oasis",
        computational=ComputationalConfig(
            gpu=gpu, amp=True, channels_last=True, seed=0, deterministic=True
        ),
    )


def _test_trainer(
    tmp_path: Path,
    ref: Path,
    bids_dir: Path,
    metadata_tsv: Path,
    split_dir: Path,
    kfold_dir: Path,
    gpu: bool,
) -> None:
    from ..utils import compare_maps_dir

    maps_path = tmp_path / "maps"

    dataset, trainer = _setup(
        bids_dir, metadata_tsv, maps_path, reset_model=gpu, gpu=gpu
    )  # test with and without model resetting
    _train(kfold_dir, dataset, trainer, gpu=gpu)
    _validate(kfold_dir, dataset, trainer)
    _test(split_dir, dataset, trainer, gpu=gpu)

    compare_maps_dir(maps_path, ref, except_=[Path("callbacks.json")])


def test_train(tmp_path, ref_data, bids_dir, metadata_tsv, split_dir, kfold_dir):
    ref_maps = ref_data / "maps_test_classification"
    _test_trainer(
        tmp_path, ref_maps, bids_dir, metadata_tsv, split_dir, kfold_dir, gpu=False
    )


@pytest.mark.gpu
def test_train_gpu(tmp_path, ref_data, bids_dir, metadata_tsv, split_dir, kfold_dir):
    ref_maps = ref_data / "maps_test_classification_gpu"
    _test_trainer(
        tmp_path, ref_maps, bids_dir, metadata_tsv, split_dir, kfold_dir, gpu=True
    )
