"""
A segmentation on 2D slices trained on 2 splits (KFold splitting) with:
- a training dataset with slice extraction and data augmentation;
- an evaluation dataset with whole image;
- a Supervised Model with a custom neural network, a custom loss,
  and slices to images inferer with postprocessing (on GPU);
- model checkpointing (with a list of epochs);
- metrics with another key than the default 'output';
- gradient norm clipping;
- validation on new metrics;
- test (two times on the same group).

The data location (i.e. the device) across the workflow is tested.
For the evaluation phases, the trainer is rebuilt from the MAPS
(with custom objects).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from monai.losses import DiceLoss

from clinicadl.callbacks import ModelCheckpointCallback
from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import T1Linear
from clinicadl.infer import SlicesToImageInferer
from clinicadl.metrics.config import (
    HausdorffDistanceMetricConfig,
    MeanIoUConfig,
)
from clinicadl.models import SupervisedModel
from clinicadl.networks.nn import UNet
from clinicadl.optim import OptimizationConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.split import KFold
from clinicadl.train import ComputationalConfig, Trainer
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import (
    ActivationsConfig,
    AsDiscreteConfig,
    CropOrPadConfig,
    OneOfConfig,
    RandomBiasFieldConfig,
    RandomBlurConfig,
    RandomGammaConfig,
    RandomGhostingConfig,
    RandomMotionConfig,
    RandomNoiseConfig,
    RandomSpikeConfig,
)
from clinicadl.transforms.extraction import Slice

from .utils import TestDevice

if TYPE_CHECKING:
    from clinicadl.callbacks import Callback
    from clinicadl.data.datasets import Dataset
    from clinicadl.models import Model


def _buid_model() -> Model:
    return SupervisedModel(
        network=UNet(spatial_dims=2, in_channels=1, out_channels=1, channels=(4, 8)),
        loss=DiceLoss(sigmoid=True),
        optimizer=AdamConfig(lr=1e-2),
        inferer=SlicesToImageInferer(
            slice_direction=0,
            batch_size=16,
            output_name="seg",
            output_type="mask",
            postprocessing=[
                ActivationsConfig(sigmoid=True, include=["seg"]),
                AsDiscreteConfig(threshold=0.5, include=["seg"]),
            ],
            postprocessing_on_cpu=False,
        ),
    )


def _build_callbacks(gpu: bool) -> list[Callback]:
    callbacks = [ModelCheckpointCallback(epochs=[4, 8], save_last=True)]
    if gpu:
        callbacks.append(
            TestDevice(
                model_on_gpu=True, post_processing_on_gpu=True, metrics_on_gpu=False
            )
        )

    return callbacks


def _setup(caps_dir: Path, metadata: Path, maps_path: Path, gpu: bool) -> None:
    train_dataset = CapsDataset(
        directory=caps_dir,
        datatype=T1Linear(use_uncropped_image=False),
        data=metadata,
        label="head",
        masks=["head"],
        transforms=TransformsHandler(
            extraction=Slice(slice_direction=0, squeeze=True),
            sample_transforms=[CropOrPadConfig(target_shape=(1, 16, 16))],
            augmentations=[
                OneOfConfig(
                    transforms=[
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
    train_dataset.read_tensor_conversion()

    eval_dataset = CapsDataset(
        directory=caps_dir,
        datatype=T1Linear(use_uncropped_image=False),
        data=metadata,
        label="head",
        masks=["head"],
        transforms=TransformsHandler(
            sample_transforms=[CropOrPadConfig(target_shape=16)],
        ),
    )
    eval_dataset.read_tensor_conversion()

    trainer = Trainer(
        maps=maps_path,
        model=_buid_model(),
        metrics={
            "IoU": MeanIoUConfig(
                pred_key="seg",
            ),
        },
        optimization=OptimizationConfig(
            num_epochs=10,
            clip_grad_norm=1,
        ),
        callbacks=_build_callbacks(gpu),
        overwrite=True,
    )

    return train_dataset, eval_dataset, trainer


def _train(
    kfold_dir: Path,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    trainer: Trainer,
    gpu: bool,
) -> None:
    kfold = KFold(kfold_dir)

    for split in kfold.get_splits(train_dataset, eval_dataset=eval_dataset, splits=[0]):
        split.build_train_loader(batch_size=8)
        split.build_val_loader(batch_size=2)

        trainer.train(
            split,
            computational=ComputationalConfig(
                gpu=gpu,
                seed=0,
                deterministic=True,
            ),
        )


def _restore_trainer(maps_path: Path, gpu: bool) -> Trainer:
    if not gpu:
        return Trainer.from_maps(maps_path, model=_buid_model())
    else:
        return Trainer.from_maps(
            maps_path,
            model=_buid_model(),
            callbacks=_build_callbacks(gpu),
        )


def _validate(maps_path: Path, gpu: bool) -> None:
    trainer = _restore_trainer(maps_path, gpu)

    trainer.add_metrics(hd=HausdorffDistanceMetricConfig(pred_key="seg"))

    trainer.validate(
        split_idx=0,
        metrics=["hd"],
        computational=ComputationalConfig(
            gpu=gpu,
            seed=0,
            deterministic=True,
        ),
    )


def _test(
    maps_path: Path, split_dir: Path, eval_dataset: Dataset, trainer: Trainer, gpu: bool
):
    trainer = _restore_trainer(maps_path, gpu)

    test_dataset = eval_dataset.subset(split_dir / "test_baseline.tsv")
    test_loader = DataLoaderConfig().get_object(test_dataset)

    trainer.test(
        model_checkpoint="split-0_final",
        dataloader=test_loader,
        group_name="oasis",
        computational=ComputationalConfig(
            gpu=gpu,
            seed=0,
            deterministic=True,
        ),
    )
    time.sleep(1)
    trainer.test(
        model_checkpoint="split-0_epoch-4",
        group_name="oasis",
        computational=ComputationalConfig(
            gpu=gpu,
            seed=0,
            deterministic=True,
        ),
    )


def _test_trainer(
    tmp_path: Path,
    ref: Path,
    caps_dir: Path,
    metadata_tsv: Path,
    split_dir: Path,
    kfold_dir: Path,
    gpu: bool,
) -> None:
    from ..utils import compare_maps_dir

    maps_path = tmp_path / "maps"

    train_dataset, eval_dataset, trainer = _setup(
        caps_dir, metadata_tsv, maps_path, gpu=gpu
    )
    _train(kfold_dir, train_dataset, eval_dataset, trainer, gpu=gpu)
    _validate(maps_path, gpu=gpu)
    _test(maps_path, split_dir, eval_dataset, trainer, gpu=gpu)

    compare_maps_dir(maps_path, ref)


def test_train(tmp_path, ref_data, caps_dir, metadata_tsv, split_dir, kfold_dir):
    ref_maps = ref_data / "maps_test_segmentation"
    _test_trainer(
        tmp_path, ref_maps, caps_dir, metadata_tsv, split_dir, kfold_dir, gpu=False
    )


@pytest.mark.gpu
def test_train_gpu(tmp_path, ref_data, caps_dir, metadata_tsv, split_dir, kfold_dir):
    ref_maps = ref_data / "maps_test_segmentation_gpu"
    _test_trainer(
        tmp_path, ref_maps, caps_dir, metadata_tsv, split_dir, kfold_dir, gpu=True
    )
