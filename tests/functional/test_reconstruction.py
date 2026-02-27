from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from clinicadl.callbacks import MonitorCallback, TrainingCheckpointCallback
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import T1Linear
from clinicadl.infer import PatchesToImageInferer
from clinicadl.losses.config import MSELossConfig
from clinicadl.metrics import MetricsHandler
from clinicadl.metrics.config import LossMetricConfig
from clinicadl.models import ReconstructionModel
from clinicadl.networks.config import AutoEncoderConfig
from clinicadl.optim import OptimizationConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.split import KFold
from clinicadl.train import ComputationalConfig, Trainer
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import ZNormalizationConfig
from clinicadl.transforms.extraction import Patch

if TYPE_CHECKING:
    from clinicadl.split import Split


def _setup(caps_dir: Path, metadata: Path, maps_path: Path) -> None:
    train_dataset = CapsDataset(
        directory=caps_dir,
        datatype=T1Linear(use_uncropped_image=False),
        data=metadata,
        transforms=TransformsHandler(
            image_transforms=[ZNormalizationConfig()],
            extraction=Patch(patch_size=8, overlap=0.5),
        ),
    )
    train_dataset.read_tensor_conversion()

    eval_dataset = CapsDataset(
        directory=caps_dir,
        datatype=T1Linear(use_uncropped_image=False),
        data=metadata,
        transforms=TransformsHandler(
            image_transforms=[ZNormalizationConfig()],
        ),
    )
    eval_dataset.read_tensor_conversion()

    model = ReconstructionModel(
        network=AutoEncoderConfig(
            in_shape=(1, 8, 8, 8), latent_size=16, conv_args={"channels": [4, 8]}
        ),
        loss=MSELossConfig(),
        optimizer=AdamConfig(lr=1e-1),
        inferer=PatchesToImageInferer(
            patch_size=8,
            batch_size=16,
            postprocessing=[ZNormalizationConfig()],
        ),
    )

    trainer = Trainer(
        maps=maps_path,
        model=model,
        metrics=MetricsHandler(
            loss=LossMetricConfig(loss_name="loss", label_key="image"),
            metrics_on_cpu=False,
        ),
        optimization=OptimizationConfig(
            num_epochs=3,
            clip_grad_value=1e-5,
        ),
        callbacks=[
            MonitorCallback(enabled=False),
            TrainingCheckpointCallback(enabled=False),
        ],
        overwrite=True,
    )

    return train_dataset, eval_dataset, trainer


def _train(
    split: Split,
    trainer: Trainer,
    gpu: bool,
) -> None:
    split.build_train_loader(batch_size=16)
    split.build_val_loader(batch_size=16)

    trainer.train(
        split,
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
    kfold_dir: Path,
    gpu: bool,
) -> None:
    from ..utils import compare_maps_dir

    maps_path = tmp_path / "maps"

    train_dataset, eval_dataset, trainer = _setup(caps_dir, metadata_tsv, maps_path)
    kfold = KFold(kfold_dir)
    split = next(
        iter(kfold.get_splits(train_dataset, eval_dataset=eval_dataset, splits=[0]))
    )
    _train(split, trainer, gpu=gpu)
    trainer = Trainer.from_maps(maps_path)
    split = next(
        iter(kfold.get_splits(train_dataset, eval_dataset=eval_dataset, splits=[1]))
    )
    _train(split, trainer, gpu=gpu)

    compare_maps_dir(maps_path, ref)


def test_train(tmp_path, ref_data, caps_dir, metadata_tsv, kfold_dir):
    ref_maps = ref_data / "maps_test_reconstruction"
    _test_trainer(tmp_path, ref_maps, caps_dir, metadata_tsv, kfold_dir, gpu=False)


@pytest.mark.gpu
def test_train_gpu(tmp_path, ref_data, caps_dir, metadata_tsv, kfold_dir):
    ref_maps = ref_data / "maps_test_reconstruction_gpu"
    _test_trainer(tmp_path, ref_maps, caps_dir, metadata_tsv, kfold_dir, gpu=True)
