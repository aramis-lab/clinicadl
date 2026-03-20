"""
A hybrid task with both regression and classification, trained on a single split:
- an PairedDataset;
- a custom model;
- two optimizers;
- two parameter groups in one optimizer;
- two losses;
- lr scheduling on both optimizers;
- postprocessing (on GPU);
- metrics (on GPU);
- monitor and checkpoint callbacks disabled.

The data location (i.e. the device) across the workflow is tested.

An error is raised to interrupt the training.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd
import pytest
import torch
import torchio as tio
from torch.nn import BCEWithLogitsLoss, MSELoss

from clinicadl.callbacks import (
    Callback,
    LRSchedulerCallback,
    ModelCheckpointCallback,
    TrainingCheckpointCallback,
)
from clinicadl.data.datasets import CapsDataset, PairedDataset
from clinicadl.data.datatypes import T1Linear
from clinicadl.infer import SimpleInferer
from clinicadl.io import Maps
from clinicadl.metrics import MetricsHandler
from clinicadl.metrics.config import LossMetricConfig, MAEMetricConfig
from clinicadl.models import Model
from clinicadl.networks.nn import CNN, MLP
from clinicadl.optim import OptimizationConfig
from clinicadl.optim.lr_schedulers.config import StepLRConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.split import SingleSplit
from clinicadl.train import ComputationalConfig, Trainer
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import (
    FormatConfig,
    MergeFieldsConfig,
    ZNormalizationConfig,
)

from .utils import ErrorCallback, TestDeviceCallback

if TYPE_CHECKING:
    from clinicadl.data.dataloader import Batch
    from clinicadl.data.datasets import Dataset


class _TwoHeadMLP(torch.nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.head_0 = MLP(num_inputs=in_features, num_outputs=1, hidden_dims=[])
        self.head_1 = MLP(num_inputs=in_features, num_outputs=1, hidden_dims=[])

    def forward(self, x):
        out1 = self.head_0(x)
        out2 = self.head_1(x)
        return torch.concat([out1, out2], dim=1)


class PostProcessTransform(tio.transforms.IntensityTransform):
    def apply_transform(self, subject: tio.Subject):
        subject["output"][0] = torch.nn.functional.relu(subject["output"][0])
        subject["output"][1] = torch.nn.functional.sigmoid(subject["output"][1])
        return subject


class TwoHeadsRegressionModel(Model):
    def __init__(
        self,
        base_model: Path,
        lr_mlp: float,
        lr_age: float,
        lr_sex: float,
        lambd: float = 1.0,
    ):
        super().__init__()
        self._build_model(base_model)
        self.loss_age = MSELoss()
        self.loss_sex = BCEWithLogitsLoss()
        self.lambd = lambd
        self.lr_mlp = lr_mlp
        self.lr_age = lr_age
        self.lr_sex = lr_sex
        self.inferer = SimpleInferer(
            postprocessing=[
                PostProcessTransform(),
            ],
            postprocessing_on_cpu=False,
        )

    def _build_model(self, base_model: Path):
        self.network = CNN(
            in_shape=(1, 16, 20, 17),
            num_outputs=6,
            conv_args={"channels": [4, 8]},
        )
        self.load_state_dict(torch.load(base_model))
        self.network = torch.nn.Sequential(
            OrderedDict(
                [("backbone", self.network), ("head", _TwoHeadMLP(in_features=6))]
            )
        )

    def build_optimizers(self):
        return {
            "optimizer_backbone": AdamConfig(
                freeze="convolutions",
                lr=self.lr_mlp,
            ).get_object(self.network.backbone),
            "optimizer_head": AdamConfig(
                lr={"head_0": self.lr_age, "ELSE": self.lr_sex},
            ).get_object(self.network.head),
        }

    def get_loss_functions(self):
        return {"loss_age": self.loss_age, "loss_sex": self.loss_sex}

    def forward_step(self, batch: Sequence[Batch]):
        image = self._merge_images(batch)
        labels = batch[0].get_field(
            "label", ensure_channel_dim=True, dtype=torch.float32
        )
        age, sex = labels[:, 0], labels[:, 1]

        out = self.network(image)
        pred_age, pred_sex = out[:, 0], out[:, 1]

        return {
            "loss_age": self.loss_age(pred_age, age),
            "loss_sex": self.loss_sex(pred_sex, sex),
        }

    def evaluation_step(self, batch: Sequence[Batch]):
        out = self.inferer(batch[0], self.network, input_dtype=torch.float32)

        out.add_field(out.get_field("output")[:, 0], "output_age")
        out.add_field(out.get_field("output")[:, 1], "output_sex")

        return out

    def backward_step(
        self,
        loss: dict[str, torch.Tensor],
        grad_scaler: torch.amp.GradScaler = torch.amp.GradScaler(enabled=False),
    ) -> None:
        total_loss = loss["loss_age"] + self.lambd * loss["loss_sex"]
        total_loss.backward()

    def optimization_step(
        self,
        optimizers: dict[str, torch.optim.Optimizer],
        grad_scaler: torch.amp.GradScaler = torch.amp.GradScaler(enabled=False),
    ) -> None:
        optimizers["optimizer_backbone"].step()
        optimizers["optimizer_head"].step()

    def prediction_step(self, batch: Batch) -> Batch:
        return self.evaluation_step(batch)

    @staticmethod
    def _merge_images(batch: tuple[Batch, Batch]) -> torch.Tensor:
        return (
            batch[0].get_field("image", dtype=torch.float32)
            + batch[1].get_field("image", dtype=torch.float32)
        ) / 2  # the same image. Just for the test


def _encode_sex(gender: pd.Series) -> pd.Series:
    def _encode(gender: str) -> int:
        if gender == "M":
            return 0
        return 1

    return gender.apply(_encode).astype(int)


def build_callbacks() -> list[Callback]:
    return [
        TrainingCheckpointCallback(every_n_epochs=5),
        LRSchedulerCallback(
            scheduler=StepLRConfig(step_size=3, gamma=0.1),
            optimizer_name="optimizer_backbone",
        ),
        LRSchedulerCallback(
            scheduler=StepLRConfig(step_size=3, gamma=10),
            optimizer_name="optimizer_head",
        ),
        ModelCheckpointCallback(metric="bce"),
    ]


def _setup(
    caps_dir: Path, metadata: Path, maps_path: Path, base_model_dir: Path, gpu: bool
) -> tuple[Dataset, Trainer]:
    data = pd.read_csv(metadata, sep="\t")
    data["sex"] = _encode_sex(data["sex"])
    dataset = CapsDataset(
        directory=caps_dir,
        datatype=T1Linear(use_uncropped_image=False),
        data=data,
        columns=["age", "sex"],
        transforms=TransformsHandler(
            image_transforms=[
                MergeFieldsConfig(keys=["age", "sex"], output_key="label"),
                ZNormalizationConfig(),
            ],
        ),
    )
    dataset.read_tensor_conversion()

    paired_dataset = PairedDataset([dataset, dataset])

    model = TwoHeadsRegressionModel(
        base_model=base_model_dir,
        lr_mlp=0.0001,
        lr_age=0.0001,
        lr_sex=0.001,
        lambd=100,
    )
    metrics = MetricsHandler(
        bce=LossMetricConfig(
            loss_name="loss_age", label_key="age", pred_key="output_age"
        ),
        mae=MAEMetricConfig(label_key="age", pred_key="output_age"),
        mse=LossMetricConfig(
            loss_name="loss_sex",
            label_key="sex",
            pred_key="output_sex",
            postprocessing=[FormatConfig(include=["sex"], dtype=torch.float)],
        ),
        metrics_on_cpu=False,
    )
    callbacks = build_callbacks()
    callbacks.append(ErrorCallback(error_epoch=7))
    if gpu:
        callbacks.append(
            TestDeviceCallback(
                model_on_gpu=True, post_processing_on_gpu=True, metrics_on_gpu=True
            )
        )

    trainer = Trainer(
        maps=maps_path,
        model=model,
        metrics=metrics,
        optimization=OptimizationConfig(
            num_epochs=10,
        ),
        callbacks=callbacks,
        overwrite=True,
    )

    return paired_dataset, trainer


def _train(
    split_dir: Path,
    dataset: Dataset,
    trainer: Trainer,
    gpu: bool,
) -> None:
    splitter = SingleSplit(split_dir)
    split = splitter.get_split(dataset)
    split.build_train_loader(batch_size=2)
    split.build_val_loader()

    try:
        trainer.train(
            split,
            computational=ComputationalConfig(
                gpu=gpu,
                amp=False,
                seed=0,
                deterministic=True,
            ),
            metrics=["bce", "mse"],
        )
    except torch.cuda.OutOfMemoryError:
        pass


def _test_trainer(
    tmp_path: Path,
    ref: Path,
    base_model: Path,
    caps_dir: Path,
    metadata_tsv: Path,
    split_dir: Path,
    gpu: bool,
) -> None:
    from ..utils import compare_maps_dir

    maps_path = tmp_path / "maps"

    dataset, trainer = _setup(caps_dir, metadata_tsv, maps_path, base_model, gpu=gpu)
    _train(split_dir, dataset, trainer, gpu=gpu)

    except_ = [
        "callbacks.json",
        "training/split-0/tmp/epoch-5/callbacks/monitor_callback.pt",
    ]
    if gpu:
        except_.append("training/split-0/tmp/epoch-5/callbacks")

    compare_maps_dir(
        maps_path,
        ref,
        except_=except_,
    )


def test_train(tmp_path, ref_data, caps_dir, metadata_tsv, split_dir):
    ref = ref_data / "maps_test_regression_interrupted"
    maps_classif = Maps(ref_data / "maps_test_classification")
    maps_classif.read()
    base_model = maps_classif.training.splits[0].models.final.model_pt
    _test_trainer(
        tmp_path,
        ref,
        base_model,
        caps_dir,
        metadata_tsv,
        split_dir,
        gpu=False,
    )


@pytest.mark.gpu
def test_train_gpu(tmp_path, ref_data, caps_dir, metadata_tsv, split_dir):
    ref = ref_data / "maps_test_regression_interrupted_gpu"
    maps_classif = Maps(ref_data / "maps_test_classification")
    maps_classif.read()
    base_model = maps_classif.training.splits[0].models.final.model_pt
    _test_trainer(
        tmp_path,
        ref,
        base_model,
        caps_dir,
        metadata_tsv,
        split_dir,
        gpu=True,
    )
