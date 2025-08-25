from pathlib import Path

import pandas as pd
from monai.metrics.regression import MAEMetric

from clinicadl.callbacks import (
    Checkpoint,
    CodeCarbon,
    EarlyStopping,
    LRScheduler,
    ModelSelection,
    Tensorboard,
)
from clinicadl.callbacks.training_state import _TrainingState
from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear
from clinicadl.IO.maps.maps import Maps
from clinicadl.losses.config import MSELossConfig
from clinicadl.metrics.config import (
    ConfusionMatrixMetricConfig,
    LossMetricConfig,
    MAEMetricConfig,
    MSEMetricConfig,
)
from clinicadl.metrics.handler import MetricsHandler
from clinicadl.metrics.monai_wrapper import MonaiMetricWrapper
from clinicadl.model import ClinicaDLModel
from clinicadl.model.example_model import example_model
from clinicadl.networks.config import ResNetConfig
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.split.split import Split
from clinicadl.utils.computational.config import ComputationalConfig

BIDS_DIR = Path(__file__).parents[1] / "resources" / "bids_example"
CAPS_DIR = Path(__file__).parents[1] / "resources" / "caps_example"
MAPS_DIR = Path(__file__).parents[1] / "resources" / "maps_example"
SPLIT_DIR = (
    Path(__file__).parents[1] / "resources" / "caps_example" / "splits" / "split"
)

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
    data=DATA.iloc[6:7],
)

TEST_DATASET = CapsDataset(
    CAPS_DIR,
    preprocessing=PETLinear(
        tracer="18FAV45",
        suvr_reference_region="pons2",
        use_uncropped_image=True,
    ),
    data=DATA.iloc[7:],
)

MAPS = Maps(MAPS_DIR)

OPTIM = OptimizationConfig(epochs=3)
COMP = ComputationalConfig(gpu=False)
DATALOADER = DataLoaderConfig(batch_size=1)


NETWORK = ResNetConfig(
    spatial_dims=3,
    in_channels=1,
    num_outputs=1,
)

LOSS = MSELossConfig()
OPTIMIZER = AdamConfig()

MODEL = ClinicaDLModel(
    network=NETWORK,
    loss=LOSS,
    optimizer=OPTIMIZER,
)

METRICS_HANDLER = MetricsHandler(
    loss=LossMetricConfig(loss_fn=LOSS.get_object()),
    **{"mae": MAEMetricConfig(), "mse": MSEMetricConfig()},
)

SPLIT = Split(
    index=1, split_dir=SPLIT_DIR, train_dataset=TRAIN_DATASET, val_dataset=VAL_DATASET
)

mae = MonaiMetricWrapper(
    MAEMetric(), optimum="min", pred_key="output", label_key="label"
)
mse = MSEMetricConfig()
matrix = ConfusionMatrixMetricConfig(metric_name="tpr")

METRICS = {"mae": mae, "mse": mse, "matrix": matrix}
CALLBACKS = [
    EarlyStopping(metrics=["mae", "loss"]),
    ModelSelection(metrics=["mae"]),
    EarlyStopping(metrics=["mse"]),
    Checkpoint(patience=2, epochs=[3]),
    Tensorboard(),
    LRScheduler(scheduler="LinearLR"),
    # CodeCarbon(),
]

TRAINING_STATE = _TrainingState(
    maps=MAPS,
    metrics=METRICS_HANDLER,
    model=MODEL,
    optim=OPTIM,
    comp=COMP,
)
SPLIT.build_train_loader(dataloader_config=DATALOADER)
SPLIT.build_val_loader(dataloader_config=DATALOADER)
TRAINING_STATE.reset(SPLIT)
