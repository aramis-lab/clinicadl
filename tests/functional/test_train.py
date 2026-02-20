from pathlib import Path

from clinicadl.callbacks import (
    EarlyStoppingCallback,
    LoggerCallback,
    LRSchedulerCallback,
    ModelCheckpointCallback,
)
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

CAPS_DIR = Path("/Users/thibault.devarax/Desktop/code/clinicadl_data_ci/new_data/caps")
REF_MAPS_DIR = Path(
    "/Users/thibault.devarax/Desktop/code/clinicadl_data_ci/new_data/maps"
)
SPLIT_DIR = CAPS_DIR / "splits" / "split"
KFOLD_DIR = SPLIT_DIR / "4_fold"


def _train(maps_path: Path):
    # dataset
    dataset = build_dataset(CAPS_DIR)

    # model
    model = SupervisedModel(
        network=CNNConfig(
            in_shape=(1, 16, 20, 17),
            num_outputs=6,
            conv_args={"channels": [1, 1], "pooling_indices": [0, 1]},
        ),
        loss=BCEWithLogitsLossConfig(),
        optimizer=AdamConfig(),
    )

    # trainer
    optim_config = OptimizationConfig(
        num_epochs=20, evaluation_interval=10, accumulation_steps=2
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
            EarlyStoppingCallback(metric="loss", patience=50, min_delta=0.1),
            ModelCheckpointCallback(metric="f1"),
            LoggerCallback(progress_bar=False, debug=False),
        ],
        optimization=optim_config,
        overwrite=True,
    )

    # training
    kfold = KFold(KFOLD_DIR)

    for split in kfold.get_splits(dataset, splits=[0, 1]):
        split.build_train_loader(sampling_weights="age", batch_size=2, drop_last=True)
        split.build_val_loader()

        trainer.train(
            split,
            computational=ComputationalConfig(
                gpu=False, amp=True, channels_last=True, seed=0, deterministic=True
            ),
        )


def test_train(tmp_path):
    from ..utils import compare_maps_dir

    maps_path = tmp_path / "maps"

    _train(maps_path=maps_path)

    compare_maps_dir(maps_path, REF_MAPS_DIR)
