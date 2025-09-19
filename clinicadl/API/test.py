from pathlib import Path

import pandas as pd
import torchio.transforms as transforms
from monai.metrics.regression import MAEMetric

from clinicadl.callbacks import (
    Checkpoint,
    CodeCarbon,
    EarlyStopping,
    LRScheduler,
    ModelSelection,
    Tensorboard,
)
from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datatypes.preprocessing import T1Linear
from clinicadl.losses.config import MSELossConfig
from clinicadl.metrics.config.factory import (
    ConfusionMatrixMetricConfig,
    MAEMetricConfig,
    MSEMetricConfig,
    SSIMMetricConfig,
)
from clinicadl.models.base import ClinicaDLModel
from clinicadl.networks.config import ImplementedNetwork, get_network_config
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.split import KFold, make_kfold, make_split
from clinicadl.train.trainer import Trainer
from clinicadl.transforms.config.factory import (
    ActivationsConfig,
    ClampConfig,
    RandomAffineConfig,
    RescaleIntensityConfig,
    ResizeConfig,
)
from clinicadl.transforms.extraction import Slice
from clinicadl.transforms.handlers import Postprocessing, Transforms
from clinicadl.utils.computational.config import ComputationalConfig

caps_directory = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps"
)  # output of clinica pipelines
sub_ses_t1 = caps_directory / "subjects_t1.tsv"  # 64 subjects

preprocessing_t1 = T1Linear()

transforms_image = Transforms(
    extraction=Slice(slices=[24, 25, 26, 27, 56, 57, 58, 78, 96, 97]),
    image_transforms=[RescaleIntensityConfig(), ResizeConfig(target_shape=(128, 128))],
    sample_transforms=[ClampConfig(min=-1000, max=1000), RandomAffineConfig()],
)
dataset_t1_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_t1,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
    label="diagnosis",
    columns={"diagnosis": None},
)
dataset_t1_image.to_tensors(conversion_name="test_bis_im", n_proc=2)

split_dir = make_split(sub_ses_t1, n_test=0.2)
fold_dir = make_kfold(split_dir / "train.tsv", n_splits=2)
splitter = KFold(fold_dir)


optim_config = OptimizationConfig(epochs=3)
comput_config = ComputationalConfig(gpu=False)
dataloader_config = DataLoaderConfig(batch_size=1)

maps_path = Path("maps_test")
loss = MSELossConfig()

model = ClinicaDLModel(
    network=get_network_config(
        ImplementedNetwork.RESNET, num_outputs=1, spatial_dims=3, in_channels=1
    ),
    loss=loss,
    optimizer=AdamConfig(),
)


mae = MAEMetricConfig()
mse = MSEMetricConfig()
# matrix = ConfusionMatrixMetricConfig(metric_name=ConfusionMatrixMetricCon)

callbacks = [
    EarlyStopping(metrics=["mae", "loss"]),
    ModelSelection(metrics=["mae"]),
    EarlyStopping(metrics=["mse"]),
    Checkpoint(patience=2, epochs=[3]),
    Tensorboard(),
    LRScheduler(scheduler="LinearLR"),
    # CodeCarbon(),
]

trainer = Trainer(
    maps_path,
    model=model,
    comp_config=comput_config,
    optim_config=optim_config,
    callbacks=callbacks,
    metrics={"mae": mae, "mse": mse},
    _overwrite=True,
)


# CROSS VALIDATION LOOP
for split in splitter.get_splits(dataset=dataset_t1_image):
    # BUILD DATALOADER
    split.build_train_loader(dataloader_config)
    split.build_val_loader(dataloader_config)

    # TRAIN
    trainer.train(split)


trainer.evaluate(split.val_loader, additional_metrics=[mae])
print("out of training")
# TEST

dataset_test = CapsDataset(
    caps_directory=caps_directory,
    data=split_dir / "test_baseline.tsv",
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
    label="diagnosis",
)
dataset_test.to_tensors(conversion_name="test_bis_im_bis", n_proc=2)
dataloader_test = dataloader_config.get_object(dataset_test)

output_transforms = Postprocessing(sample_transforms=[transforms.RandomMotion()])  # type: ignore
add_metrics = []


trainer.predict(
    dataloader_test,
    additional_metrics=add_metrics,
    split=1,
    data_group="test",
    output_transforms=output_transforms,
)
