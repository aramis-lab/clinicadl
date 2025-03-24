from pathlib import Path

import torchio.transforms as transforms

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datasets.concat import ConcatDataset
from clinicadl.data.datatypes.preprocessing import (
    PETLinear,
    T1Linear,
)
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.losses.config import CrossEntropyLossConfig
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.factory import ImplementedNetwork, get_network_config
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter import KFold, make_kfold, make_split
from clinicadl.trainer.trainer import Trainer
from clinicadl.transforms import Transforms
from clinicadl.transforms.extraction import Slice
from clinicadl.transforms.output_transforms import OutputTransforms
from clinicadl.utils.computational.computational import ComputationalConfig

caps_directory = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps"
)  # output of clinica pipelines
sub_ses_t1 = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_t1.tsv"
)  # 64 subjects

# DEFINE PREPROCESSING
preprocessing_t1 = T1Linear()

# DEFINE TRANSFORMS
transforms_image = Transforms(
    extraction=Slice(slices=[24, 25, 26, 27, 56, 57, 58, 78, 96, 97]),
)

# CREATE CAPSDATASET
dataset_t1_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_t1,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
    label="diagnosis",
)
dataset_t1_image.to_tensors(json_name="test.json", n_proc=2)


# CAS CROSS-VALIDATION

split_dir = make_split(sub_ses_t1, n_test=0.2)
fold_dir = make_kfold(split_dir / "train.tsv", n_splits=2)
splitter = KFold(fold_dir)


optim_config = OptimizationConfig(epochs=4)
comput_config = ComputationalConfig(gpu=False)
dataloader_config = DataLoaderConfig(batch_size=3)


maps_path = Path("maps_test")

# DEFINE MODEL
model = ClinicaDLModel.from_config(
    network_config=get_network_config(
        ImplementedNetwork.RESNET, num_outputs=1, spatial_dims=2, in_channels=1
    ),
    loss_config=MSELossConfig(),
    optimizer_config=AdamConfig(),
)


# DEFINE METRICS
metrics = Metrics(
    metrics=[MSEMetricConfig(), MAEMetricConfig()],
    selection_metrics=[MSEMetricConfig(), "Loss"],
)


trainer = Trainer(
    maps_path,
    model=model,
    comp_config=comput_config,
    optim_config=optim_config,
    metrics=metrics,
)


# CROOS VALIDATION LOOP
for split in splitter.get_splits(dataset=dataset_t1_image):
    print(f"Training for split {split.index}")

    # BUILD DATALOADER
    split.build_train_loader(dataloader_config)
    split.build_val_loader(dataloader_config)

    # TRAIN
    trainer.train(split)


# TEST

dataset_test = CapsDataset(
    caps_directory=caps_directory,
    data=split_dir / "test_baseline.tsv",
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
    label="diagnosis",
)
dataset_test.to_tensors(json_name="test_test.json", n_proc=2)


output_transforms = OutputTransforms(sample_transforms=[transforms.RandomMotion()])  # type: ignore

predictor = Predictor(maps_path, comp_config=comput_config, model=model)

dataloader = dataloader_config.get_dataloader(dataset_test)

predictor.predict(dataloader, metrics=metrics, split=1, data_group="test")
