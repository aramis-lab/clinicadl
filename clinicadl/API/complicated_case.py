from pathlib import Path

import torch
import torchio.transforms as transforms
from monai.metrics.regression import MAEMetric, MSEMetric

from clinicadl.data.dataloader.config import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatype.preprocessing import T1Linear
from clinicadl.losses.config import MSELossConfig
from clinicadl.metrics.config.classification import (
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)
from clinicadl.metrics.config.regression import MAEMetricConfig, MSEMetricConfig
from clinicadl.metrics.metrics import Metrics
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.factory import ImplementedNetwork, get_network_config
from clinicadl.optim.config import OptimizationConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter import KFold, make_kfold, make_split
from clinicadl.trainer.trainer import Trainer
from clinicadl.transforms import OutputTransforms, Transforms
from clinicadl.transforms.extraction import Image, Slice
from clinicadl.utils.computational.computational import ComputationalConfig
from clinicadl.utils.seed import seed_everything

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
dataset_t1_image.to_tensors(
    json_name="test.json", n_proc=2
)  # give random name to the json if not given ?


# CAS CROSS-VALIDATION

split_dir = make_split(
    sub_ses_t1, n_test=0.2
)  # Optional data tsv and output_dir DOIT RETOURNER UN SPLIT
fold_dir = make_kfold(split_dir / "train.tsv", n_splits=2)
splitter = KFold(fold_dir)


optim_config = OptimizationConfig()
comput_config = ComputationalConfig(gpu=False)
maps_path = Path("maps_test")

trainer = Trainer(maps_path, optim_config, comput_config)
print(maps_path.resolve())

dataloader_config = DataLoaderConfig(
    batch_size=3,
)

# CROOS VALIDATION LOOP
for split in splitter.get_splits(dataset=dataset_t1_image):
    print(f"Training for split {split.index}")

    # DEFINE MODEL
    model = ClinicaDLModel.from_config(
        network_config=get_network_config(
            ImplementedNetwork.RESNET, num_outputs=1, spatial_dims=2, in_channels=1
        ),
        loss_config=MSELossConfig(),
        optimizer_config=AdamConfig(),
    )
    # DEFINE METRICS
    metrics = Metrics(metrics=[MSEMetric(), MAEMetric()])

    # BUILD DATALOADER
    split.build_train_loader(dataloader_config)
    split.build_val_loader(dataloader_config)

    # TRAIN
    trainer.train(
        model,
        split,
        metrics=metrics,
    )
    # le trainer va instancier un predictor/valdiator dans le train ou dans le init

# TEST

# dataset_test = dataset_t1_image.subset(
#     "/Users/camille.brianceau/aramis/CLINICADL/caps/split/test_baseline.tsv"
# )

dataset_test = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_t1,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
    label="diagnosis",
)

# output_transforms = OutputTransforms(
#     sample_transforms=[transforms.RandomMotion()]
# )

predictor = Predictor(maps_path, comp_config=comput_config, model=model)
predictor.predict(
    dataset=dataset_t1_image,
    split_dir=split_dir,
    data_loader_config=dataloader_config,
    metrics=metrics,
)
# predictor.predict(dataset_tes_2)
