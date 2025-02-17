from pathlib import Path

import torchio.transforms as transforms
from monai.metrics.metric import Metric
from monai.metrics.regression import MAEMetric, MSEMetric

from clinicadl.data.dataloader import DataLoaderConfig
from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datasets.concat import ConcatDataset
from clinicadl.data.datatype.preprocessing import (
    PETLinear,
    T1Linear,
)
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.losses.config import MSELossConfig
from clinicadl.metrics.config.classification import (
    ConfusionMatrixMetricConfig,
    ROCAUCMetricConfig,
)
from clinicadl.metrics.factory import get_metric_from_config
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.config.resnet import ResNet18Config, ResNetConfig
from clinicadl.networks.factory import ImplementedNetwork, get_network_config
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.splitter import KFold, make_kfold, make_split
from clinicadl.trainer.trainer import Trainer
from clinicadl.transforms.extraction import Image
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.config import ClinicaDLConfig

caps_directory = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps"
)  # output of clinica pipelines

# 64 subjects
sub_ses_t1 = Path("/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_t1.tsv")
preprocessing_t1 = T1Linear()
transforms_image = Transforms(
    augmentations=[transforms.RandomMotion()],  # type: ignore
    extraction=Image(),
    image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],  # type: ignore
)

print("T1 and image ")

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


sub_ses_pet_45 = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_pet_18FAV45.tsv"
)
preprocessing_pet_45 = PETLinear(tracer="18FAV45", suvr_reference_region="pons2")  # type: ignore

dataset_pet_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_pet_45,
    preprocessing=preprocessing_pet_45,
    transforms=transforms_image,
)
dataset_pet_image.to_tensors(
    json_name="test_pet.json", n_proc=2
)  # to extract the tensor of the PET file


# dataset_multi_modality = ConcatDataset(
#     [
#         dataset_t1_image,
#         dataset_pet_image,
#     ]
# )  # 3 train.tsv en entrée qu'il faut concat et pareil pour les transforms à faire attention


# CAS CROSS-VALIDATION

split_dir = make_split(sub_ses_t1, n_test=0.2)  # Optional data tsv and output_dir
fold_dir = make_kfold(split_dir / "train.tsv", n_splits=2)

# train : 24
# train baseline : 16
# val baseline : 16
splitter = KFold(fold_dir)


maps_path = Path("maps_test")
manager = ExperimentManager(maps_path, overwrite=False)

config_file = Path("config_file")
trainer = Trainer(maps_path)


# define metrics
train_metric, _ = get_metric_from_config(ConfusionMatrixMetricConfig())
val_metric, _ = get_metric_from_config(ROCAUCMetricConfig())


class Metrics(ClinicaDLConfig):
    train: list[Metric]
    val: list[Metric]


for split in splitter.get_splits(dataset=dataset_t1_image):
    print(f"########SPLIT{split}########")
    model = ClinicaDLModel.from_config(
        network_config=get_network_config(
            ImplementedNetwork.RESNET, num_outputs=1, spatial_dims=2, in_channels=1
        ),
        loss_config=MSELossConfig(),
        optimizer_config=AdamConfig(),
    )

    trainer.train(
        model,
        split,
        metrics=Metrics(train=[MSEMetric()], val=[MSEMetric(), MAEMetric()]),
    )
    # le trainer va instancier un predictor/valdiator dans le train ou dans le init

# TEST

dataset_test = dataset_t1_image.subset(
    "/Users/camille.brianceau/aramis/CLINICADL/caps/split_78/test_baseline.tsv"
)


# dataset_test = CapsDataset(
#     caps_directory=caps_directory,
#     data=sub_ses_t1,
#     preprocessing=preprocessing_t1,
#     transforms=transforms_image,
#     label= "diagnosis",
# )
