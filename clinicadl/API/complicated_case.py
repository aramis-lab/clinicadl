from pathlib import Path

import torch
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
from clinicadl.metrics.metrics import Metrics
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.config.resnet import ResNet18Config, ResNetConfig
from clinicadl.networks.factory import ImplementedNetwork, get_network_config
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.splitter import KFold, make_kfold, make_split
from clinicadl.trainer.trainer import Trainer
from clinicadl.transforms.extraction import Image, Slice
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils import cluster
from clinicadl.utils.config import ClinicaDLConfig
from clinicadl.utils.seed import seed_everything

### PARAMETERS #####
seed = 3
batch_size: int = 5
epochs: int = 3
lr: float = 0.1
weight_decay: float = 0.0
momentum: float = 0.9
num_workers: int = 0
persistent_workers: bool = True
pin_memory: bool = True

non_blocking: bool = True
prefetch_factor: int = 0
drop_last: bool = True
amp: bool = True
accumulation_steps: int = 1  # gives the number of iterations during which gradients are accumulated before performing the weights update. This allows to virtually increase the size of the batch. Default: 1.
evaluation_steps: int = 5  # gives the number of iterations to perform an evaluation internal to an epoch. Default will only perform an evaluation at the end of each epoch.
current_epoch = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tolerance = 0
patience = 10
num_replica = cluster.size
mini_batch_size = batch_size
global_batch_size = mini_batch_size * num_replica


###########################


dataloader_config = DataLoaderConfig(
    batch_size=mini_batch_size,
    sampling_weights=None,
    shuffle=False,
    drop_last=drop_last,
    num_workers=num_workers,
    prefetch_factor=None,
    pin_memory=pin_memory,
)  #  persistent_workers=self.persistent_workers,


caps_directory = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps"
)  # output of clinica pipelines

# 64 subjects
sub_ses_t1 = Path("/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_t1.tsv")
preprocessing_t1 = T1Linear()
transforms_image = Transforms(
    extraction=Slice(slices=[24, 25, 26, 27, 56, 57, 58, 78, 96, 97]),
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

# dataset_t1_image.read_tensor_conversion(
#     json_name="test_tensors.json", check_transforms=False
# )  # give random name to the json if not given ?


# sub_ses_pet_45 = Path(
#     "/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_pet_18FAV45.tsv"
# )
# preprocessing_pet_45 = PETLinear(tracer="18FAV45", suvr_reference_region="pons2")  # type: ignore

# dataset_pet_image = CapsDataset(
#     caps_directory=caps_directory,
#     data=sub_ses_pet_45,
#     preprocessing=preprocessing_pet_45,
#     transforms=transforms_image,
# )
# dataset_pet_image.to_tensors(
#     json_name="test_pet.json", n_proc=2
# )  # to extract the tensor of the PET file


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

config_file = Path("config_file")
trainer = Trainer(maps_path)
print(maps_path.resolve())


for split in splitter.get_splits(dataset=dataset_t1_image):
    print(f"########SPLIT{split.index}########")
    model = ClinicaDLModel.from_config(
        network_config=get_network_config(
            ImplementedNetwork.RESNET, num_outputs=1, spatial_dims=2, in_channels=1
        ),
        loss_config=MSELossConfig(),
        optimizer_config=AdamConfig(),
    )

    metrics = Metrics(metrics=[MSEMetric(), MAEMetric()])

    split.build_train_loader(dataloader_config)
    split.build_val_loader(dataloader_config)

    trainer.train(
        model,
        split,
        metrics=metrics,
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
