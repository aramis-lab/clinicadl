from pathlib import Path

import torchio.transforms as transforms

from clinicadl.dataset.dataloader import DataLoaderConfig
from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.datasets.concat import ConcatDataset
from clinicadl.dataset.preprocessing import (
    PreprocessingPET,
    PreprocessingT1,
)
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.losses.config import CrossEntropyLossConfig
from clinicadl.networks.config.resnet import ResNetConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.splitter import KFold, make_kfold,  make_split
from clinicadl.trainer.trainer import Trainer
from clinicadl.transforms.extraction import Extraction, Image, Patch, Slice
from clinicadl.transforms.transforms import Transforms

caps_directory = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps"
)  # output of clinica pipelines

sub_ses_t1 = Path("/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_t1.tsv")
preprocessing_t1 = PreprocessingT1()
transforms_image = Transforms(
    image_augmentation=[transforms.RandomMotion()],
    extraction=Image(),
    image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],
)

print("T1 and image ")

dataset_t1_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_t1,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
)
dataset_t1_image.prepare_data(n_proc=2)  # to extract the tensor of the T1 file


sub_ses_pet_45 = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_pet_18FAV45.tsv"
)
preprocessing_pet_45 = PreprocessingPET(tracer="18FAV45", suvr_reference_region="pons2")  # type: ignore

dataset_pet_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_pet_45,
    preprocessing=preprocessing_pet_45,
    transforms=transforms_image,
)
dataset_t1_image.prepare_data(n_proc=2)  # to extract the tensor of the PET file


dataset_multi_modality = ConcatDataset(
    [
        dataset_t1_image,
        dataset_pet_image,
    ]
)  # 3 train.tsv en entrée qu'il faut concat et pareil pour les transforms à faire attention


# CAS CROSS-VALIDATION

split_dir = make_split(sub_ses_t1, n_test=0.2)  # Optional data tsv and output_dir
fold_dir = make_kfold(split_dir / "train.tsv", n_splits=2)

splitter = KFold(fold_dir)


maps_path = Path("/")
manager = ExperimentManager(maps_path, overwrite=False)

config_file = Path("config_file")
trainer = Trainer.from_json(config_file=config_file, manager=manager)


for split in splitter.get_splits(dataset=dataset_t1_image):
    train_loader = split.build_train_loader(batch_size=2)
    val_loader = split.build_val_loader(DataLoaderConfig())

    model = ClinicaDLModel.from_config(
        network_config=ResNetConfig(num_outputs=1, spatial_dims=1, in_channels=1),
        loss_config=CrossEntropyLossConfig(),
        optimizer_config=AdamConfig(),
    )

    trainer.train(model, split)
    # le trainer va instancier un predictor/valdiator dans le train ou dans le init

# TEST


dataset_test = CapsDataset(
    caps_directory=caps_directory,
    preprocessing=preprocessing_t1,
    data=Path("test.tsv"),  # test only on data from the first dataset
    transforms=transforms_image,
)
