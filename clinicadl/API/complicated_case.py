from pathlib import Path

import torchio.transforms as transforms

from clinicadl.dataset.dataloader_config import DataLoaderConfig
from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.datasets.concat import ConcatDataset
from clinicadl.dataset.preprocessing import (
    PreprocessingCustom,
    PreprocessingPET,
    PreprocessingT1,
)
from clinicadl.dataset.readers.caps_reader import CapsReader
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.losses.config import CrossEntropyLossConfig
from clinicadl.losses.factory import get_loss_function
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.config import ImplementedNetworks
from clinicadl.networks.factory import (
    ConvEncoderOptions,
    create_network_config,
    get_network_from_config,
)
from clinicadl.optimization.optimizer.config import AdamConfig, OptimizerConfig
from clinicadl.optimization.optimizer.factory import get_optimizer
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.kfold import KFolder
from clinicadl.splitter.split import get_single_split, split_tsv
from clinicadl.trainer.trainer import Trainer
from clinicadl.transforms.extraction import ROI, BaseExtraction, Image, Patch, Slice
from clinicadl.transforms.transforms import Transforms

# Create the Maps Manager / Read/write manager /
maps_path = Path("/")
manager = ExperimentManager(
    maps_path, overwrite=False
)  # a ajouter dans le manager: mlflow/ profiler/ etc ...

caps_directory = Path("caps_directory")  # output of clinica pipelines

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
preprocessing_pet_45 = PreprocessingPET(tracer="18FAV45", suvr_reference_region="pons2")

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


config_file = Path("config_file")
trainer = Trainer.from_json(config_file=config_file, manager=manager)

# CAS CROSS-VALIDATION
splitter = KFolder(caps_dataset=dataset_multi_modality_multi_extract, manager=manager)
split_dir = splitter.make_splits(
    n_splits=3, output_dir=Path(""), subset_name="validation", stratification=""
)  # Optional data tsv and output_dir

dataloader_config = DataLoaderConfig(n_procs=3, batch_size=10)


# CAS 1

# Prérequis : déjà avoir des fichiers avec les listes train et validation
split_dir = make_kfold(
    "dataset.tsv"
)  # lit dataset.tsv => fait le kfold => ecrit la sortie dans split_dir
splitter = KFolder(
    dataset_multi_modality, split_dir
)  # c'est plutôt un iterable de dataloader

# CAS 2
splitter = KFolder(caps_dataset=dataset_t1_image)
splitter.make_splits(n_splits=3)
splitter.write(split_dir)

# or
splitter = KFolder(caps_dataset=dataset_t1_image)
splitter.read(split_dir)

for split in splitter.get_splits(splits=(0, 3, 4), dataloader_config=dataloader_config):
    # bien définir ce qu'il y a dans l'objet split

    network_config = create_network_config(ImplementedNetworks.CNN)(
        in_shape=[2, 2, 2],
        num_outputs=1,
        conv_args=ConvEncoderOptions(channels=[3, 2, 2]),
    )
    model = ClinicaDLModelClassif.from_config(
        network_config=network_config,
        loss_config=CrossEntropyLossConfig(),
        optimizer_config=AdamConfig(),
    )

    trainer.train(model, split)
    # le trainer va instancier un predictor/valdiator dans le train ou dans le init

# TEST


dataset_test = CapsDataset(
    caps_directory=caps_directory,
    preprocessing=preprocessing_t1,
    sub_ses_tsv=Path("test.tsv"),  # test only on data from the first dataset
    transforms=transforms_image,
)

predictor = Predictor(model=model, manager=manager)
predictor.predict(dataset_test=dataset_test, split_number=2)
