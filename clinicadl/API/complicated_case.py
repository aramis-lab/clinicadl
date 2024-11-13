from pathlib import Path

import torchio.transforms as transforms

from clinicadl.dataset.caps_dataset import (
    CapsDatasetPatch,
    CapsDatasetRoi,
    CapsDatasetSlice,
)
from clinicadl.dataset.caps_reader import CapsReader
from clinicadl.dataset.concat import ConcatDataset
from clinicadl.dataset.config.extraction import ExtractionConfig
from clinicadl.dataset.config.preprocessing import (
    PreprocessingConfig,
    T1PreprocessingConfig,
)
from clinicadl.dataset.dataloader_config import DataLoaderConfig
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
from clinicadl.transforms.transforms import Transforms

# Create the Maps Manager / Read/write manager /
maps_path = Path("/")
manager = ExperimentManager(
    maps_path, overwrite=False
)  # a ajouter dans le manager: mlflow/ profiler/ etc ...

caps_directory = Path("caps_directory")  # output of clinica pipelines
caps_reader = CapsReader(caps_directory, manager=manager)

preprocessing_1 = caps_reader.get_preprocessing("t1-linear")
caps_reader.prepare_data(
    preprocessing=preprocessing_1, data_tsv=Path(""), n_proc=2
)  # don't return anything -> just extract the image tensor and compute some information for each images

transforms_1 = Transforms(
    object_augmentation=[transforms.Crop, transforms.Transform],
    image_augmentation=[transforms.Crop, transforms.Transform],
    extraction=ExtractionPatchConfig(patch_size=3),
    image_transforms=[transforms.Blur, transforms.Ghosting],
    object_transforms=[transforms.BiasField, transforms.Motion],
)  # not mandatory

preprocessing_2 = caps_reader.get_preprocessing("pet-linear")
caps_reader.prepare_data(
    preprocessing=preprocessing_1, data_tsv=Path(""), n_proc=2
)  # to extract the tensor of the PET file this time
transforms_2 = Transforms(
    object_augmentation=[transforms.Crop, transforms.Transform],
    image_augmentation=[transforms.Crop, transforms.Transform],
    extraction=ExtractionSliceConfig(),
    image_transforms=[transforms.Blur, transforms.Ghosting],
    object_transforms=[transforms.BiasField, transforms.Motion],
)

sub_ses_tsv_1 = Path("")
split_dir_1 = split_tsv(sub_ses_tsv_1)  # -> creer un test.tsv et un train.tsv

sub_ses_tsv_2 = Path("")
split_dir_2 = split_tsv(sub_ses_tsv_2)

dataset_t1_roi = caps_reader.get_dataset(
    preprocessing=preprocessing_1,
    sub_ses_tsv=split_dir_1 / "train.tsv",
    transforms=transforms_1,
)  # do we give config or object for transforms ?
dataset_pet_patch = caps_reader.get_dataset(
    preprocessing=preprocessing_2,
    sub_ses_tsv=split_dir_2 / "train.tsv",
    transforms=transforms_2,
)

dataset_multi_modality_multi_extract = ConcatDataset(
    [
        dataset_t1_roi,
        dataset_pet_patch,
        caps_reader.get_dataset_from_json(json_path=Path("dataset.json")),
    ]
)  # 3 train.tsv en entrée qu'il faut concat et pareil pour les transforms à faire attention

# TODO : think about adding transforms in extract_json


config_file = Path("config_file")
trainer = Trainer.from_json(config_file=config_file, manager=manager)

# CAS CROSS-VALIDATION
splitter = KFolder(caps_dataset=dataset_multi_modality_multi_extract, manager=manager)
split_dir = splitter.make_splits(
    n_splits=3, output_dir=Path(""), subset_name="validation", stratification=""
)  # Optional data tsv and output_dir

dataloader_config = DataLoaderConfig(n_procs=3, batch_size=10)

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

preprocessing_test = caps_reader.get_preprocessing("pet-linear")
transforms_test = Transforms(
    object_augmentation=[transforms.Crop, transforms.Transform],
    image_augmentation=[transforms.Crop, transforms.Transform],
    extraction=ExtractioImageConfig(),
    image_transforms=[transforms.Blur, transforms.Ghosting],
    object_transforms=[transforms.BiasField, transforms.Motion],
)

dataset_test = caps_reader.get_dataset(
    preprocessing=preprocessing_test,
    sub_ses_tsv=split_dir_1 / "test.tsv",  # test only on data from the first dataset
    transforms=transforms_test,
)

predictor = Predictor(model=model, manager=manager)
predictor.predict(dataset_test=dataset_test, split_number=2)
