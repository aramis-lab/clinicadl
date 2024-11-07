from pathlib import Path

import torchio.transforms as transforms

from clinicadl.dataset.caps_reader import CapsReader
from clinicadl.dataset.concat import ConcatDataset
from clinicadl.dataset.config.extraction import ExtractionConfig
from clinicadl.dataset.config.preprocessing import (
    PreprocessingConfig,
    T1PreprocessingConfig,
)
from clinicadl.dataset.old_caps_dataset import (
    CapsDatasetPatch,
    CapsDatasetRoi,
    CapsDatasetSlice,
)
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
from clinicadl.transforms.config import TransformsConfig

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


transforms_1 = TransformsConfig(
    object_augmentation=[transforms.Crop, transforms.Transform],
    image_augmentation=[transforms.Crop, transforms.Transform],
    extraction=ExtractionPatchConfig(patch_size=3),
    image_transforms=[transforms.Blur, transforms.Ghosting],
    object_transforms=[transforms.BiasField, transforms.Motion],
)  # not mandatory

preprocessing_2 = caps_reader.get_preprocessing("pet-linear")
transforms_2 = TransformsConfig(
    object_augmentation=[transforms.Crop, transforms.Transform],
    image_augmentation=[transforms.Crop, transforms.Transform],
    extraction=ExtractionSliceConfig(),
    image_transforms=[transforms.Blur, transforms.Ghosting],
    object_transforms=[transforms.BiasField, transforms.Motion],
)

sub_ses_tsv = Path("")
split_dir = split_tsv(sub_ses_tsv)  # -> creer un test.tsv et un train.tsv

dataset_t1_roi = caps_reader.get_dataset(
    preprocessing=preprocessing_1,
    sub_ses_tsv=split_dir / "train.tsv",
    transforms=transforms_1,
)  # do we give config or object for transforms ?
dataset_pet_patch = caps_reader.get_dataset(
    preprocessing=preprocessing_2,
    sub_ses_tsv=split_dir / "train.tsv",
    transforms=transforms_2,
)

dataset_multi_modality_multi_extract = ConcatDataset(
    [
        dataset_t1_roi,
        dataset_pet_patch,
        caps_reader.get_dataset_from_json(json_path=Path(""), sub_ses_tsv=sub_ses_tsv),
    ]
)  # 2 train.tsv en entrée qu'il faut concat et pareil pour les transforms à faire attention

# TODO : think about adding transforms in extract_json


config_file = Path("config_file")
trainer = Trainer.from_json(config_file=config_file, manager=manager)

# CAS CROSS-VALIDATION
splitter = KFolder(
    n_splits=3, caps_dataset=dataset_multi_modality_multi_extract, manager=manager
)

for split in splitter.split_iterator(split_list=[0, 1]):
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
    sub_ses_tsv=split_dir / "test.tsv",
    transforms=transforms_test,
)

predictor = Predictor(manager=manager)
predictor.predict(dataset_test=dataset_test, split_number=2)
