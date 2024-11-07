from pathlib import Path

import torchio.transforms as transforms

from clinicadl.dataset.caps_reader import CapsReader
from clinicadl.dataset.concat import ConcatDataset
from clinicadl.dataset.config.extraction import ExtractionConfig, ExtractionPatchConfig
from clinicadl.dataset.config.preprocessing import (
    PreprocessingConfig,
    T1PreprocessingConfig,
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
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.enum import ExtractionMethod

# SIMPLE EXPERIMENT


maps_path = Path("/")
manager = ExperimentManager(maps_path, overwrite=False)

caps_directory = Path("caps_directory")  # output of clinica pipelines
caps_reader = CapsReader(
    caps_directory, manager=manager
)  # un peu bizarre de passer un maps_path a cet endroit via le manager pq on veut pas forcmeent faire un entrainement ??

preprocessing_t1 = caps_reader.get_preprocessing("t1-linear")
caps_reader.prepare_data(
    preprocessing=preprocessing_t1,
    data_tsv=Path(""),
    n_proc=2,
    use_uncropped_images=False,
)
transforms_1 = Transforms(
    object_augmentation=[transforms.RandomMotion()],  # default = no transforms
    image_augmentation=[transforms.RandomMotion()],  # default = no transforms
    object_transforms=[transforms.Blur((0.4, 0.5, 0.6))],  # default = none
    image_transforms=[transforms.Noise(0.2, 0.5, 3)],  # default = MiniMax
    extraction=ExtractionPatchConfig(patch_size=30, stride_size=20),  # default = Image
)  # not mandatory

sub_ses_tsv = Path("")
split_dir = split_tsv(sub_ses_tsv)  # -> creer un test.tsv et un train.tsv

# for the cli I think we should have a splitter for the single split case so we can have the same behaviour for single and kfold

dataset_t1_image = caps_reader.get_dataset(
    preprocessing=preprocessing_t1,
    sub_ses_tsv=split_dir / "train.tsv",
    transforms=transforms_1,
)  # do we give config or ob


# CAS SINGLE SPLIT
split = get_single_split(
    n_subject_validation=0,
    caps_dataset=dataset_t1_image,
    manager=manager,
)
config_file = Path("config_file")
trainer = Trainer.from_json(config_file=config_file, manager=manager)


loss, loss_config = get_loss_function(CrossEntropyLossConfig())
network_config = create_network_config(ImplementedNetworks.CNN)(
    in_shape=[2, 2, 2], num_outputs=1, conv_args=ConvEncoderOptions(channels=[3, 2, 2])
)
network, _ = get_network_from_config(network_config)
optimizer, _ = get_optimizer(network, AdamConfig())
model = ClinicaDLModelClassif(network=network, loss=loss, optimizer=optimizer)

trainer.train(model, split)
# le trainer va instancier un predictor/valdiator dans le train ou dans le init
