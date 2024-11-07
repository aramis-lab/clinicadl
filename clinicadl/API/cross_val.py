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

# SIMPLE EXPERIMENT WITH A CAPS ALREADY EXISTING

maps_path = Path("/")
manager = ExperimentManager(maps_path, overwrite=False)

dataset_t1_image = CapsDatasetPatch.from_json(
    extraction=Path("test.json"),
    sub_ses_tsv=Path("split_dir") / "train.tsv",
)
config_file = Path("config_file")
trainer = Trainer.from_json(config_file=config_file, manager=manager)

# CAS CROSS-VALIDATION
splitter = KFolder(n_splits=3, caps_dataset=dataset_t1_image, manager=manager)

for split in splitter.split_iterator(split_list=[0, 1]):
    # bien définir ce qu'il y a dans l'objet split

    loss, loss_config = get_loss_function(CrossEntropyLossConfig())
    network_config = create_network_config(ImplementedNetworks.CNN)(
        in_shape=[2, 2, 2],
        num_outputs=1,
        conv_args=ConvEncoderOptions(channels=[3, 2, 2]),
    )
    network, _ = get_network_from_config(network_config)
    optimizer, _ = get_optimizer(network, AdamConfig())
    model = ClinicaDLModel(network=network, loss=loss, optimizer=optimizer)

    trainer.train(model, split)
    # le trainer va instancier un predictor/valdiator dans le train ou dans le init
