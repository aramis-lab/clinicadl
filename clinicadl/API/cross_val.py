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
from clinicadl.transforms.config import TransformsConfig

# # SIMPLE EXPERIMENT WITH A CAPS ALREADY EXISTING

# maps_path = Path("/")
# manager = ExperimentManager(maps_path, overwrite=False)

# dataset_t1_image = CapsDatasetPatch.from_json(
#     extraction=Path("test.json"),
#     sub_ses_tsv=Path("split_dir") / "train.tsv",
# )
# config_file = Path("config_file")
# trainer = Trainer.from_json(
#     config_file=config_file, manager=manager
# )  # gpu, amp, fsdp, seed

# # CAS CROSS-VALIDATION
# splitter = KFolder.from_dir(manager=manager)
# split_dir = splitter.make_splits(
#     n_splits=3,
#     output_dir=Path(""),
#     data_tsv=Path("labels.tsv"),
#     subset_name="validation",
#     stratification="",
# )  # Optional data tsv and output_dir
# # n_splits must be >1
# # for the single split case, this method output a path to the directory containing the train and test tsv files so we should have the same output here

# # Prérequis : déjà avoir des fichiers avec les listes train et validation
# split_dir = make_kfold("dataset.tsv") # lit dataset.tsv => fait le kfold => ecrit la sortie dans split_dir
# split_dir_2 = make_kfold("dataset.tsv", output_dr)
# split_dir_3 = make_kfold("dataset.tsv",)
# split_dir_4 = make_kfold("dataset.tsv")

# splitter = KFolder(dataset, split_dir) # c'est plutôt un iterable de dataloader

# # CAS EXISTING CROSS-VALIDATION
# splitter = KFolder(caps_dataset=dataset_t1_image)
# splitter.make_folds(n_splits = 3)
# splitter.write(split_dir)

# splitter.make_folds(n_splits = 5)
# # define the needed parameters for the dataloader
# dataloader_config = DataLoaderConfig(n_procs=3, batch_size=10)


# for split in splitter.get_splits(splits=(0, 3, 4), dataloader_config):
#     # bien définir ce qu'il y a dans l'objet split

#     network_config = create_network_config(ImplementedNetworks.CNN)(
#         in_shape=[2, 2, 2],
#         num_outputs=1,
#         conv_args=ConvEncoderOptions(channels=[3, 2, 2]),
#     )
#     optimizer, _ = get_optimizer(network, AdamConfig())
#     model = ClinicaDLModel(network=network_config, loss=nn.MSE(), optimizer=optimizer)

#     trainer.train(model, split)
#     # le trainer va instancier un predictor/valdiator dans le train ou dans le init
