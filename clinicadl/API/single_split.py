from pathlib import Path

import torchio.transforms as transforms

from clinicadl.data.datasets.caps_dataset import CapsDataset
from clinicadl.data.datasets.concat import ConcatDataset
from clinicadl.data.preprocessing import (
    PreprocessingPET,
    PreprocessingT1,
)
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.losses.config import CrossEntropyLossConfig
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.config.resnet import ResNetConfig
from clinicadl.optim.optimizers.config import AdamConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.dataloader import DataLoaderConfig
from clinicadl.splitter.make_splits import make_kfold, make_split
from clinicadl.splitter.splitter import KFold, SingleSplit
from clinicadl.trainer.trainer import Trainer
from clinicadl.transforms.extraction import Image
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

dataset_t1_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_t1,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
)
dataset_t1_image.prepare_data(n_proc=2)  # to extract the tensor of the T1 file


split_dir = make_split(
    sub_ses_t1, n_test=0.2, subset_name="test"
)  # Optional data tsv and output_dir
split_dir_val = make_split(
    split_dir / "train.tsv", n_test=0.2, subset_name="validation"
)

splitter = SingleSplit(split_dir_val)


maps_path = Path("/")
manager = ExperimentManager(maps_path, overwrite=False)

config_file = Path("config_file")
trainer = Trainer.from_json(config_file=config_file, manager=manager)

split = splitter.get_splits(dataset=dataset_t1_image)

train_loader = split.build_train_loader(batch_size=2)
val_loader = split.build_val_loader(DataLoaderConfig())

model = ClinicaDLModel.from_config(
    network_config=ResNetConfig(num_outputs=1, spatial_dims=1, in_channels=1),
    loss_config=CrossEntropyLossConfig(),
    optimizer_config=AdamConfig(),
)

trainer.train(model, split)
