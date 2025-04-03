from pathlib import Path

from clinicadl.data.dataloader.config import DataLoaderConfig
from clinicadl.data.datasets import CapsDataset
from clinicadl.experiment_manager import ExperimentManager
from clinicadl.splitter import KFold, make_kfold, make_split
from clinicadl.trainer import Trainer

# SIMPLE EXPERIMENT WITH A CAPS ALREADY EXISTING

maps_path = Path("/")
manager = ExperimentManager(maps_path, overwrite=False)

dataset_t1_image = CapsDataset.from_json(Path("json_path.json"))

config_file = Path("config_file")
trainer = Trainer.from_json(
    config_file=config_file, manager=manager
)  # gpu, amp, fsdp, seed

split_dir = make_split(
    dataset_t1_image.df, n_test=0.2, subset_name="validation", output_dir="test"
)  # Optional data tsv and output_dir
fold_dir = make_kfold(split_dir / "train.tsv", n_splits=2)

splitter = KFold(fold_dir)


# define the needed parameters for the dataloader
dataloader_config = DataLoaderConfig(num_workers=3, batch_size=10)


for split in splitter.get_splits(dataset=dataset_t1_image):
    split.build_train_loader(dataloader_config)
    split.build_val_loader(num_workers=3, batch_size=10)
