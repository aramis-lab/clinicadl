from pathlib import Path

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.predictor.predictor import Predictor
from clinicadl.splitter.new_splitter.dataloader import DataLoaderConfig
from clinicadl.splitter.new_splitter.splitter.kfold import KFold
from clinicadl.trainer.trainer import Trainer

# SIMPLE EXPERIMENT WITH A CAPS ALREADY EXISTING

maps_path = Path("/")
manager = ExperimentManager(maps_path, overwrite=False)

dataset_t1_image = CapsDataset.from_json(Path("json_path.json"))

config_file = Path("config_file")
trainer = Trainer.from_json(
    config_file=config_file, manager=manager
)  # gpu, amp, fsdp, seed

splitter = KFold(dataset=dataset_t1_image)
splitter.make_splits(n_splits=3)
split_dir = Path("")
splitter.write(split_dir)

splitter.read(split_dir)

# define the needed parameters for the dataloader
dataloader_config = DataLoaderConfig(num_workers=3, batch_size=10)


for split in splitter.get_splits(splits=(0, 3, 4)):
    print(split)
    split.build_train_loader(dataloader_config)
    split.build_val_loader(num_workers=3, batch_size=10)

    print(split)
