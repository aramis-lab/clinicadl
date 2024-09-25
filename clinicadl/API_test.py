from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.prepare_data.prepare_data import DeepLearningPrepareData
from clinicadl.trainer.config.classification import ClassificationConfig
from clinicadl.trainer.trainer import Trainer
from clinicadl.utils.enum import ExtractionMethod, Preprocessing, Task
from clinicadl.utils.iotools.train_utils import merge_cli_and_config_file_options

image_config = CapsDatasetConfig.from_preprocessing_and_extraction_method(
    extraction=ExtractionMethod.IMAGE,
    preprocessing_type=Preprocessing.T1_LINEAR,
)

DeepLearningPrepareData(image_config)

config = ClassificationConfig()
trainer = Trainer(config)
trainer.train(split_list=config.cross_validation.split, overwrite=True)
