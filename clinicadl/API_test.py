from pathlib import Path

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.data import return_dataset
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


dataset = return_dataset(
    input_dir,
    data_df,
    preprocessing_dict,
    transforms_config,
    label,
    label_code,
    cnn_index,
    label_presence,
    multi_cohort,
)


config = ClassificationConfig()
trainer = Trainer(config)
trainer.train(
    dataset.dataloader,
)
