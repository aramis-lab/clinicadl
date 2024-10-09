from pathlib import Path

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.data import return_dataset
from clinicadl.predictor.config import PredictConfig
from clinicadl.predictor.predictor import Predictor
from clinicadl.prepare_data.prepare_data import DeepLearningPrepareData
from clinicadl.splitter.config import SplitterConfig
from clinicadl.splitter.splitter import Splitter
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

split_config = SplitterConfig()
splitter = Splitter(split_config)

validator_config = PredictConfig()
validator = Predictor(validator_config)

train_config = ClassificationConfig()
trainer = Trainer(train_config, validator)

for split in splitter.split_iterator():
    for network in range(
        first_network, self.maps_manager.num_networks
    ):  # for multi_network
        ###### actual _train_single method of the Trainer ############
        train_loader = trainer.get_dataloader(dataset, split, network, "train", config)
        valid_loader = validator.get_dataloader(
            dataset, split, network, "valid", config
        )  # ?? validatior, trainer ?

        trainer._train(
            train_loader,
            valid_loader,
            split=split,
            network=network,
            resume=resume,  # in a config class
            callbacks=[CodeCarbonTracker],  # in a config class ?
        )

        validator._ensemble_prediction(
            self.maps_manager,
            "train",
            split,
            self.config.validation.selection_metrics,
        )
        validator._ensemble_prediction(
            self.maps_manager,
            "validation",
            split,
            self.config.validation.selection_metrics,
        )
        ###### end ############


for split in splitter.split_iterator():
    for network in range(
        first_network, self.maps_manager.num_networks
    ):  # for multi_network
        ###### actual _train_single method of the Trainer ############
        test_loader = trainer.get_dataloader(dataset, split, network, "test", config)
        validator.predict(test_loader)

interpret_config = PredictConfig(**kwargs)
predict_manager = Predictor(interpret_config)
predict_manager.interpret()
