import click

from clinicadl.train.tasks import training_cli_options
from clinicadl.train.trainer import Task, Trainer
from clinicadl.train.utils import merge_cli_and_config_file_options

from ..classification import classification_cli_options
from .classification_config import ClassificationConfig


@click.command(name="classification", no_args_is_help=True)
# Mandatory arguments
@training_cli_options.caps_directory
@training_cli_options.preprocessing_json
@training_cli_options.tsv_directory
@training_cli_options.output_maps
# Options
@training_cli_options.config_file
# Computational
@training_cli_options.gpu
@training_cli_options.n_proc
@training_cli_options.batch_size
@training_cli_options.evaluation_steps
@training_cli_options.fully_sharded_data_parallel
@training_cli_options.amp
# Reproducibility
@training_cli_options.seed
@training_cli_options.deterministic
@training_cli_options.compensation
@training_cli_options.save_all_models
# Model
@classification_cli_options.architecture
@training_cli_options.multi_network
@training_cli_options.ssda_network
# Data
@training_cli_options.multi_cohort
@training_cli_options.diagnoses
@training_cli_options.baseline
@training_cli_options.valid_longitudinal
@training_cli_options.normalize
@training_cli_options.data_augmentation
@training_cli_options.sampler
@training_cli_options.caps_target
@training_cli_options.tsv_target_lab
@training_cli_options.tsv_target_unlab
@training_cli_options.preprocessing_json_target
# Cross validation
@training_cli_options.n_splits
@training_cli_options.split
# Optimization
@training_cli_options.optimizer
@training_cli_options.epochs
@training_cli_options.learning_rate
@training_cli_options.adaptive_learning_rate
@training_cli_options.weight_decay
@training_cli_options.dropout
@training_cli_options.patience
@training_cli_options.tolerance
@training_cli_options.accumulation_steps
@training_cli_options.profiler
@training_cli_options.track_exp
# transfer learning
@training_cli_options.transfer_path
@training_cli_options.transfer_selection_metric
@training_cli_options.nb_unfrozen_layer
# Task-related
@classification_cli_options.label
@classification_cli_options.selection_metrics
@classification_cli_options.threshold
@classification_cli_options.loss
# information
@training_cli_options.emissions_calculator
def cli(**kwargs):
    """
    Train a deep learning model to learn a classification task on neuroimaging data.

    CAPS_DIRECTORY is the CAPS folder from where tensors will be loaded.

    PREPROCESSING_JSON is the name of the JSON file in CAPS_DIRECTORY/tensor_extraction folder where
    all information about extraction are stored in order to read the wanted tensors.

    TSV_DIRECTORY is a folder were TSV files defining train and validation sets are stored.

    OUTPUT_MAPS_DIRECTORY is the path to the MAPS folder where outputs and results will be saved.

    Options for this command can be input by declaring argument on the command line or by providing a
    configuration file in TOML format. For more details, please visit the documentation:
    https://clinicadl.readthedocs.io/en/stable/Train/Introduction/#configuration-file
    """
    options = merge_cli_and_config_file_options(Task.CLASSIFICATION, **kwargs)
    config = ClassificationConfig(**options)
    trainer = Trainer(config)
    trainer.train(split_list=config.cross_validation.split, overwrite=True)
