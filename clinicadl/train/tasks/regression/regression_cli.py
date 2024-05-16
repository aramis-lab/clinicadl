import click

from clinicadl.train.tasks import train_task_cli_options
from clinicadl.train.trainer import Task, Trainer
from clinicadl.train.utils import merge_cli_and_config_file_options

from ..regression import regression_cli_options
from .regression_config import RegressionConfig


@click.command(name="regression", no_args_is_help=True)
# Mandatory arguments
@train_task_cli_options.caps_directory
@train_task_cli_options.preprocessing_json
@train_task_cli_options.tsv_directory
@train_task_cli_options.output_maps
# Options
@train_task_cli_options.config_file
# Computational
@train_task_cli_options.gpu
@train_task_cli_options.n_proc
@train_task_cli_options.batch_size
@train_task_cli_options.evaluation_steps
@train_task_cli_options.fully_sharded_data_parallel
@train_task_cli_options.amp
# Reproducibility
@train_task_cli_options.seed
@train_task_cli_options.deterministic
@train_task_cli_options.compensation
@train_task_cli_options.save_all_models
# Model
@regression_cli_options.architecture
@train_task_cli_options.multi_network
@train_task_cli_options.ssda_network
# Data
@train_task_cli_options.multi_cohort
@train_task_cli_options.diagnoses
@train_task_cli_options.baseline
@train_task_cli_options.valid_longitudinal
@train_task_cli_options.normalize
@train_task_cli_options.data_augmentation
@train_task_cli_options.sampler
@train_task_cli_options.caps_target
@train_task_cli_options.tsv_target_lab
@train_task_cli_options.tsv_target_unlab
@train_task_cli_options.preprocessing_json_target
# Cross validation
@train_task_cli_options.n_splits
@train_task_cli_options.split
# Optimization
@train_task_cli_options.optimizer
@train_task_cli_options.epochs
@train_task_cli_options.learning_rate
@train_task_cli_options.adaptive_learning_rate
@train_task_cli_options.weight_decay
@train_task_cli_options.dropout
@train_task_cli_options.patience
@train_task_cli_options.tolerance
@train_task_cli_options.accumulation_steps
@train_task_cli_options.profiler
@train_task_cli_options.track_exp
# transfer learning
@train_task_cli_options.transfer_path
@train_task_cli_options.transfer_selection_metric
@train_task_cli_options.nb_unfrozen_layer
# Task-related
@regression_cli_options.label
@regression_cli_options.selection_metrics
@regression_cli_options.loss
# information
@train_task_cli_options.emissions_calculator
def cli(**kwargs):
    """
    Train a deep learning model to learn a regression task on neuroimaging data.

    CAPS_DIRECTORY is the CAPS folder from where tensors will be loaded.

    PREPROCESSING_JSON is the name of the JSON file in CAPS_DIRECTORY/tensor_extraction folder where
    all information about extraction are stored in order to read the wanted tensors.

    TSV_DIRECTORY is a folder were TSV files defining train and validation sets are stored.

    OUTPUT_MAPS_DIRECTORY is the path to the MAPS folder where outputs and results will be saved.

    Options for this command can be input by declaring argument on the command line or by providing a
    configuration file in TOML format. For more details, please visit the documentation:
    https://clinicadl.readthedocs.io/en/stable/Train/Introduction/#configuration-file
    """
    options = merge_cli_and_config_file_options(Task.REGRESSION, **kwargs)
    config = RegressionConfig(**options)
    trainer = Trainer(config)
    trainer.train(split_list=config.cross_validation.split, overwrite=True)
