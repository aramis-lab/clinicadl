import click

from clinicadl.config import arguments
from clinicadl.config.options import (
    callbacks,
    computational,
    cross_validation,
    data,
    dataloader,
    early_stopping,
    lr_scheduler,
    model,
    optimization,
    optimizer,
    reproducibility,
    ssda,
    task,
    transfer_learning,
    transforms,
    validation,
)
from clinicadl.train.tasks.classification.config import ClassificationConfig
from clinicadl.train.trainer import Trainer
from clinicadl.train.utils import merge_cli_and_config_file_options
from clinicadl.utils.enum import Task


@click.command(name="classification", no_args_is_help=True)
# Mandatory arguments
@arguments.caps_directory
@arguments.preprocessing_json
@arguments.tsv_directory
@arguments.output_maps
# Options
# Computational
@computational.gpu
@computational.fully_sharded_data_parallel
@computational.amp
# Reproducibility
@reproducibility.seed
@reproducibility.deterministic
@reproducibility.compensation
@reproducibility.save_all_models
@reproducibility.config_file
# Model
@model.dropout
@model.multi_network
# Data
@data.multi_cohort
@data.diagnoses
@data.baseline
# validation
@validation.valid_longitudinal
@validation.evaluation_steps
# transforms
@transforms.normalize
@transforms.data_augmentation
# dataloader
@dataloader.batch_size
@dataloader.sampler
@dataloader.n_proc
# ssda option
@ssda.ssda_network
@ssda.caps_target
@ssda.tsv_target_lab
@ssda.tsv_target_unlab
@ssda.preprocessing_json_target
# Cross validation
@cross_validation.n_splits
@cross_validation.split
# Optimization
@optimizer.optimizer
@optimizer.weight_decay
@optimizer.learning_rate
# lr scheduler
@lr_scheduler.adaptive_learning_rate
# early stopping
@early_stopping.patience
@early_stopping.tolerance
# optimization
@optimization.accumulation_steps
@optimization.profiler
@optimization.epochs
# transfer learning
@transfer_learning.transfer_path
@transfer_learning.transfer_selection_metric
@transfer_learning.nb_unfrozen_layer
# callbacks
@callbacks.emissions_calculator
@callbacks.track_exp
# Task-related
@task.classification.architecture
@task.classification.label
@task.classification.selection_metrics
@task.classification.threshold
@task.classification.loss
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

    kwargs["tsv_dir"] = kwargs["tsv_directory"]
    kwargs["maps_dir"] = kwargs["output_maps_directory"]
    options = merge_cli_and_config_file_options(Task.CLASSIFICATION, **kwargs)

    options["maps_dir"] = options["output_maps_directory"]
    config = ClassificationConfig(**options)
    trainer = Trainer(config)
    trainer.train(split_list=config.cross_validation.split, overwrite=True)
