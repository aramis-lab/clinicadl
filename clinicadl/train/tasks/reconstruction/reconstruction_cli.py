import click

from clinicadl.train.tasks import Task, base_task_cli_options
from clinicadl.train.trainer import Trainer
from clinicadl.train.utils import (
    merge_cli_and_config_file_options,
    preprocessing_json_reader,
)
from clinicadl.utils.maps_manager import MapsManager

from ..reconstruction import reconstruction_cli_options
from .reconstruction_config import ReconstructionConfig


@click.command(name="reconstruction", no_args_is_help=True)
# Mandatory arguments
@base_task_cli_options.caps_directory
@base_task_cli_options.preprocessing_json
@base_task_cli_options.tsv_directory
@base_task_cli_options.output_maps
# Options
@base_task_cli_options.config_file
# Computational
@base_task_cli_options.gpu
@base_task_cli_options.n_proc
@base_task_cli_options.batch_size
@base_task_cli_options.evaluation_steps
@base_task_cli_options.fully_sharded_data_parallel
@base_task_cli_options.amp
# Reproducibility
@base_task_cli_options.seed
@base_task_cli_options.deterministic
@base_task_cli_options.compensation
@base_task_cli_options.save_all_models
# Model
@reconstruction_cli_options.architecture
@base_task_cli_options.multi_network
@base_task_cli_options.ssda_network
# Data
@base_task_cli_options.multi_cohort
@base_task_cli_options.diagnoses
@base_task_cli_options.baseline
@base_task_cli_options.valid_longitudinal
@base_task_cli_options.normalize
@base_task_cli_options.data_augmentation
@base_task_cli_options.sampler
@base_task_cli_options.caps_target
@base_task_cli_options.tsv_target_lab
@base_task_cli_options.tsv_target_unlab
@base_task_cli_options.preprocessing_dict_target
# Cross validation
@base_task_cli_options.n_splits
@base_task_cli_options.split
# Optimization
@base_task_cli_options.optimizer
@base_task_cli_options.epochs
@base_task_cli_options.learning_rate
@base_task_cli_options.adaptive_learning_rate
@base_task_cli_options.weight_decay
@base_task_cli_options.dropout
@base_task_cli_options.patience
@base_task_cli_options.tolerance
@base_task_cli_options.accumulation_steps
@base_task_cli_options.profiler
@base_task_cli_options.track_exp
# transfer learning
@base_task_cli_options.transfer_path
@base_task_cli_options.transfer_selection_metric
@base_task_cli_options.nb_unfrozen_layer
# Task-related
@reconstruction_cli_options.selection_metrics
@reconstruction_cli_options.loss
# information
@base_task_cli_options.emissions_calculator
def cli(**kwargs):
    """
    Train a deep learning model to learn a reconstruction task on neuroimaging data.

    CAPS_DIRECTORY is the CAPS folder from where tensors will be loaded.

    PREPROCESSING_JSON is the name of the JSON file in CAPS_DIRECTORY/tensor_extraction folder where
    all information about extraction are stored in order to read the wanted tensors.

    TSV_DIRECTORY is a folder were TSV files defining train and validation sets are stored.

    OUTPUT_MAPS_DIRECTORY is the path to the MAPS folder where outputs and results will be saved.

    Options for this command can be input by declaring argument on the command line or by providing a
    configuration file in TOML format. For more details, please visit the documentation:
    https://clinicadl.readthedocs.io/en/stable/Train/Introduction/#configuration-file
    """
    options = merge_cli_and_config_file_options(Task.RECONSTRUCTION, **kwargs)
    config = ReconstructionConfig(**options)
    config = preprocessing_json_reader(
        config
    )  # TODO : put elsewhere. In BaseTaskConfig?

    # temporary # TODO : change MAPSManager and Trainer to give them a config object
    maps_dir = config.output_maps_directory
    train_dict = config.model_dump(
        exclude=["output_maps_directory", "preprocessing_json", "tsv_directory"]
    )
    train_dict["tsv_path"] = config.tsv_directory
    train_dict[
        "preprocessing_dict"
    ] = config._preprocessing_dict  # private attributes are not dumped
    train_dict["mode"] = config._mode
    if config.ssda_network:
        train_dict["preprocessing_dict_target"] = config._preprocessing_dict_target
    train_dict["network_task"] = config._network_task
    if train_dict["transfer_path"] is None:
        train_dict["transfer_path"] = False
    if train_dict["data_augmentation"] == ():
        train_dict["data_augmentation"] = False
    split_list = train_dict.pop("split")
    train_dict["compensation"] = config.compensation.value
    train_dict["size_reduction_factor"] = config.size_reduction_factor.value
    if train_dict["track_exp"]:
        train_dict["track_exp"] = config.track_exp.value
    else:
        train_dict["track_exp"] = ""
    train_dict["sampler"] = config.sampler.value
    if train_dict["network_task"] == "reconstruction":
        train_dict["normalization"] = config.normalization.value
    #############

    maps_manager = MapsManager(maps_dir, train_dict, verbose=None)
    trainer = Trainer(maps_manager)
    trainer.train(split_list=split_list, overwrite=True)
