import click

from clinicadl.train import preprocessing_json_reader
from clinicadl.train.tasks.base_training_config import Task
from clinicadl.train.train_utils import merge_cli_and_config_file_options
from clinicadl.utils.cli_param import train_option
from clinicadl.utils.maps_manager import MapsManager
from clinicadl.utils.trainer import Trainer

from .reconstruction_config import ReconstructionConfig


@click.command(name="reconstruction", no_args_is_help=True)
# Mandatory arguments
@train_option.caps_directory
@train_option.preprocessing_json
@train_option.tsv_directory
@train_option.output_maps
# Options
@train_option.config_file
# Computational
@train_option.gpu
@train_option.n_proc
@train_option.batch_size
@train_option.evaluation_steps
@train_option.fully_sharded_data_parallel
@train_option.amp
# Reproducibility
@train_option.seed
@train_option.deterministic
@train_option.compensation
@train_option.save_all_models
# Model
@train_option.reconstruction_architecture
@train_option.multi_network
@train_option.ssda_network
# Data
@train_option.multi_cohort
@train_option.diagnoses
@train_option.baseline
@train_option.valid_longitudinal
@train_option.normalize
@train_option.data_augmentation
@train_option.sampler
@train_option.caps_target
@train_option.tsv_target_lab
@train_option.tsv_target_unlab
@train_option.preprocessing_dict_target
# Cross validation
@train_option.n_splits
@train_option.split
# Optimization
@train_option.optimizer
@train_option.epochs
@train_option.learning_rate
@train_option.adaptive_learning_rate
@train_option.weight_decay
@train_option.dropout
@train_option.patience
@train_option.tolerance
@train_option.accumulation_steps
@train_option.profiler
@train_option.track_exp
# transfer learning
@train_option.transfer_path
@train_option.transfer_selection_metric
@train_option.nb_unfrozen_layer
# Task-related
@train_option.reconstruction_selection_metrics
@train_option.reconstruction_loss
# information
@train_option.emissions_calculator
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
