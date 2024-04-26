from typing import get_args

import click

from clinicadl.train.tasks.base_task_config import BaseTaskConfig
from clinicadl.utils import cli_param

# Arguments
caps_directory = cli_param.argument.caps_directory
preprocessing_json = cli_param.argument.preprocessing_json
tsv_directory = click.argument(
    "tsv_directory",
    type=click.Path(exists=True),
)
output_maps = cli_param.argument.output_maps
# Config file
config_file = click.option(
    "--config_file",
    "-c",
    type=click.Path(exists=True),
    help="Path to the TOML or JSON file containing the values of the options needed for training.",
)

# Options #
base_config = BaseTaskConfig.model_fields

# Computational
gpu = cli_param.option_group.computational_group.option(
    "--gpu/--no-gpu",
    default=base_config["gpu"].default,
    help="Use GPU by default. Please specify `--no-gpu` to force using CPU.",
    show_default=True,
)
n_proc = cli_param.option_group.computational_group.option(
    "-np",
    "--n_proc",
    type=base_config["n_proc"].annotation,
    default=base_config["n_proc"].default,
    help="Number of cores used during the task.",
    show_default=True,
)
batch_size = cli_param.option_group.computational_group.option(
    "--batch_size",
    type=base_config["batch_size"].annotation,
    default=base_config["batch_size"].default,
    help="Batch size for data loading.",
    show_default=True,
)
evaluation_steps = cli_param.option_group.computational_group.option(
    "--evaluation_steps",
    "-esteps",
    type=base_config["evaluation_steps"].annotation,
    default=base_config["evaluation_steps"].default,
    help="Fix the number of iterations to perform before computing an evaluation. Default will only "
    "perform one evaluation at the end of each epoch.",
    show_default=True,
)
fully_sharded_data_parallel = cli_param.option_group.computational_group.option(
    "--fully_sharded_data_parallel",
    "-fsdp",
    is_flag=True,
    help="Enables Fully Sharded Data Parallel with Pytorch to save memory at the cost of communications. "
    "Currently this only enables ZeRO Stage 1 but will be entirely replaced by FSDP in a later patch, "
    "this flag is already set to FSDP to that the zero flag is never actually removed.",
)
amp = cli_param.option_group.computational_group.option(
    "--amp/--no-amp",
    default=base_config["amp"].default,
    help="Enables automatic mixed precision during training and inference.",
    show_default=True,
)
# Reproducibility
seed = cli_param.option_group.reproducibility_group.option(
    "--seed",
    type=base_config["seed"].annotation,
    default=base_config["seed"].default,
    help="Value to set the seed for all random operations."
    "Default will sample a random value for the seed.",
    show_default=True,
)
deterministic = cli_param.option_group.reproducibility_group.option(
    "--deterministic/--nondeterministic",
    default=base_config["deterministic"].default,
    help="Forces Pytorch to be deterministic even when using a GPU. "
    "Will raise a RuntimeError if a non-deterministic function is encountered.",
    show_default=True,
)
compensation = cli_param.option_group.reproducibility_group.option(
    "--compensation",
    type=click.Choice(list(base_config["compensation"].annotation)),
    default=base_config["compensation"].default.value,
    help="Allow the user to choose how CUDA will compensate the deterministic behaviour.",
    show_default=True,
)
save_all_models = cli_param.option_group.reproducibility_group.option(
    "--save_all_models/--save_only_best_model",
    type=base_config["save_all_models"].annotation,
    default=base_config["save_all_models"].default,
    help="If provided, enables the saving of models weights for each epochs.",
    show_default=True,
)
# Model
multi_network = cli_param.option_group.model_group.option(
    "--multi_network/--single_network",
    default=base_config["multi_network"].default,
    help="If provided uses a multi-network framework.",
    show_default=True,
)
ssda_network = cli_param.option_group.model_group.option(
    "--ssda_network/--single_network",
    default=base_config["ssda_network"].default,
    help="If provided uses a ssda-network framework.",
    show_default=True,
)
# Data
multi_cohort = cli_param.option_group.data_group.option(
    "--multi_cohort/--single_cohort",
    default=base_config["multi_cohort"].default,
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
    show_default=True,
)
diagnoses = cli_param.option_group.data_group.option(
    "--diagnoses",
    "-d",
    type=get_args(base_config["diagnoses"].annotation)[0],
    default=base_config["diagnoses"].default,
    multiple=True,
    help="List of diagnoses used for training.",
    show_default=True,
)
baseline = cli_param.option_group.data_group.option(
    "--baseline/--longitudinal",
    default=base_config["baseline"].default,
    help="If provided, only the baseline sessions are used for training.",
    show_default=True,
)
valid_longitudinal = cli_param.option_group.data_group.option(
    "--valid_longitudinal/--valid_baseline",
    default=base_config["valid_longitudinal"].default,
    help="If provided, not only the baseline sessions are used for validation (careful with this bad habit).",
    show_default=True,
)
normalize = cli_param.option_group.data_group.option(
    "--normalize/--unnormalize",
    default=base_config["normalize"].default,
    help="Disable default MinMaxNormalization.",
    show_default=True,
)
data_augmentation = cli_param.option_group.data_group.option(
    "--data_augmentation",
    "-da",
    type=click.Choice(BaseTaskConfig.get_available_transforms()),
    default=list(base_config["data_augmentation"].default),
    multiple=True,
    help="Randomly applies transforms on the training set.",
    show_default=True,
)
sampler = cli_param.option_group.data_group.option(
    "--sampler",
    "-s",
    type=click.Choice(list(base_config["sampler"].annotation)),
    default=base_config["sampler"].default.value,
    help="Sampler used to load the training data set.",
    show_default=True,
)
caps_target = cli_param.option_group.data_group.option(
    "--caps_target",
    "-d",
    type=base_config["caps_target"].annotation,
    default=base_config["caps_target"].default,
    help="CAPS of target data.",
    show_default=True,
)
tsv_target_lab = cli_param.option_group.data_group.option(
    "--tsv_target_lab",
    "-d",
    type=base_config["tsv_target_lab"].annotation,
    default=base_config["tsv_target_lab"].default,
    help="TSV of labeled target data.",
    show_default=True,
)
tsv_target_unlab = cli_param.option_group.data_group.option(
    "--tsv_target_unlab",
    "-d",
    type=base_config["tsv_target_unlab"].annotation,
    default=base_config["tsv_target_unlab"].default,
    help="TSV of unllabeled target data.",
    show_default=True,
)
preprocessing_dict_target = cli_param.option_group.data_group.option(  # TODO : change that name, it is not a dict.
    "--preprocessing_dict_target",
    "-d",
    type=base_config["preprocessing_dict_target"].annotation,
    default=base_config["preprocessing_dict_target"].default,
    help="Path to json target.",
    show_default=True,
)
# Cross validation
n_splits = cli_param.option_group.cross_validation.option(
    "--n_splits",
    type=base_config["n_splits"].annotation,
    default=base_config["n_splits"].default,
    help="If a value is given for k will load data of a k-fold CV. "
    "Default value (0) will load a single split.",
    show_default=True,
)
split = cli_param.option_group.cross_validation.option(
    "--split",
    "-s",
    type=get_args(base_config["split"].annotation)[0],
    default=base_config["split"].default,
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
    show_default=True,
)
# Optimization
optimizer = cli_param.option_group.optimization_group.option(
    "--optimizer",
    type=click.Choice(BaseTaskConfig.get_available_optimizers()),
    default=base_config["optimizer"].default,
    help="Optimizer used to train the network.",
    show_default=True,
)
epochs = cli_param.option_group.optimization_group.option(
    "--epochs",
    type=base_config["epochs"].annotation,
    default=base_config["epochs"].default,
    help="Maximum number of epochs.",
    show_default=True,
)
learning_rate = cli_param.option_group.optimization_group.option(
    "--learning_rate",
    "-lr",
    type=base_config["learning_rate"].annotation,
    default=base_config["learning_rate"].default,
    help="Learning rate of the optimization.",
    show_default=True,
)
adaptive_learning_rate = cli_param.option_group.optimization_group.option(
    "--adaptive_learning_rate",
    "-alr",
    is_flag=True,
    help="Whether to diminish the learning rate",
)
weight_decay = cli_param.option_group.optimization_group.option(
    "--weight_decay",
    "-wd",
    type=base_config["weight_decay"].annotation,
    default=base_config["weight_decay"].default,
    help="Weight decay value used in optimization.",
    show_default=True,
)
dropout = cli_param.option_group.optimization_group.option(
    "--dropout",
    type=base_config["dropout"].annotation,
    default=base_config["dropout"].default,
    help="Rate value applied to dropout layers in a CNN architecture.",
    show_default=True,
)
patience = cli_param.option_group.optimization_group.option(
    "--patience",
    type=base_config["patience"].annotation,
    default=base_config["patience"].default,
    help="Number of epochs for early stopping patience.",
    show_default=True,
)
tolerance = cli_param.option_group.optimization_group.option(
    "--tolerance",
    type=base_config["tolerance"].annotation,
    default=base_config["tolerance"].default,
    help="Value for early stopping tolerance.",
    show_default=True,
)
accumulation_steps = cli_param.option_group.optimization_group.option(
    "--accumulation_steps",
    "-asteps",
    type=base_config["accumulation_steps"].annotation,
    default=base_config["accumulation_steps"].default,
    help="Accumulates gradients during the given number of iterations before performing the weight update "
    "in order to virtually increase the size of the batch.",
    show_default=True,
)
profiler = cli_param.option_group.optimization_group.option(
    "--profiler/--no-profiler",
    default=base_config["profiler"].default,
    help="Use `--profiler` to enable Pytorch profiler for the first 30 steps after a short warmup. "
    "It will make an execution trace and some statistics about the CPU and GPU usage.",
    show_default=True,
)
track_exp = cli_param.option_group.optimization_group.option(
    "--track_exp",
    "-te",
    type=click.Choice(list(get_args(base_config["track_exp"].annotation)[0])),
    default=base_config["track_exp"].default,
    help="Use `--track_exp` to enable wandb/mlflow to track the metric (loss, accuracy, etc...) during the training.",
    show_default=True,
)
# Transfer Learning
transfer_path = cli_param.option_group.transfer_learning_group.option(
    "-tp",
    "--transfer_path",
    type=get_args(base_config["transfer_path"].annotation)[0],
    default=base_config["transfer_path"].default,
    help="Path of to a MAPS used for transfer learning.",
    show_default=True,
)
transfer_selection_metric = cli_param.option_group.transfer_learning_group.option(
    "-tsm",
    "--transfer_selection_metric",
    type=base_config["transfer_selection_metric"].annotation,
    default=base_config["transfer_selection_metric"].default,
    help="Metric used to select the model for transfer learning in the MAPS defined by transfer_path.",
    show_default=True,
)
nb_unfrozen_layer = cli_param.option_group.transfer_learning_group.option(
    "-nul",
    "--nb_unfrozen_layer",
    type=base_config["nb_unfrozen_layer"].annotation,
    default=base_config["nb_unfrozen_layer"].default,
    help="Number of layer that will be retrain during training. For example, if it is 2, the last two layers of the model will not be freezed.",
    show_default=True,
)
# Information
emissions_calculator = cli_param.option_group.informations_group.option(
    "--calculate_emissions/--dont_calculate_emissions",
    default=base_config["emissions_calculator"].default,
    help="Flag to allow calculate the carbon emissions during training.",
    show_default=True,
)
