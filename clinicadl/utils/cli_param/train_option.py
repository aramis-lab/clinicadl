from typing import get_args

import click

from clinicadl.train.tasks.base_training_config import BaseTaskConfig
from clinicadl.train.tasks.classification_config import ClassificationConfig
from clinicadl.train.tasks.reconstruction_config import ReconstructionConfig
from clinicadl.train.tasks.regression_config import RegressionConfig
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
config = BaseTaskConfig.model_fields

# Computational
gpu = cli_param.option_group.computational_group.option(
    "--gpu/--no-gpu",
    default=config["gpu"].default,
    help="Use GPU by default. Please specify `--no-gpu` to force using CPU.",
    show_default=True,
)
n_proc = cli_param.option_group.computational_group.option(
    "-np",
    "--n_proc",
    type=config["n_proc"].annotation,
    default=config["n_proc"].default,
    help="Number of cores used during the task.",
    show_default=True,
)
batch_size = cli_param.option_group.computational_group.option(
    "--batch_size",
    type=config["batch_size"].annotation,
    default=config["batch_size"].default,
    help="Batch size for data loading.",
    show_default=True,
)
evaluation_steps = cli_param.option_group.computational_group.option(
    "--evaluation_steps",
    "-esteps",
    type=config["evaluation_steps"].annotation,
    default=config["evaluation_steps"].default,
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
    default=config["amp"].default,
    help="Enables automatic mixed precision during training and inference.",
    show_default=True,
)
# Reproducibility
seed = cli_param.option_group.reproducibility_group.option(
    "--seed",
    type=config["seed"].annotation,
    default=config["seed"].default,
    help="Value to set the seed for all random operations."
    "Default will sample a random value for the seed.",
    show_default=True,
)
deterministic = cli_param.option_group.reproducibility_group.option(
    "--deterministic/--nondeterministic",
    default=config["deterministic"].default,
    help="Forces Pytorch to be deterministic even when using a GPU. "
    "Will raise a RuntimeError if a non-deterministic function is encountered.",
    show_default=True,
)
compensation = cli_param.option_group.reproducibility_group.option(
    "--compensation",
    type=click.Choice(get_args(config["compensation"].annotation)),
    default=config["compensation"].default,
    help="Allow the user to choose how CUDA will compensate the deterministic behaviour.",
    show_default=True,
)
save_all_models = cli_param.option_group.reproducibility_group.option(
    "--save_all_models/--save_only_best_model",
    type=config["save_all_models"].annotation,
    default=config["save_all_models"].default,
    help="If provided, enables the saving of models weights for each epochs.",
    show_default=True,
)
# Model
architecture = cli_param.option_group.model_group.option(
    "-a",
    "--architecture",
    type=str,
    help="Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).",
)
multi_network = cli_param.option_group.model_group.option(
    "--multi_network/--single_network",
    default=config["multi_network"].default,
    help="If provided uses a multi-network framework.",
    show_default=True,
)
ssda_network = cli_param.option_group.model_group.option(
    "--ssda_network/--single_network",
    default=config["ssda_network"].default,
    help="If provided uses a ssda-network framework.",
    show_default=True,
)
# Task
classification_label = cli_param.option_group.task_group.option(
    "--label",
    type=ClassificationConfig.model_fields["label"].annotation,
    default=ClassificationConfig.model_fields["label"].default,
    help="Target label used for training.",
    show_default=True,
)
regression_label = cli_param.option_group.task_group.option(
    "--label",
    type=RegressionConfig.model_fields["label"].annotation,
    default=RegressionConfig.model_fields["label"].default,
    help="Target label used for training.",
    show_default=True,
)
classification_selection_metrics = cli_param.option_group.task_group.option(
    "--selection_metrics",
    "-sm",
    multiple=True,
    type=get_args(ClassificationConfig.model_fields["selection_metrics"].annotation)[0],
    default=ClassificationConfig.model_fields["selection_metrics"].default,
    help="""Allow to save a list of models based on their selection metric. Default will
    only save the best model selected on loss.""",
    show_default=True,
)
reconstruction_selection_metrics = cli_param.option_group.task_group.option(
    "--selection_metrics",
    "-sm",
    multiple=True,
    type=get_args(ReconstructionConfig.model_fields["selection_metrics"].annotation)[0],
    default=ReconstructionConfig.model_fields["selection_metrics"].default,
    help="""Allow to save a list of models based on their selection metric. Default will
    only save the best model selected on loss.""",
    show_default=True,
)
regression_selection_metrics = cli_param.option_group.task_group.option(
    "--selection_metrics",
    "-sm",
    multiple=True,
    type=get_args(RegressionConfig.model_fields["selection_metrics"].annotation)[0],
    default=RegressionConfig.model_fields["selection_metrics"].default,
    help="""Allow to save a list of models based on their selection metric. Default will
    only save the best model selected on loss.""",
    show_default=True,
)
selection_threshold = cli_param.option_group.task_group.option(
    "--selection_threshold",
    type=ClassificationConfig.model_fields["selection_threshold"].annotation,
    default=ClassificationConfig.model_fields["selection_threshold"].default,
    help="""Selection threshold for soft-voting. Will only be used if num_networks > 1.""",
    show_default=True,
)
classification_loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(ClassificationConfig.get_compatible_losses()),
    default=ClassificationConfig.model_fields["loss"].default,
    help="Loss used by the network to optimize its training task.",
    show_default=True,
)
regression_loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(RegressionConfig.get_compatible_losses()),
    default=RegressionConfig.model_fields["loss"].default,
    help="Loss used by the network to optimize its training task.",
    show_default=True,
)
reconstruction_loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(ReconstructionConfig.get_compatible_losses()),
    default=ReconstructionConfig.model_fields["loss"].default,
    help="Loss used by the network to optimize its training task.",
    show_default=True,
)
# Data
multi_cohort = cli_param.option_group.data_group.option(
    "--multi_cohort/--single_cohort",
    default=config["multi_cohort"].default,
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
    show_default=True,
)
diagnoses = cli_param.option_group.data_group.option(
    "--diagnoses",
    "-d",
    type=get_args(config["diagnoses"].annotation)[0],
    default=config["diagnoses"].default,
    multiple=True,
    help="List of diagnoses used for training.",
    show_default=True,
)
baseline = cli_param.option_group.data_group.option(
    "--baseline/--longitudinal",
    default=config["baseline"].default,
    help="If provided, only the baseline sessions are used for training.",
    show_default=True,
)
valid_longitudinal = cli_param.option_group.data_group.option(
    "--valid_longitudinal/--valid_baseline",
    default=config["valid_longitudinal"].default,
    help="If provided, not only the baseline sessions are used for validation (careful with this bad habit).",
    show_default=True,
)
normalize = cli_param.option_group.data_group.option(
    "--normalize/--unnormalize",
    default=config["normalize"].default,
    help="Disable default MinMaxNormalization.",
    show_default=True,
)
data_augmentation = cli_param.option_group.data_group.option(
    "--data_augmentation",
    "-da",
    type=click.Choice(BaseTaskConfig.get_available_transforms()),
    default=list(config["data_augmentation"].default),
    multiple=True,
    help="Randomly applies transforms on the training set.",
    show_default=True,
)
sampler = cli_param.option_group.data_group.option(
    "--sampler",
    "-s",
    type=click.Choice(get_args(config["sampler"].annotation)),
    default=config["sampler"].default,
    help="Sampler used to load the training data set.",
    show_default=True,
)
caps_target = cli_param.option_group.data_group.option(
    "--caps_target",
    "-d",
    type=config["caps_target"].annotation,
    default=config["caps_target"].default,
    help="CAPS of target data.",
    show_default=True,
)
tsv_target_lab = cli_param.option_group.data_group.option(
    "--tsv_target_lab",
    "-d",
    type=config["tsv_target_lab"].annotation,
    default=config["tsv_target_lab"].default,
    help="TSV of labeled target data.",
    show_default=True,
)
tsv_target_unlab = cli_param.option_group.data_group.option(
    "--tsv_target_unlab",
    "-d",
    type=config["tsv_target_unlab"].annotation,
    default=config["tsv_target_unlab"].default,
    help="TSV of unllabeled target data.",
    show_default=True,
)
preprocessing_dict_target = cli_param.option_group.data_group.option(  # TODO : change that name, it is not a dict.
    "--preprocessing_dict_target",
    "-d",
    type=config["preprocessing_dict_target"].annotation,
    default=config["preprocessing_dict_target"].default,
    help="Path to json target.",
    show_default=True,
)
# Cross validation
n_splits = cli_param.option_group.cross_validation.option(
    "--n_splits",
    type=config["n_splits"].annotation,
    default=config["n_splits"].default,
    help="If a value is given for k will load data of a k-fold CV. "
    "Default value (0) will load a single split.",
    show_default=True,
)
split = cli_param.option_group.cross_validation.option(
    "--split",
    "-s",
    type=get_args(config["split"].annotation)[0],
    default=config["split"].default,
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
    show_default=True,
)
# Optimization
optimizer = cli_param.option_group.optimization_group.option(
    "--optimizer",
    type=click.Choice(BaseTaskConfig.get_available_optimizers()),
    default=config["optimizer"].default,
    help="Optimizer used to train the network.",
    show_default=True,
)
epochs = cli_param.option_group.optimization_group.option(
    "--epochs",
    type=config["epochs"].annotation,
    default=config["epochs"].default,
    help="Maximum number of epochs.",
    show_default=True,
)
learning_rate = cli_param.option_group.optimization_group.option(
    "--learning_rate",
    "-lr",
    type=config["learning_rate"].annotation,
    default=config["learning_rate"].default,
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
    type=config["weight_decay"].annotation,
    default=config["weight_decay"].default,
    help="Weight decay value used in optimization.",
    show_default=True,
)
dropout = cli_param.option_group.optimization_group.option(
    "--dropout",
    type=config["dropout"].annotation,
    default=config["dropout"].default,
    help="Rate value applied to dropout layers in a CNN architecture.",
    show_default=True,
)
patience = cli_param.option_group.optimization_group.option(
    "--patience",
    type=config["patience"].annotation,
    default=config["patience"].default,
    help="Number of epochs for early stopping patience.",
    show_default=True,
)
tolerance = cli_param.option_group.optimization_group.option(
    "--tolerance",
    type=config["tolerance"].annotation,
    default=config["tolerance"].default,
    help="Value for early stopping tolerance.",
    show_default=True,
)
accumulation_steps = cli_param.option_group.optimization_group.option(
    "--accumulation_steps",
    "-asteps",
    type=config["accumulation_steps"].annotation,
    default=config["accumulation_steps"].default,
    help="Accumulates gradients during the given number of iterations before performing the weight update "
    "in order to virtually increase the size of the batch.",
    show_default=True,
)
profiler = cli_param.option_group.optimization_group.option(
    "--profiler/--no-profiler",
    default=config["profiler"].default,
    help="Use `--profiler` to enable Pytorch profiler for the first 30 steps after a short warmup. "
    "It will make an execution trace and some statistics about the CPU and GPU usage.",
    show_default=True,
)
track_exp = cli_param.option_group.optimization_group.option(
    "--track_exp",
    "-te",
    type=click.Choice(get_args(config["track_exp"].annotation)),
    default=config["track_exp"].default,
    help="Use `--track_exp` to enable wandb/mlflow to track the metric (loss, accuracy, etc...) during the training.",
    show_default=True,
)
# Transfer Learning
transfer_path = cli_param.option_group.transfer_learning_group.option(
    "-tp",
    "--transfer_path",
    type=get_args(config["transfer_path"].annotation)[0],
    default=config["transfer_path"].default,
    help="Path of to a MAPS used for transfer learning.",
    show_default=True,
)
transfer_selection_metric = cli_param.option_group.transfer_learning_group.option(
    "-tsm",
    "--transfer_selection_metric",
    type=config["transfer_selection_metric"].annotation,
    default=config["transfer_selection_metric"].default,
    help="Metric used to select the model for transfer learning in the MAPS defined by transfer_path.",
    show_default=True,
)
nb_unfrozen_layer = cli_param.option_group.transfer_learning_group.option(
    "-nul",
    "--nb_unfrozen_layer",
    type=config["nb_unfrozen_layer"].annotation,
    default=config["nb_unfrozen_layer"].default,
    help="Number of layer that will be retrain during training. For example, if it is 2, the last two layers of the model will not be freezed.",
    show_default=True,
)
# Information
emissions_calculator = cli_param.option_group.informations_group.option(
    "--calculate_emissions/--dont_calculate_emissions",
    default=config["emissions_calculator"].default,
    help="Flag to allow calculate the carbon emissions during training.",
    show_default=True,
)
