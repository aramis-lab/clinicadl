import click

from clinicadl.train.tasks.base_task_config import BaseTaskConfig
from clinicadl.utils import cli_param
from clinicadl.utils.config_utils import get_default_from_config_class as get_default
from clinicadl.utils.config_utils import get_type_from_config_class as get_type

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
base_config = BaseTaskConfig

# Computational
gpu = cli_param.option_group.computational_group.option(
    "--gpu/--no-gpu",
    default=get_default("gpu", base_config),
    help="Use GPU by default. Please specify `--no-gpu` to force using CPU.",
    show_default=True,
)
n_proc = cli_param.option_group.computational_group.option(
    "-np",
    "--n_proc",
    type=get_type("n_proc", base_config),
    default=get_default("n_proc", base_config),
    help="Number of cores used during the task.",
    show_default=True,
)
batch_size = cli_param.option_group.computational_group.option(
    "--batch_size",
    type=get_type("batch_size", base_config),
    default=get_default("batch_size", base_config),
    help="Batch size for data loading.",
    show_default=True,
)
evaluation_steps = cli_param.option_group.computational_group.option(
    "--evaluation_steps",
    "-esteps",
    type=get_type("evaluation_steps", base_config),
    default=get_default("evaluation_steps", base_config),
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
    default=get_default("amp", base_config),
    help="Enables automatic mixed precision during training and inference.",
    show_default=True,
)
# Reproducibility
seed = cli_param.option_group.reproducibility_group.option(
    "--seed",
    type=get_type("seed", base_config),
    default=get_default("seed", base_config),
    help="Value to set the seed for all random operations."
    "Default will sample a random value for the seed.",
    show_default=True,
)
deterministic = cli_param.option_group.reproducibility_group.option(
    "--deterministic/--nondeterministic",
    default=get_default("deterministic", base_config),
    help="Forces Pytorch to be deterministic even when using a GPU. "
    "Will raise a RuntimeError if a non-deterministic function is encountered.",
    show_default=True,
)
compensation = cli_param.option_group.reproducibility_group.option(
    "--compensation",
    type=get_type("compensation", base_config),
    default=get_default("compensation", base_config),
    help="Allow the user to choose how CUDA will compensate the deterministic behaviour.",
    show_default=True,
)
save_all_models = cli_param.option_group.reproducibility_group.option(
    "--save_all_models/--save_only_best_model",
    type=get_type("save_all_models", base_config),
    default=get_default("save_all_models", base_config),
    help="If provided, enables the saving of models weights for each epochs.",
    show_default=True,
)
# Model
multi_network = cli_param.option_group.model_group.option(
    "--multi_network/--single_network",
    default=get_default("multi_network", base_config),
    help="If provided uses a multi-network framework.",
    show_default=True,
)
ssda_network = cli_param.option_group.model_group.option(
    "--ssda_network/--single_network",
    default=get_default("ssda_network", base_config),
    help="If provided uses a ssda-network framework.",
    show_default=True,
)
# Data
multi_cohort = cli_param.option_group.data_group.option(
    "--multi_cohort/--single_cohort",
    default=get_default("multi_cohort", base_config),
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
    show_default=True,
)
diagnoses = cli_param.option_group.data_group.option(
    "--diagnoses",
    "-d",
    type=get_type("diagnoses", base_config),
    default=get_default("diagnoses", base_config),
    multiple=True,
    help="List of diagnoses used for training.",
    show_default=True,
)
baseline = cli_param.option_group.data_group.option(
    "--baseline/--longitudinal",
    default=get_default("baseline", base_config),
    help="If provided, only the baseline sessions are used for training.",
    show_default=True,
)
valid_longitudinal = cli_param.option_group.data_group.option(
    "--valid_longitudinal/--valid_baseline",
    default=get_default("valid_longitudinal", base_config),
    help="If provided, not only the baseline sessions are used for validation (careful with this bad habit).",
    show_default=True,
)
normalize = cli_param.option_group.data_group.option(
    "--normalize/--unnormalize",
    default=get_default("normalize", base_config),
    help="Disable default MinMaxNormalization.",
    show_default=True,
)
data_augmentation = cli_param.option_group.data_group.option(
    "--data_augmentation",
    "-da",
    type=get_type("data_augmentation", base_config),
    default=get_default("data_augmentation", base_config),
    multiple=True,
    help="Randomly applies transforms on the training set.",
    show_default=True,
)
sampler = cli_param.option_group.data_group.option(
    "--sampler",
    "-s",
    type=get_type("sampler", base_config),
    default=get_default("sampler", base_config),
    help="Sampler used to load the training data set.",
    show_default=True,
)
caps_target = cli_param.option_group.data_group.option(
    "--caps_target",
    "-d",
    type=get_type("caps_target", base_config),
    default=get_default("caps_target", base_config),
    help="CAPS of target data.",
    show_default=True,
)
tsv_target_lab = cli_param.option_group.data_group.option(
    "--tsv_target_lab",
    "-d",
    type=get_type("tsv_target_lab", base_config),
    default=get_default("tsv_target_lab", base_config),
    help="TSV of labeled target data.",
    show_default=True,
)
tsv_target_unlab = cli_param.option_group.data_group.option(
    "--tsv_target_unlab",
    "-d",
    type=get_type("tsv_target_unlab", base_config),
    default=get_default("tsv_target_unlab", base_config),
    help="TSV of unllabeled target data.",
    show_default=True,
)
preprocessing_dict_target = cli_param.option_group.data_group.option(  # TODO : change that name, it is not a dict.
    "--preprocessing_dict_target",
    "-d",
    type=get_type("preprocessing_dict_target", base_config),
    default=get_default("preprocessing_dict_target", base_config),
    help="Path to json target.",
    show_default=True,
)
# Cross validation
n_splits = cli_param.option_group.cross_validation.option(
    "--n_splits",
    type=get_type("n_splits", base_config),
    default=get_default("n_splits", base_config),
    help="If a value is given for k will load data of a k-fold CV. "
    "Default value (0) will load a single split.",
    show_default=True,
)
split = cli_param.option_group.cross_validation.option(
    "--split",
    "-s",
    type=get_type("split", base_config),
    default=get_default("split", base_config),
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
    show_default=True,
)
# Optimization
optimizer = cli_param.option_group.optimization_group.option(
    "--optimizer",
    type=get_type("optimizer", base_config),
    default=get_default("optimizer", base_config),
    help="Optimizer used to train the network.",
    show_default=True,
)
epochs = cli_param.option_group.optimization_group.option(
    "--epochs",
    type=get_type("epochs", base_config),
    default=get_default("epochs", base_config),
    help="Maximum number of epochs.",
    show_default=True,
)
learning_rate = cli_param.option_group.optimization_group.option(
    "--learning_rate",
    "-lr",
    type=get_type("learning_rate", base_config),
    default=get_default("learning_rate", base_config),
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
    type=get_type("weight_decay", base_config),
    default=get_default("weight_decay", base_config),
    help="Weight decay value used in optimization.",
    show_default=True,
)
dropout = cli_param.option_group.optimization_group.option(
    "--dropout",
    type=get_type("dropout", base_config),
    default=get_default("dropout", base_config),
    help="Rate value applied to dropout layers in a CNN architecture.",
    show_default=True,
)
patience = cli_param.option_group.optimization_group.option(
    "--patience",
    type=get_type("patience", base_config),
    default=get_default("patience", base_config),
    help="Number of epochs for early stopping patience.",
    show_default=True,
)
tolerance = cli_param.option_group.optimization_group.option(
    "--tolerance",
    type=get_type("tolerance", base_config),
    default=get_default("tolerance", base_config),
    help="Value for early stopping tolerance.",
    show_default=True,
)
accumulation_steps = cli_param.option_group.optimization_group.option(
    "--accumulation_steps",
    "-asteps",
    type=get_type("accumulation_steps", base_config),
    default=get_default("accumulation_steps", base_config),
    help="Accumulates gradients during the given number of iterations before performing the weight update "
    "in order to virtually increase the size of the batch.",
    show_default=True,
)
profiler = cli_param.option_group.optimization_group.option(
    "--profiler/--no-profiler",
    default=get_default("profiler", base_config),
    help="Use `--profiler` to enable Pytorch profiler for the first 30 steps after a short warmup. "
    "It will make an execution trace and some statistics about the CPU and GPU usage.",
    show_default=True,
)
track_exp = cli_param.option_group.optimization_group.option(
    "--track_exp",
    "-te",
    type=get_type("track_exp", base_config),
    default=get_default("track_exp", base_config),
    help="Use `--track_exp` to enable wandb/mlflow to track the metric (loss, accuracy, etc...) during the training.",
    show_default=True,
)
# Transfer Learning
transfer_path = cli_param.option_group.transfer_learning_group.option(
    "-tp",
    "--transfer_path",
    type=get_type("transfer_path", base_config),
    default=get_default("transfer_path", base_config),
    help="Path of to a MAPS used for transfer learning.",
    show_default=True,
)
transfer_selection_metric = cli_param.option_group.transfer_learning_group.option(
    "-tsm",
    "--transfer_selection_metric",
    type=get_type("transfer_selection_metric", base_config),
    default=get_default("transfer_selection_metric", base_config),
    help="Metric used to select the model for transfer learning in the MAPS defined by transfer_path.",
    show_default=True,
)
nb_unfrozen_layer = cli_param.option_group.transfer_learning_group.option(
    "-nul",
    "--nb_unfrozen_layer",
    type=get_type("nb_unfrozen_layer", base_config),
    default=get_default("nb_unfrozen_layer", base_config),
    help="Number of layer that will be retrain during training. For example, if it is 2, the last two layers of the model will not be freezed.",
    show_default=True,
)
# Information
emissions_calculator = cli_param.option_group.informations_group.option(
    "--calculate_emissions/--dont_calculate_emissions",
    default=get_default("emissions_calculator", base_config),
    help="Flag to allow calculate the carbon emissions during training.",
    show_default=True,
)
