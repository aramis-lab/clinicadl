import click

from clinicadl.utils import cli_param

caps_directory = cli_param.argument.caps_directory
preprocessing_json = cli_param.argument.preprocessing_json
tsv_directory = click.argument(
    "tsv_directory",
    type=click.Path(exists=True),
)
output_maps = cli_param.argument.output_maps
# train option
config_file = click.option(
    "--config_file",
    "-c",
    type=click.Path(exists=True),
    help="Path to the TOML or JSON file containing the values of the options needed for training.",
)
# Computational
gpu = cli_param.option_group.computational_group.option(
    "--gpu/--no-gpu",
    type=bool,
    default=None,
    help="Use GPU by default. Please specify `--no-gpu` to force using CPU.",
)
n_proc = cli_param.option_group.computational_group.option(
    "-np",
    "--n_proc",
    type=int,
    # default=2,
    help="Number of cores used during the task.",
)
batch_size = cli_param.option_group.computational_group.option(
    "--batch_size",
    type=int,
    # default=2,
    help="Batch size for data loading.",
)
evaluation_steps = cli_param.option_group.computational_group.option(
    "--evaluation_steps",
    "-esteps",
    type=int,
    # default=0,
    help="Fix the number of iterations to perform before computing an evaluation. Default will only "
    "perform one evaluation at the end of each epoch.",
)
fully_sharded_data_parallel = cli_param.option_group.computational_group.option(
    "--fully_sharded_data_parallel",
    "-fsdp",
    type=bool,
    is_flag=True,
    help="Enables Fully Sharded Data Parallel with Pytorch to save memory at the cost of communications. "
    "Currently this only enables ZeRO Stage 1 but will be entirely replaced by FSDP in a later patch, "
    "this flag is already set to FSDP to that the zero flag is never actually removed.",
    default=False,
)

amp = cli_param.option_group.computational_group.option(
    "--amp/--no-amp",
    type=bool,
    help="Enables automatic mixed precision during training and inference.",
)
# Reproducibility
seed = cli_param.option_group.reproducibility_group.option(
    "--seed",
    help="Value to set the seed for all random operations."
    "Default will sample a random value for the seed.",
    # default=None,
    type=int,
)
deterministic = cli_param.option_group.reproducibility_group.option(
    "--deterministic/--nondeterministic",
    type=bool,
    default=None,
    help="Forces Pytorch to be deterministic even when using a GPU. "
    "Will raise a RuntimeError if a non-deterministic function is encountered.",
)
compensation = cli_param.option_group.reproducibility_group.option(
    "--compensation",
    help="Allow the user to choose how CUDA will compensate the deterministic behaviour.",
    # default="memory",
    type=click.Choice(["memory", "time"]),
)
save_all_models = cli_param.option_group.reproducibility_group.option(
    "--save_all_models/--save_only_best_model",
    type=bool,
    help="If provided, enables the saving of models weights for each epochs.",
)

# Model
architecture = cli_param.option_group.model_group.option(
    "-a",
    "--architecture",
    type=str,
    # default=0,
    help="Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).",
)
multi_network = cli_param.option_group.model_group.option(
    "--multi_network/--single_network",
    type=bool,
    default=None,
    help="If provided uses a multi-network framework.",
)
ssda_network = cli_param.option_group.model_group.option(
    "--ssda_network/--single_network",
    type=bool,
    default=None,
    help="If provided uses a ssda-network framework.",
)
# Task
label = cli_param.option_group.task_group.option(
    "--label",
    type=str,
    help="Target label used for training.",
)
selection_metrics = cli_param.option_group.task_group.option(
    "--selection_metrics",
    "-sm",
    multiple=True,
    help="""Allow to save a list of models based on their selection metric. Default will
    only save the best model selected on loss.""",
)
selection_threshold = cli_param.option_group.task_group.option(
    "--selection_threshold",
    type=float,
    # default=0,
    help="""Selection threshold for soft-voting. Will only be used if num_networks > 1.""",
)
classification_loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(["CrossEntropyLoss", "MultiMarginLoss"]),
    help="Loss used by the network to optimize its training task.",
)
regression_loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(
        [
            "L1Loss",
            "MSELoss",
            "KLDivLoss",
            "BCEWithLogitsLoss",
            "HuberLoss",
            "SmoothL1Loss",
        ]
    ),
    help="Loss used by the network to optimize its training task.",
)
reconstruction_loss = cli_param.option_group.task_group.option(
    "--loss",
    "-l",
    type=click.Choice(
        [
            "L1Loss",
            "MSELoss",
            "KLDivLoss",
            "BCEWithLogitsLoss",
            "HuberLoss",
            "SmoothL1Loss",
        ]
    ),
    help="Loss used by the network to optimize its training task.",
)
# Data
multi_cohort = cli_param.option_group.data_group.option(
    "--multi_cohort/--single_cohort",
    type=bool,
    default=None,
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
)
diagnoses = cli_param.option_group.data_group.option(
    "--diagnoses",
    "-d",
    type=str,
    # default=(),
    multiple=True,
    help="List of diagnoses used for training.",
)
baseline = cli_param.option_group.data_group.option(
    "--baseline/--longitudinal",
    type=bool,
    default=None,
    help="If provided, only the baseline sessions are used for training.",
)
normalize = cli_param.option_group.data_group.option(
    "--normalize/--unnormalize",
    type=bool,
    default=None,
    help="Disable default MinMaxNormalization.",
)
data_augmentation = cli_param.option_group.data_group.option(
    "--data_augmentation",
    "-da",
    type=click.Choice(
        [
            "None",
            "Noise",
            "Erasing",
            "CropPad",
            "Smoothing",
            "Motion",
            "Ghosting",
            "Spike",
            "BiasField",
            "RandomBlur",
            "RandomSwap",
        ]
    ),
    # default=(),
    multiple=True,
    help="Randomly applies transforms on the training set.",
)
sampler = cli_param.option_group.data_group.option(
    "--sampler",
    "-s",
    type=click.Choice(["random", "weighted"]),
    # default="random",
    help="Sampler used to load the training data set.",
)
caps_target = cli_param.option_group.data_group.option(
    "--caps_target",
    "-d",
    type=str,
    default=None,
    help="CAPS of target data.",
)
tsv_target_lab = cli_param.option_group.data_group.option(
    "--tsv_target_lab",
    "-d",
    type=str,
    default=None,
    help="TSV of labeled target data.",
)
tsv_target_unlab = cli_param.option_group.data_group.option(
    "--tsv_target_unlab",
    "-d",
    type=str,
    default=None,
    help="TSV of unllabeled target data.",
)
preprocessing_dict_target = cli_param.option_group.data_group.option(
    "--preprocessing_dict_target",
    "-d",
    type=str,
    default=None,
    help="Path to json taget.",
)
# Cross validation
n_splits = cli_param.option_group.cross_validation.option(
    "--n_splits",
    type=int,
    # default=0,
    help="If a value is given for k will load data of a k-fold CV. "
    "Default value (0) will load a single split.",
)
split = cli_param.option_group.cross_validation.option(
    "--split",
    "-s",
    type=int,
    # default=(),
    multiple=True,
    help="Train the list of given splits. By default, all the splits are trained.",
)
# Optimization
optimizer = cli_param.option_group.optimization_group.option(
    "--optimizer",
    type=click.Choice(
        [
            "Adadelta",
            "Adagrad",
            "Adam",
            "AdamW",
            "Adamax",
            "ASGD",
            "NAdam",
            "RAdam",
            "RMSprop",
            "SGD",
        ]
    ),
    help="Optimizer used to train the network.",
)
epochs = cli_param.option_group.optimization_group.option(
    "--epochs",
    type=int,
    # default=20,
    help="Maximum number of epochs.",
)
learning_rate = cli_param.option_group.optimization_group.option(
    "--learning_rate",
    "-lr",
    type=float,
    # default=1e-4,
    help="Learning rate of the optimization.",
)
adaptive_learning_rate = cli_param.option_group.optimization_group.option(
    "--adaptive_learning_rate",
    "-alr",
    type=bool,
    help="Whether to diminish the learning rate",
    is_flag=True,
    default=False,
)
weight_decay = cli_param.option_group.optimization_group.option(
    "--weight_decay",
    "-wd",
    type=float,
    # default=1e-4,
    help="Weight decay value used in optimization.",
)
dropout = cli_param.option_group.optimization_group.option(
    "--dropout",
    type=float,
    # default=0,
    help="Rate value applied to dropout layers in a CNN architecture.",
)
patience = cli_param.option_group.optimization_group.option(
    "--patience",
    type=int,
    # default=0,
    help="Number of epochs for early stopping patience.",
)
tolerance = cli_param.option_group.optimization_group.option(
    "--tolerance",
    type=float,
    # default=0.0,
    help="Value for early stopping tolerance.",
)
accumulation_steps = cli_param.option_group.optimization_group.option(
    "--accumulation_steps",
    "-asteps",
    type=int,
    # default=1,
    help="Accumulates gradients during the given number of iterations before performing the weight update "
    "in order to virtually increase the size of the batch.",
)
profiler = cli_param.option_group.optimization_group.option(
    "--profiler/--no-profiler",
    type=bool,
    help="Use `--profiler` to enable Pytorch profiler for the first 30 steps after a short warmup. "
    "It will make an execution trace and some statistics about the CPU and GPU usage.",
)
track_exp = cli_param.option_group.optimization_group.option(
    "--track_exp",
    "-te",
    type=click.Choice(
        [
            "wandb",
            "mlflow",
            "",
        ]
    ),
    help="Use `--track_exp` to enable wandb/mlflow to track the metric (loss, accuracy, etc...) during the training.",
)
# transfer learning
transfer_path = cli_param.option_group.transfer_learning_group.option(
    "-tp",
    "--transfer_path",
    type=click.Path(),
    # default=0.0,
    help="Path of to a MAPS used for transfer learning.",
)
transfer_selection_metric = cli_param.option_group.transfer_learning_group.option(
    "-tsm",
    "--transfer_selection_metric",
    type=str,
    # default="loss",
    help="Metric used to select the model for transfer learning in the MAPS defined by transfer_path.",
)
nb_unfrozen_layer = cli_param.option_group.transfer_learning_group.option(
    "-nul",
    "--nb_unfrozen_layer",
    type=int,
    default=0,
    help="Number of layer that will be retrain during training. For example, if it is 2, the last two layers of the model will not be freezed.",
)
# informations
emissions_calculator = cli_param.option_group.informations_group.option(
    "--calculate_emissions/--dont_calculate_emissions",
    type=bool,
    default=None,
    help="Flag to allow calculate the carbon emissions during training.",
)
