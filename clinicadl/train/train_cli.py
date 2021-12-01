import os
from logging import getLogger

import click
import toml

from clinicadl.utils import cli_param
from clinicadl.utils.caps_dataset.data import CapsDataset


@click.command(name="train")
@click.argument(
    "network_task",
    type=click.Choice(["classification", "regression", "reconstruction"]),
)
@cli_param.argument.caps_directory
@cli_param.argument.preprocessing_json
@click.argument(
    "tsv_directory",
    type=click.Path(exists=True),
)
@cli_param.argument.output_maps
# train option
@click.option(
    "--config_file",
    "-c",
    type=click.File(),
    help="Path to the TOML file containing the values of the options needed for training.",
)
@click.option(
    "--label",
    type=str,
    help="Target label used for training (if NETWORK_TASK in [`regression`, `classification`]).",
)
# Computational
@click.option(
    "--gpu/--no-gpu",
    type=bool,
    default=None,
    help="Use GPU by default. Please specify `--no-gpu` to force using CPU.",
)
@click.option(
    "-np",
    "--n_proc",
    type=int,
    # default=2,
    help="Number of cores used during the task.",
)
@click.option(
    "--batch_size",
    type=int,
    # default=2,
    help="Batch size for data loading.",
)
@click.option(
    "--evaluation_steps",
    "-esteps",
    type=int,
    # default=0,
    help="Fix the number of iterations to perform before computing an evaluation. Default will only "
    "perform one evaluation at the end of each epoch.",
)
# Reproducibility
@click.option(
    "--seed",
    help="Value to set the seed for all random operations."
    "Default will sample a random value for the seed.",
    # default=None,
    type=int,
)
@click.option(
    "--nondeterministic/--deterministic",
    type=bool,
    default=None,
    help="Forces Pytorch to be deterministic even when using a GPU. "
    "Will raise a RuntimeError if a non-deterministic function is encountered.",
)
@click.option(
    "--compensation",
    help="Allow the user to choose how CUDA will compensate the deterministic behaviour.",
    # default="memory",
    type=click.Choice(["memory", "time"]),
)
# Model
@click.option(
    "-a",
    "--architecture",
    type=str,
    # default=0,
    help="Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).",
)
@click.option(
    "--multi_network/--single_network",
    type=bool,
    default=None,
    help="If provided uses a multi-network framework.",
)
# Mode
@click.option(
    "--selection_threshold",
    type=float,
    # default=0,
    help="""Selection threshold for soft-voting when network_task is 'classification'. 

    Will only be used if num_networks > 1.""",
)
# Data
@click.option(
    "--multi_cohort/--single_cohort",
    type=bool,
    default=None,
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
)
@click.option(
    "--diagnoses",
    "-d",
    type=str,
    # default=(),
    multiple=True,
    help="List of diagnoses used for training.",
)
@click.option(
    "--baseline/--longitudinal",
    type=bool,
    default=None,
    help="If provided, only the baseline sessions are used for training.",
)
@click.option(
    "--normalize/--unnormalize",
    type=bool,
    default=None,
    help="Disable default MinMaxNormalization.",
)
@click.option(
    "--data_augmentation",
    "-da",
    type=click.Choice(["None", "Noise", "Erasing", "CropPad", "Smoothing"]),
    # default=(),
    multiple=True,
    help="Randomly applies transforms on the training set.",
)
@click.option(
    "--sampler",
    "-s",
    type=click.Choice(["random", "weighted"]),
    # default="random",
    help="Sampler used to load the training data set.",
)
# Cross validation
@click.option(
    "--n_splits",
    type=int,
    # default=0,
    help="If a value is given for k will load data of a k-fold CV. "
    "Default value (0) will load a single split.",
)
@click.option(
    "--split",
    "-s",
    type=int,
    # default=(),
    multiple=True,
    help="Train the list of given folds. By default, all the folds are trained.",
)
# Optimization
@click.option(
    "--epochs",
    type=int,
    # default=20,
    help="Maximum number of epochs.",
)
@click.option(
    "--learning_rate",
    "-lr",
    type=float,
    # default=1e-4,
    help="Learning rate of the optimization.",
)
@click.option(
    "--weight_decay",
    "-wd",
    type=float,
    # default=1e-4,
    help="Weight decay value used in optimization.",
)
@click.option(
    "--dropout",
    type=float,
    # default=0,
    help="Rate value applied to dropout layers in a CNN architecture.",
)
@click.option(
    "--patience",
    type=int,
    # default=0,
    help="Number of epochs for early stopping patience.",
)
@click.option(
    "--tolerance",
    type=float,
    # default=0.0,
    help="Value for early stopping tolerance.",
)
@click.option(
    "--accumulation_steps",
    "-asteps",
    type=int,
    # default=1,
    help="Accumulates gradients during the given number of iterations before performing the weight update "
    "in order to virtually increase the size of the batch.",
)
# transfer learning
@click.option(
    "-tp",
    "--transfer_path",
    type=click.Path(),
    # default=0.0,
    help="Path of to a MAPS used for transfer learning.",
)
@click.option(
    "-tsm",
    "--transfer_selection_metric",
    type=str,
    # default="loss",
    help="Metric used to select the model for transfer learning in the MAPS defined by transfer_path.",
)
def cli(
    network_task,
    caps_directory,
    preprocessing_json,
    tsv_directory,
    output_maps_directory,
    config_file,
    label,
    selection_threshold,
    gpu,
    n_proc,
    batch_size,
    evaluation_steps,
    seed,
    nondeterministic,
    compensation,
    architecture,
    multi_network,
    multi_cohort,
    diagnoses,
    baseline,
    normalize,
    data_augmentation,
    sampler,
    n_splits,
    split,
    epochs,
    learning_rate,
    weight_decay,
    dropout,
    patience,
    tolerance,
    accumulation_steps,
    transfer_path,
    transfer_selection_metric,
):
    """Train a deep learning model on your neuroimaging dataset.

    NETWORK_TASK is the task learnt by the network [classification|regression|reconstruction]

    CAPS_DIRECTORY is the CAPS folder from where tensors will be loaded.

    PREPROCESSING_JSON is the name of the JSON file in CAPS_DIRECTORY/tensor_extraction folder where
    all information about extraction are stored in order to read the wanted tensors.

    TSV_DIRECTORY is a folder were TSV files defining train and validation sets are stored.

    OUTPUT_MAPS_DIRECTORY is the path to the MAPS folder where outputs and results will be saved.

    This pipeline includes many options. To make its usage easier, you can write all the configuration
    in a TOML file as explained in the documentation:
    https://clinicadl.readthedocs.io/en/stable/Train/Introduction/#configuration-file
    """
    from clinicadl.utils.cmdline_utils import check_gpu

    from .launch import train
    from .train_utils import get_train_dict

    logger = getLogger("clinicadl")

    if not multi_cohort:
        preprocessing_json = os.path.join(
            caps_directory, "tensor_extraction", preprocessing_json
        )
    else:
        caps_dict = CapsDataset.create_caps_dict(caps_directory, multi_cohort)
        json_found = False
        for caps_name, caps_path in caps_dict.items():
            if os.path.exists(
                os.path.join(caps_path, "tensor_extraction", preprocessing_json)
            ):
                preprocessing_json = os.path.join(
                    caps_path, "tensor_extraction", preprocessing_json
                )
                logger.info(
                    f"Preprocessing JSON {preprocessing_json} found in CAPS {caps_name}."
                )
                json_found = True
        if not json_found:
            raise ValueError(
                f"Preprocessing JSON {preprocessing_json} was not found for any CAPS "
                f"in {caps_dict}."
            )

    user_dict = None
    if config_file:
        user_dict = toml.load(config_file)
    train_dict = get_train_dict(user_dict, preprocessing_json, network_task)

    # Add arguments
    train_dict["network_task"] = network_task
    train_dict["caps_directory"] = caps_directory
    train_dict["tsv_path"] = tsv_directory

    # Change value in train dict depending on user provided options
    standard_options_list = [
        "label",
        "accumulation_steps",
        "baseline",
        "batch_size",
        "data_augmentation",
        "diagnoses",
        "dropout",
        "epochs",
        "evaluation_steps",
        "architecture",
        "multi_network",
        "learning_rate",
        "multi_cohort",
        "n_splits",
        "patience",
        "tolerance",
        "transfer_selection_metric",
        "selection_threshold",
        "weight_decay",
        "sampler",
        "seed",
        "compensation",
        "transfer_path",
    ]

    for option in standard_options_list:
        if (eval(option) is not None and not isinstance(eval(option), tuple)) or (
            isinstance(eval(option), tuple) and len(eval(option)) != 0
        ):
            train_dict[option] = eval(option)

    if gpu is not None:
        train_dict["use_cpu"] = not gpu
    if not train_dict["use_cpu"]:
        check_gpu()
    if n_proc is not None:
        train_dict["num_workers"] = n_proc
    if normalize is not None:
        train_dict["minmaxnormalization"] = normalize
    if split:
        train_dict["folds"] = split
    if nondeterministic:
        train_dict["deterministic"] = not nondeterministic

    # Splits
    if train_dict["n_splits"] and train_dict["n_splits"] > 1:
        train_dict["validation"] = "KFoldSplit"
    else:
        train_dict["validation"] = "SingleSplit"

    train(output_maps_directory, train_dict, train_dict.pop("folds"))


if __name__ == "__main__":
    cli()
