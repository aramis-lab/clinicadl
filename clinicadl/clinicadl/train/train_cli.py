import click
import toml
import os

from clinicadl.utils import cli_param

current_file_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(
    *os.path.split(current_file_path)[:-1], "resources", "config", "train_config.toml"
)

cmd_name = "train"


@click.command(name=cmd_name)
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
    "--configuration_toml",
    "-c",
    type=click.File(),
    help="Path to the toml file containing all training configuration",
)
@click.option(
    "--label",
    type=str,
    help="Target label to use for training.",
)
@click.option(
    "--gpu/--no-gpu",
    # default=True,
    help="Use GPU by default. Please specify  --no-gpu  to use CPU instead.",
)
@click.option(
    "-np",
    "--nproc",
    type=int,
    # default=2,
    help="Number of cores used during the task.",
)
@click.option(
    "--batch_size",
    # default=2,
    show_default=True,
    type=int,
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
# Model
@click.option(
    "-a",
    "--architecture",
    type=str,
    # default=0,
    help="Architecture of the chosen model to train. A set of model is available in ClinicaDL, default architecture depends on the NETWORK_TASK (see the documentation for more information).",
)
@click.option(
    "--multi",
    type=bool,
    is_flag=True,
    # default=false,
    help="If provided uses a multi-network framework.",
)
# Mode
@click.option(
    "--use_extracted_features",
    type=bool,
    # default=False,
    is_flag=True,
    help="""If provided the outputs of extract preprocessing are used, else the whole
            MRI is loaded.""",
)
# Data
@click.option(
    "--multi_cohort",
    type=bool,
    # default=False,
    is_flag=True,
    help="Performs multi-cohort training. In this case, caps_dir and tsv_path must be paths to TSV files.",
)
@click.option(
    "--diagnoses",
    "-d",
    type=click.Choice(["AD", "BV", "CN", "MCI", "sMCI", "pMCI"]),
    # default=(),
    multiple=True,
    help="List of diagnoses that will be selected for training.",
)
@click.option(
    "--baseline",
    type=bool,
    # default=False,
    is_flag=True,
    help="If provided, only the baseline sessions are used for training.",
)
@click.option(
    "--normalize/--unnormalize",
    # default=False,
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
    help="Sampler choice (random, or weighted for imbalanced datasets).",
)
@click.option(
    "--atlas_weight",
    type=float,
    # default=1,
    help="Weight to put on the MSE loss used to compute the error on atlas intensities.",
)
@click.option(
    "--merged_tsv",
    type=click.File(),
    # default="",
    help="Path to the output of clinica iotools merged-tsv (concatenation for multi-cohort). "
    "Can accelerate training if atlas intensities are predicted.",
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
    help="Train the list of given folds. By default train all folds.",
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
    help="rate of dropout that will be applied to dropout layers in CNN.",
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
    help="Value for the early stopping tolerance.",
)
@click.option(
    "--accumulation_steps",
    "-asteps",
    type=int,
    # default=1,
    help="Accumulates gradients during the given number of iterations before performing the weight update "
    "in order to virtually increase the size of the batch.",
)
# transfert learning
@click.option(
    "-tlp",
    "--transfer_learning_path",
    type=click.Path(),
    # default=0.0,
    help="Path of model used for transfert learning",
)
@click.option(
    "-tls",
    "--transfer_learning_selection",
    type=str,
    # default="best_loss",
    help="Transfert learning selection metric",
)
def cli(
    network_task,
    input_caps_directory,
    preprocessing_json,
    tsv_directory,
    output_maps_directory,
    configuration_toml,
    label,
    use_extracted_features,
    gpu,
    nproc,
    batch_size,
    evaluation_steps,
    architecture,
    multi,
    multi_cohort,
    diagnoses,
    baseline,
    normalize,
    data_augmentation,
    sampler,
    atlas_weight,
    merged_tsv,
    n_splits,
    split,
    epochs,
    learning_rate,
    weight_decay,
    dropout,
    patience,
    tolerance,
    accumulation_steps,
    transfer_learning_path,
    transfer_learning_selection,
):
    """
    Train a deep learning model for NETWORK_TASK on INPUT_CAPS_DIRECTORY data.
    The list of data in loaded from TSV_DIRECTORY.
    Data will be selected with respect to PREPROCESSING_JSON file stored in INPUT_CAPS_DIRECTORY.
    Results will be saved in OUTPUT_MAPS_DIRECTORY.
    """
    from .launch import train

    train_dict = get_train_dict(configuration_toml, preprocessing_json, network_task)

    # Add arguments
    train_dict["network_task"] = network_task
    train_dict["caps_directory"] = input_caps_directory
    train_dict["tsv_path"] = tsv_directory

    # Change value in train dict depending on user provided options
    if label is not None:
        train_dict["label"] = label
    if accumulation_steps is not None:
        train_dict["accumulation_steps"] = accumulation_steps
    if baseline is not None:
        train_dict["baseline"] = baseline
    if batch_size is not None:
        train_dict["batch_size"] = batch_size
    if data_augmentation != ():
        train_dict["data_augmentation"] = data_augmentation
    if diagnoses != ():
        train_dict["diagnoses"] = diagnoses
    if dropout is not None:
        train_dict["dropout"] = dropout
    if epochs is not None:
        train_dict["epochs"] = epochs
    if evaluation_steps is not None:
        train_dict["evaluation_steps"] = evaluation_steps
    if architecture is not None:
        train_dict["architecture"] = architecture
    if multi is not None:
        train_dict["multi"] = multi
    if gpu is not None:
        train_dict["use_cpu"] = not gpu
    if learning_rate is not None:
        train_dict["learning_rate"] = learning_rate
    if multi_cohort is not None:
        train_dict["multi_cohort"] = multi_cohort
    if n_splits is not None:
        train_dict["n_splits"] = n_splits
    if nproc is not None:
        train_dict["num_workers"] = nproc
    if normalize is not None:
        train_dict["unnormalize"] = not normalize
    if patience is not None:
        train_dict["patience"] = patience
    if split != ():
        train_dict["folds"] = split
    if tolerance is not None:
        train_dict["tolerance"] = tolerance
    if transfer_learning_path is not None:
        train_dict["transfer_path"] = transfer_learning_path
    if transfer_learning_selection is not None:
        train_dict["transfer_learning_selection"] = transfer_learning_selection
    if use_extracted_features is not None:
        train_dict["use_extracted_features"] = use_extracted_features
    if weight_decay is not None:
        train_dict["weight_decay"] = weight_decay
    if sampler is not None:
        train_dict["sampler"] = sampler

    # Splits
    if train_dict["n_splits"] > 1:
        train_dict["validation"] = "KFoldSplit"
    else:
        train_dict["validation"] = "SingleSplit"

    # use extracted features
    if "use_extracted_features" in train_dict:
        train_dict["prepare_dl"] = train_dict["use_extracted_features"]
    else:
        train_dict["prepare_dl"] = False

    train(output_maps_directory, train_dict, split)


def get_train_dict(configuration_toml, preprocessing_json, task):
    # read default values
    config_dict = toml.load(config_path)
    # read user specified config
    if configuration_toml is not None:
        user_config = toml.load(configuration_toml)
        for config_section in user_config:
            if config_section not in config_dict:
                raise IOError(f"{config_section} section is not valid in TOML configuration file. Please see the documentation to see the list of option in TOML configuration file")
            for key in config_section:
                if key not in config_dict[config_section]:
                    raise IOError(f"{key} option in {config_section} is not valid in TOML configuration file. Please see the documentation to see the list of option in TOML configuration file")
                config_dict[config_section[key]] = user_config[config_dict[key]]

    # From config file
    train_dict = {
        "accumulation_steps": config_dict["Optimization"]["accumulation_steps"],
        "architecture": config_dict["Model"]["architecture"],
        "baseline": config_dict["Data"]["baseline"],
        "batch_size": config_dict["Computational"]["batch_size"],
        "data_augmentation": config_dict["Data"]["data_augmentation"],
        "diagnoses": config_dict["Data"]["diagnoses"],
        "dropout": config_dict["Architecture"]["dropout"],
        "epochs": config_dict["Optimization"]["epochs"],
        "evaluation_steps": config_dict["Computational"]["evaluation_steps"],
        "learning_rate": config_dict["Optimization"]["learning_rate"],
        "minmaxnormalization": config_dict["Data"]["normalize"],
        "multi": config_dict["Model"]["multi"],
        "multi_cohort": config_dict["Data"]["multi_cohort"],
        "n_splits": config_dict["Cross_validation"]["n_splits"],
        "num_workers": config_dict["Computational"]["n_proc"],
        "patience": config_dict["Optimization"]["patience"],
        "folds": config_dict["Cross_validation"]["split"],
        "tolerance": config_dict["Optimization"]["tolerance"],
        "transfer_path": config_dict["Transfert_learning"]["transfer_path"],
        "transfer_learning_selection": config_dict["Transfert_learning"][
            "transfer_selection_metric"
        ],
        "use_cpu": not config_dict["Computational"]["use_gpu"],
        "weight_decay": config_dict["Optimization"]["weight_decay"],
        "sampler": config_dict["Data"]["sampler"],
    }

    # task dependent
    if task == "classification":
        train_dict["loss"] = config_dict["Classification"]["optimization_metric"]
        train_dict["selection_metrics"] = config_dict["Classification"][
            "selection_metrics"
        ]
        train_dict["label"] = config_dict["Classification"]["label"]
    elif task == "regression":
        train_dict["loss"] = config_dict["Regression"]["optimization_metric"]
        train_dict["selection_metrics"] = config_dict["Regression"]["selection_metrics"]
        train_dict["label"] = config_dict["Regression"]["label"]
    elif task == "reconstruction":
        train_dict["loss"] = config_dict["Reconstruction"]["optimization_metric"]
        train_dict["selection_metrics"] = config_dict["Reconstruction"][
            "selection_metrics"
        ]
    else:
        raise ValueError("Invalid network_task")

    # optimizer
    train_dict["optimizer"] = "Adam"

    # use extracted features
    train_dict["use_extracted_features"] = config_dict["Mode"]["use_extracted_features"]

    # Mode and preprocessing
    from clinicadl.utils.preprocessing import read_preprocessing
    preprocessing_dict = read_preprocessing(preprocessing_json.name)
    train_dict.update(preprocessing_dict)

    return train_dict


if __name__ == "__main__":
    cli()
