import random
from os import path
from typing import Any, Dict

from clinicadl.train.train_utils import get_train_dict


def get_space_dict(toml_options: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Transforms the TOML dictionary in one dimension dictionary."""
    if "Random_Search" not in toml_options:
        raise ValueError(
            "Category 'Random_Search' must be defined in the random_search.toml file. "
            "All random search arguments AND options must be defined in this category."
        )

    space_dict = dict()
    for key in toml_options["Random_Search"]:
        space_dict[key] = toml_options["Random_Search"][key]

    # Check presence of mandatory arguments
    mandatory_arguments = [
        "network_task",
        "tsv_path",
        "caps_directory",
        "preprocessing_json",
        "n_convblocks",
        "first_conv_width",
        "n_fcblocks",
    ]

    for argument in mandatory_arguments:
        if argument not in space_dict:
            raise ValueError(
                f"The argument {argument} must be specified in the random_search.toml file (Random_Search category)."
            )

    # Default of specific options of random search
    random_search_specific_options = {
        "d_reduction": "MaxPooling",
        "network_normalization": "BatchNorm",
        "channels_limit": 512,
        "n_conv": 1,
        "wd_bool": True,
    }

    for option, value in random_search_specific_options.items():
        if option not in space_dict:
            space_dict[option] = value

    del toml_options["Random_Search"]

    preprocessing_json = path.join(
        space_dict["caps_directory"],
        "tensor_extraction",
        space_dict.pop("preprocessing_json"),
    )
    train_default = get_train_dict(
        toml_options, preprocessing_json, space_dict["network_task"]
    )
    space_dict.update(train_default)

    return space_dict


def sampling_fn(value, sampling_type):
    if isinstance(value, (tuple, list)):
        if sampling_type is "fixed":
            return value
        elif sampling_type is "choice":
            return random.choice(value)
        elif sampling_type is "exponent":
            exponent = random.uniform(*value)
            return 10 ** -exponent
        elif sampling_type is "randint":
            return random.randint(*value)
        elif sampling_type is "uniform":
            return random.uniform(*value)
        else:
            raise ValueError("Sampling type %s is not implemented" % sampling_type)
    else:
        if sampling_type is "exponent":
            return 10 ** -value
        else:
            return value


def random_sampling(rs_options, options):
    """
    Samples all the hyperparameters of the model.
    Args:
        rs_options: (Namespace) parameters of the random search
        options: (Namespace) options of the training
    Returns:
        options (Namespace), options updated to train the model generated randomly
    """
    print(rs_options)

    sampling_dict = {
        "accumulation_steps": "randint",
        "baseline": "choice",
        "batch_size": "fixed",
        "caps_directory": "fixed",
        "channels_limit": "fixed",
        "compensation": "fixed",
        "data_augmentation": "fixed",
        "deterministic": "fixed",
        "diagnoses": "fixed",
        "dropout": "uniform",
        "epochs": "fixed",
        "evaluation_steps": "fixed",
        "folds": "fixed",
        "label": "fixed",
        "learning_rate": "exponent",
        "minmaxnormalization": "choice",
        "mode": "choice",
        "multi_cohort": "fixed",
        "multi_network": "choice",
        "n_fcblocks": "randint",
        "n_splits": "fixed",
        "num_workers": "fixed",
        "network_task": "fixed",
        "network_normalization": "choice",
        "optimizer": "choice",
        "patience": "fixed",
        "preprocessing": "choice",
        "seed": "fixed",
        "selection_metrics": "fixed",
        "sampler": "choice",
        "tolerance": "fixed",
        "transfer_path": "choice",
        "transfer_selection_metric": "choice",
        "tsv_path": "fixed",
        "use_cpu": "fixed",
        "wd_bool": "choice",
        "weight_decay": "exponent",
    }

    additional_mode_dict = {
        "image": {},
        "patch": {
            "patch_size": "randint",
            "selection_threshold": "uniform",
            "stride_size": "randint",
            "use_extracted_features": "fixed",
        },
        "roi": {
            "selection_threshold": "uniform",
            "roi_list": "fixed",
            "use_extracted_features": "fixed",
            "uncropped_roi": "fixed",
        },
        "slice": {
            "discarded_slices": "randint",
            "selection_threshold": "uniform",
            "slice_direction": "choice",
            "use_extracted_features": "fixed",
        },
    }

    for name, sampling_type in sampling_dict.items():
        sampled_value = sampling_fn(rs_options[name], sampling_type)
        options[name] = sampled_value

    if options["mode"] not in additional_mode_dict.keys():
        raise NotImplementedError(
            "Mode %s was not correctly implemented for random search" % options.mode
        )

    additional_dict = additional_mode_dict[options["mode"]]
    for name, sampling_type in additional_dict.items():
        sampled_value = sampling_fn(rs_options[name], sampling_type)
        options[name] = sampled_value

    # Exceptions to classical sampling functions
    if not options["wd_bool"]:
        options["weight_decay"] = 0

    options["evaluation_steps"] = find_evaluation_steps(
        options["accumulation_steps"], goal=options["evaluation_steps"]
    )
    options["convolutions_dict"] = random_conv_sampling(rs_options)

    # Hard-coded options
    if options["n_splits"] and options["n_splits"] > 1:
        options["validation"] = "KFoldSplit"
    else:
        options["validation"] = "SingleSplit"
    if "use_extracted_features" in options:
        options["prepare_dl"] = options["use_extracted_features"]
    else:
        options["prepare_dl"] = False
    options["optimizer"] = "Adam"

    return options


def find_evaluation_steps(accumulation_steps, goal=18):
    """
    Compute the evaluation steps to be a multiple of accumulation steps as close possible as the goal.
    Args:
        accumulation_steps: (int) number of times the gradients are accumulated before parameters update.
    Returns:
        (int) number of evaluation_steps
    """
    if goal == 0 or goal % accumulation_steps == 0:
        return goal
    else:
        return (goal // accumulation_steps + 1) * accumulation_steps


def random_conv_sampling(rs_options):
    """
    Generate random parameters for a random architecture (convolutional part).
    Args:
        rs_options: (Namespace) parameters of the random search
    Returns
        (dict) parameters of the architecture
    """
    n_convblocks = sampling_fn(rs_options["n_convblocks"], "randint")
    first_conv_width = sampling_fn(rs_options["first_conv_width"], "choice")
    d_reduction = sampling_fn(rs_options["d_reduction"], "choice")

    # Sampling the parameters of each convolutional block
    convolutions = dict()
    current_in_channels = None
    current_out_channels = first_conv_width
    for i in range(n_convblocks):
        conv_dict = dict()
        conv_dict["in_channels"] = current_in_channels
        conv_dict["out_channels"] = current_out_channels

        current_in_channels, current_out_channels = update_channels(
            current_out_channels, rs_options["channels_limit"]
        )
        conv_dict["n_conv"] = sampling_fn(rs_options["n_conv"], "choice")
        conv_dict["d_reduction"] = d_reduction
        convolutions["conv" + str(i)] = conv_dict

    return convolutions


def update_channels(out_channels, channels_limit=512):
    in_channels = out_channels
    if out_channels < channels_limit:
        out_channels = 2 * out_channels

    return in_channels, out_channels
