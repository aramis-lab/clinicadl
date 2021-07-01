import random

"""
All the architectures are built here
"""


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

    sampling_dict = {
        "accumulation_steps": "randint",
        "atlas_weight": "uniform",
        "baseline": "choice",
        "batch_size": "fixed",
        "caps_dir": "fixed",
        "channels_limit": "fixed",
        "data_augmentation": "fixed",
        "diagnoses": "fixed",
        "dropout": "uniform",
        "epochs": "fixed",
        "evaluation_steps": "fixed",
        "learning_rate": "exponent",
        "loss": "choice",
        "merged_tsv_path": "fixed",
        "mode": "choice",
        "multi_cohort": "fixed",
        "n_fcblocks": "randint",
        "n_splits": "fixed",
        "nproc": "fixed",
        "network_type": "choice",
        "network_normalization": "choice",
        "optimizer": "choice",
        "patience": "fixed",
        "preprocessing": "choice",
        "predict_atlas_intensities": "fixed",
        "sampler": "choice",
        "split": "fixed",
        "tolerance": "fixed",
        "transfer_learning_path": "choice",
        "transfer_learning_selection": "choice",
        "tsv_path": "fixed",
        "unnormalize": "choice",
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
            "use_extracted_patches": "fixed",
        },
        "roi": {
            "selection_threshold": "uniform",
            "roi_list": "fixed",
            "use_extracted_roi": "fixed",
            "uncropped_roi": "fixed",
        },
        "slice": {
            "discarded_slices": "randint",
            "selection_threshold": "uniform",
            "slice_direction": "choice",
            "use_extracted_slices": "fixed",
        },
    }

    for name, sampling_type in sampling_dict.items():
        sampled_value = sampling_fn(getattr(rs_options, name), sampling_type)
        setattr(options, name, sampled_value)

    if options.mode not in additional_mode_dict.keys():
        raise NotImplementedError(
            "Mode %s was not correctly implemented for random search" % options.mode
        )

    additional_dict = additional_mode_dict[options.mode]
    for name, sampling_type in additional_dict.items():
        sampled_value = sampling_fn(getattr(rs_options, name), sampling_type)
        setattr(options, name, sampled_value)

    # Exceptions to classical sampling functions
    if not options.wd_bool:
        options.weight_decay = 0

    options.evaluation_steps = find_evaluation_steps(
        options.accumulation_steps, goal=options.evaluation_steps
    )
    options.convolutions_dict = random_conv_sampling(rs_options)

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
    n_convblocks = sampling_fn(rs_options.n_convblocks, "randint")
    first_conv_width = sampling_fn(rs_options.first_conv_width, "choice")
    d_reduction = sampling_fn(rs_options.d_reduction, "choice")

    # Sampling the parameters of each convolutional block
    convolutions = dict()
    current_in_channels = None
    current_out_channels = first_conv_width
    for i in range(n_convblocks):
        conv_dict = dict()
        conv_dict["in_channels"] = current_in_channels
        conv_dict["out_channels"] = current_out_channels

        current_in_channels, current_out_channels = update_channels(
            current_out_channels, rs_options.channels_limit
        )
        conv_dict["n_conv"] = sampling_fn(rs_options.n_conv, "choice")
        conv_dict["d_reduction"] = d_reduction
        convolutions["conv" + str(i)] = conv_dict

    return convolutions


def update_channels(out_channels, channels_limit=512):
    in_channels = out_channels
    if out_channels < channels_limit:
        out_channels = 2 * out_channels

    return in_channels, out_channels
