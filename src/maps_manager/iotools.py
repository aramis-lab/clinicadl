# coding: utf8
import logging
from pathlib import Path, PosixPath

from clinicadl.utils.exceptions import ClinicaDLArgumentError

LOG_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]


computational_list = ["gpu", "batch_size", "n_proc", "evaluation_steps"]


def write_requirements_version(output_path: Path):
    import subprocess
    from warnings import warn

    try:
        env_variables = subprocess.check_output("pip freeze", shell=True).decode(
            "utf-8"
        )
        with (output_path / "environment.txt").open(mode="w") as file:
            file.write(env_variables)
    except subprocess.CalledProcessError:
        warn(
            "You do not have the right to execute pip freeze. Your environment will not be written"
        )


def check_and_clean(directory_path: Path):
    import shutil

    if directory_path.is_dir():
        shutil.rmtree(directory_path)
    directory_path.mkdir(parents=True)


def commandline_to_json(commandline, logger=None, filename="commandline.json"):
    """
    This is a function to write the python argparse object into a json file.
    This helps for DL when searching for hyperparameters
    Args:
        commandline: (dict) dictionary with all the command line options values.
        logger: (logging object) writer to stdout and stderr.
        filename: (str) name of the JSON file.

    :return:
    """
    if logger is None:
        logger = logging

    import json
    from copy import copy

    if isinstance(commandline, dict):
        commandline_arg_dict = copy(commandline)
    else:
        commandline_arg_dict = copy(vars(commandline))
    output_dir = commandline_arg_dict["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # remove these entries from the commandline log file
    remove_list = ["func", "output_dir", "launch_dir", "name", "verbose", "logname"]
    for variable in remove_list:
        if variable in commandline_arg_dict:
            del commandline_arg_dict[variable]

    # change path to str for json.dumps
    for key, value in commandline_arg_dict.items():
        if isinstance(value, PosixPath):
            commandline_arg_dict[key] = value.as_posix()

    # save to json file
    json = json.dumps(commandline_arg_dict, skipkeys=True, ensure_ascii=False, indent=4)
    logger.info(f"Path of json file: {output_dir / 'commandline.json'}")
    f = open(output_dir / filename, "w")
    f.write(json)
    f.close()


def read_json(
    options=None, json_path: Path = None, test=False, read_computational=False
):
    """
    Read a json file to update options dictionary.
    Ensures retro-compatibility with previous namings in clinicadl.

    Args:
        options: (dict) options of the model.
        json_path: (str) If given path to the json file, else found with options.model_path.
        test: (bool) If given the reader will ignore some options specific to data.
        read_computational: (bool) if set to True, the computational arguments are also read.
    Returns:
        options (dict) options of the model updated
    """
    import json

    if options is None:
        options = {}

    evaluation_parameters = ["diagnosis_path", "input_dir", "diagnoses"]
    prep_compatibility_dict = {"mni": "t1-extensive", "linear": "t1-linear"}
    if json_path is None:
        json_path = options["model_path"] / "commandline.json"

    with json_path.open(mode="r") as f:
        json_data = json.load(f)

    for key, item in json_data.items():
        # We do not change computational options
        if key in computational_list and not read_computational:
            pass
        # If used for evaluation, some parameters were already given
        if test and key in evaluation_parameters:
            pass
        else:
            options[key] = item

    # Retro-compatibility with runs of previous versions
    if "network" in options:
        options["model"] = options["network"]
        del options["network"]

    if "discarded_slices" not in options:
        options["discarded_slices"] = 20

    if isinstance(options["preprocessing"], str):
        if options["preprocessing"] in prep_compatibility_dict.keys():
            options["preprocessing"] = prep_compatibility_dict[options["preprocessing"]]

    if "mri_plane" in options:
        options["slice_direction"] = options["mri_plane"]
        del options["mri_plane"]

    if "hippocampus_roi" in options:
        if options["hippocampus_roi"]:
            options["mode"] = "roi"
            del options["hippocampus_roi"]

    if "pretrained_path" in options:
        options["transfer_learning_path"] = options["pretrained_path"]
        del options["pretrained_path"]

    if "pretrained_difference" in options:
        options["transfer_learning_difference"] = options["pretrained_difference"]
        del options["pretrained_difference"]

    if "patch_stride" in options:
        options["stride_size"] = options["patch_stride"]

    if "use_gpu" in options:
        options["gpu"] = options["use_gpu"]

    if "mode" in options:
        if options["mode"] == "subject":
            options["mode"] = "image"
        if options["mode"] == "slice" and "network_type" not in options:
            options["network_type"] = "cnn"
        if options["mode"] == "patch" and "network_type" in options:
            if options["network_type"] == "multi":
                options["network_type"] = "multicnn"

    if "network_type" not in options:
        if "mode_task" in options:
            options["network_type"] = options["mode_task"]
        elif "train_autoencoder" in options:
            options["network_type"] = "autoencoder"
        else:
            options["network_type"] = "cnn"

    if "selection" in options:
        options["transfer_learning_selection"] = options["selection"]

    if "loss" not in options:
        options["loss"] = "default"

    if "dropout" not in options or options["dropout"] is None:
        options["dropout"] = 0

    if "uncropped_roi" not in options:
        options["uncropped_roi"] = False

    if "roi_list" not in options:
        options["roi_list"] = None

    if "multi_cohort" not in options:
        options["multi_cohort"] = False

    if "predict_atlas_intensities" not in options:
        options["predict_atlas_intensities"] = None

    if "merged_tsv_path" not in options:
        options["merged_tsv_path"] = None

    if "atlas_weight" not in options:
        options["atlas_weight"] = 1

    if "n_splits" in options and options["n_splits"] is None:
        options["n_splits"] = 0

    if not hasattr(options, "seed"):
        options.seed = None

    if not hasattr(options, "deterministic"):
        options.deterministic = False

    if not hasattr(options, "compensation"):
        options.compensation = "memory"

    return options


def check_and_complete(options, random_search=False):
    """
    This function initializes missing fields with missing values.
    Some fields are mandatory and cannot be initialized by default; this will raise an issue if they are missing.

    Args:
        options: (dict) the options used for training.
        random_search: (bool) If True the options are looking for mandatory values of random-search.
    """

    def set_default(params_dict, default_dict):
        for name, default_value in default_dict.items():
            if name not in params_dict:
                params_dict[name] = default_value

    default_values = {
        "accumulation_steps": 1,
        "baseline": False,
        "batch_size": 2,
        "compensation": "memory",
        "data_augmentation": False,
        "diagnoses": ["AD", "CN"],
        "dropout": 0,
        "epochs": 20,
        "evaluation_steps": 0,
        "learning_rate": 4,
        "loss": "default",
        "multi": False,
        "multi_cohort": False,
        "n_splits": 0,
        "n_proc": 2,
        "optimizer": "Adam",
        "unnormalize": False,
        "patience": 0,
        "predict_atlas_intensities": [],
        "split": [],
        "seed": None,
        "selection_metrics": ["loss"],
        "tolerance": 0.0,
        "deterministic": False,
        "transfer_learning_path": "",
        "transfer_learning_selection": "best_loss",
        "gpu": True,
        "wd_bool": True,
        "weight_decay": 4,
        "sampler": "random",
    }
    mode_default_values = {
        "patch": {
            "patch_size": 50,
            "stride_size": 50,
            "use_extracted_patches": False,
        },
        "roi": {
            "roi_list": [],
            "uncropped_roi": False,
            "use_extracted_roi": False,
        },
        "slice": {
            "discarded_slices": 20,
            "slice_direction": 0,
            "use_extracted_slices": False,
        },
        "image": {},
    }

    task_default_values = {
        "classification": {
            "label": "diagnosis",
            "selection_threshold": 0,
        },
        "regression": {
            "label": "age",
        },
    }
    if random_search:
        default_values["d_reduction"] = "MaxPooling"
        default_values["network_normalization"] = "BatchNorm"
        default_values["channels_limit"] = 512
        default_values["n_conv"] = 1

    set_default(options, default_values)

    mandatory_arguments = [
        "network_task",
        "mode",
        "tsv_path",
        "caps_directory",
        "preprocessing",
    ]
    if random_search:
        mandatory_arguments += ["n_convblocks", "first_conv_width", "n_fcblocks"]

    for argument in mandatory_arguments:
        if argument not in options:
            raise ClinicaDLArgumentError(
                f"The argument {argument} must be specified in the parameters."
            )

    if random_search:
        for mode, mode_dict in mode_default_values.items():
            set_default(options, mode_dict)
        if options["network_task"] not in task_default_values:
            raise NotImplementedError(
                f"The task default arguments corresponding to {options['network_task']} were not implemented."
            )
        task_dict = task_default_values[options["network_task"]]
        set_default(options, task_dict)
    else:
        if options["mode"] not in mode_default_values:
            raise NotImplementedError(
                f"The mode default arguments corresponding to {options['mode']} were not implemented."
            )
        if options["network_task"] not in task_default_values:
            raise NotImplementedError(
                f"The task default arguments corresponding to {options['network_task']} were not implemented."
            )
        mode_dict = mode_default_values[options["mode"]]
        task_dict = task_default_values[options["network_task"]]
        set_default(options, mode_dict)
        set_default(options, task_dict)


def memReport():
    import gc

    import torch

    cnt_tensor = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and (
            hasattr(obj, "data") and torch.is_tensor(obj.data)
        ):
            print(type(obj), obj.size(), obj.is_cuda)
            cnt_tensor += 1
    print("Count: ", cnt_tensor)


def cpuStats():
    import multiprocessing
    import sys

    import psutil

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    process = multiprocessing.current_process()
    pid = process.pid

    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0**30  # memory use in GB...I think
    print("memory GB:", memoryUse)
