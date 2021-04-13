# coding: utf8
import logging
import sys

LOG_LEVELS = [logging.WARNING, logging.INFO, logging.DEBUG]


class StdLevelFilter(logging.Filter):
    def __init__(self, err=False):
        super().__init__()
        self.err = err

    def filter(self, record):
        if record.levelno <= logging.INFO:
            return not self.err
        return self.err


def return_logger(verbose, name_fn):
    logger = logging.getLogger(name_fn)
    if verbose < len(LOG_LEVELS):
        logger.setLevel(LOG_LEVELS[verbose])
    else:
        logger.setLevel(logging.DEBUG)
    stdout = logging.StreamHandler(sys.stdout)
    stdout.addFilter(StdLevelFilter())
    stderr = logging.StreamHandler(sys.stderr)
    stderr.addFilter(StdLevelFilter(err=True))
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    # add formatter to ch
    stdout.setFormatter(formatter)
    stderr.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(stdout)
    logger.addHandler(stderr)

    return logger


computational_list = ['gpu', 'batch_size', 'num_workers', 'evaluation_steps']


def write_requirements_version(output_path):
    import subprocess
    from os import path
    from warnings import warn

    try:
        env_variables = subprocess.check_output("pip freeze", shell=True).decode("utf-8")
        with open(path.join(output_path, "environment.txt"), "w") as file:
            file.write(env_variables)
    except subprocess.CalledProcessError:
        warn("You do not have the right to execute pip freeze. Your environment will not be written")


def translate_parameters(args):
    """
    Translate the names of the parameters between command line and source code.
    """
    args.gpu = not args.use_cpu
    args.num_workers = args.nproc
    args.optimizer = "Adam"
    args.loss = "default"

    if hasattr(args, "caps_dir"):
        args.input_dir = args.caps_dir
    if hasattr(args, "unnormalize"):
        args.minmaxnormalization = not args.unnormalize
    if hasattr(args, "slice_direction"):
        args.mri_plane = args.slice_direction
    if hasattr(args, "network_type"):
        args.mode_task = args.network_type

    if not hasattr(args, "selection_threshold"):
        args.selection_threshold = None

    if not hasattr(args, "prepare_dl"):
        if hasattr(args, "use_extracted_features"):
            args.prepare_dl = args.use_extracted_features
        elif hasattr(args, "use_extracted_patches") and args.mode == "patch":
            args.prepare_dl = args.use_extracted_patches
        elif hasattr(args, "use_extracted_slices") and args.mode == "slice":
            args.prepare_dl = args.use_extracted_slices
        elif hasattr(args, "use_extracted_roi") and args.mode == "roi":
            args.prepare_dl = args.use_extracted_roi
        else:
            args.prepare_dl = False

    return args


def check_and_clean(d):
    import shutil
    import os

    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def commandline_to_json(commandline, logger=None, filename="commandline.json"):
    """
    This is a function to write the python argparse object into a json file.
    This helps for DL when searching for hyperparameters
    Args:
        commandline: (Namespace or dict) the output of `parser.parse_known_args()`
        logger: (logging object) writer to stdout and stderr
        filename: (str) name of the JSON file.

    :return:
    """
    if logger is None:
        logger = logging

    import json
    import os
    from copy import copy

    if isinstance(commandline, dict):
        commandline_arg_dict = copy(commandline)
    else:
        commandline_arg_dict = copy(vars(commandline))
    output_dir = commandline_arg_dict['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # remove these entries from the commandline log file
    if 'func' in commandline_arg_dict:
        del commandline_arg_dict['func']

    if 'output_dir' in commandline_arg_dict:
        del commandline_arg_dict['output_dir']

    if 'launch_dir' in commandline_arg_dict:
        del commandline_arg_dict['launch_dir']

    if 'name' in commandline_arg_dict:
        del commandline_arg_dict['name']

    if 'verbose' in commandline_arg_dict:
        del commandline_arg_dict['verbose']

    # save to json file
    json = json.dumps(commandline_arg_dict, skipkeys=True, indent=4)
    logger.info("Path of json file: %s" % os.path.join(output_dir, "commandline.json"))
    f = open(os.path.join(output_dir, filename), "w")
    f.write(json)
    f.close()


def read_json(options, json_path=None, test=False):
    """
    Read a json file to update python argparse Namespace.
    Ensures retro-compatibility with previous namings in clinicadl.

    Args:
        options: (argparse.Namespace) options of the model.
        json_path: (str) If given path to the json file, else found with options.model_path.
        test: (bool) If given the reader will ignore some options specific to data.
    Returns:
        options (args.Namespace) options of the model updated
    """
    import json
    from os import path

    evaluation_parameters = ["diagnosis_path", "input_dir", "diagnoses"]
    prep_compatibility_dict = {"mni": "t1-extensive", "linear": "t1-linear"}
    if json_path is None:
        json_path = path.join(options.model_path, 'commandline.json')

    with open(json_path, "r") as f:
        json_data = json.load(f)

    for key, item in json_data.items():
        # We do not change computational options
        if key in computational_list:
            pass
        # If used for evaluation, some parameters were already given
        if test and key in evaluation_parameters:
            pass
        else:
            setattr(options, key, item)

    # Retro-compatibility with runs of previous versions
    if hasattr(options, "network"):
        options.model = options.network
        del options.network

    if not hasattr(options, 'discarded_sliced'):
        options.discarded_slices = 20

    if isinstance(options.preprocessing, str):
        if options.preprocessing in prep_compatibility_dict.keys():
            options.preprocessing = prep_compatibility_dict[options.preprocessing]

    if hasattr(options, 'mri_plane'):
        options.slice_direction = options.mri_plane
        del options.mri_plane

    if hasattr(options, "hippocampus_roi"):
        if options.hippocampus_roi:
            options.mode = "roi"
            del options.hippocampus_roi

    if hasattr(options, "pretrained_path"):
        options.transfer_learning_path = options.pretrained_path
        del options.pretrained_path

    if hasattr(options, "pretrained_difference"):
        options.transfer_learning_difference = options.pretrained_difference
        del options.pretrained_difference

    if hasattr(options, 'patch_stride'):
        options.stride_size = options.patch_stride

    if hasattr(options, 'use_gpu'):
        options.use_cpu = not options.use_gpu

    if hasattr(options, 'mode'):
        if options.mode == "subject":
            options.mode = "image"
        if options.mode == "slice" and not hasattr(options, "network_type"):
            options.network_type = "cnn"
        if options.mode == "patch" and hasattr(options, "network_type"):
            if options.network_type == "multi":
                options.network_type = "multicnn"

    if not hasattr(options, "network_type"):
        if hasattr(options, "mode_task"):
            options.network_type = options.mode_task
        elif hasattr(options, "train_autoencoder"):
            options.network_type = "autoencoder"
        else:
            options.network_type = "cnn"

    if hasattr(options, "selection"):
        options.transfer_learning_selection = options.selection

    if not hasattr(options, "loss"):
        options.loss = "default"

    if not hasattr(options, 'dropout') or options.dropout is None:
        options.dropout = None
        set_default_dropout(options)

    if not hasattr(options, 'uncropped_roi'):
        options.uncropped_roi = False

    if not hasattr(options, 'roi_list'):
        options.roi_list = None

    if not hasattr(options, 'multi_cohort'):
        options.multi_cohort = False

    return options


def set_default_dropout(args):
    if args.dropout is None:
        if args.mode == 'image':
            args.dropout = 0.5
        elif args.mode == 'slice':
            args.dropout = 0.8
        else:
            args.dropout = 0


def memReport():
    import gc
    import torch

    cnt_tensor = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size(), obj.is_cuda)
            cnt_tensor += 1
    print('Count: ', cnt_tensor)


def cpuStats():
    import sys
    import psutil
    import os

    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)
