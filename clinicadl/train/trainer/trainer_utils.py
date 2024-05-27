import shutil
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def create_parameters_dict(config):
    parameters = {}
    config_dict = config.model_dump()
    for key in config_dict:
        if isinstance(config_dict[key], dict):
            parameters.update(config_dict[key])
        else:
            parameters[key] = config_dict[key]

    maps_path = parameters["maps_dir"]
    del parameters["maps_dir"]
    for parameter in parameters:
        if parameters[parameter] == Path("."):
            parameters[parameter] = ""
    if parameters["transfer_path"] is None:
        parameters["transfer_path"] = False
    if parameters["data_augmentation"] == ():
        parameters["data_augmentation"] = False
    parameters["preprocessing_dict_target"] = parameters["preprocessing_json_target"]
    del parameters["preprocessing_json_target"]
    del parameters["preprocessing_json"]
    parameters["tsv_path"] = parameters["tsv_directory"]
    del parameters["tsv_directory"]
    parameters["compensation"] = parameters["compensation"].value
    parameters["size_reduction_factor"] = parameters["size_reduction_factor"].value
    if parameters["track_exp"]:
        parameters["track_exp"] = parameters["track_exp"].value
    else:
        parameters["track_exp"] = ""
    parameters["sampler"] = parameters["sampler"].value
    if parameters["network_task"] == "reconstruction":
        parameters["normalization"] = parameters["normalization"].value
    parameters[
        "split"
    ] = []  # TODO : this is weird, see old ClinicaDL behavior (.pop("split") in task_launcher)
    if len(config.data.label_code) == 0:
        del parameters["label_code"]

    # if parameters["selection_threshold"]== 0.0:
    #     parameters["selection_threshold"] = False
    if parameters["n_subjects"] == 300:
        del parameters["n_subjects"]
    if parameters["overwrite"] is False:
        del parameters["overwrite"]
    if parameters["save_nifti"] is False:
        del parameters["save_nifti"]
    if parameters["skip_leak_check"] is False:
        del parameters["skip_leak_check"]

    return parameters, maps_path
