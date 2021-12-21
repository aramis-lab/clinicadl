"""
Automatic relaunch of jobs that were stopped before the end of training.
Unfinished splits are detected as they do not contain a "performances" sub-folder
"""
import os
from glob import glob
from logging import getLogger
from os import path

from clinicadl import MapsManager
from clinicadl.utils.exceptions import MAPSError

logger = getLogger("clinicadl")


def replace_arg(options, key_name, value):
    if value is not None:
        setattr(options, key_name, value)


def automatic_resume(model_path, verbose=0):
    logger = getLogger("clinicadl")

    verbose_list = ["warning", "info", "debug"]
    maps_manager = MapsManager(model_path, verbose=verbose_list[verbose])
    if len(glob(os.path.join(model_path, "fold-*"))) > 0:
        raise MAPSError(
            "This MAPS cannot be resumed with the current version of ClinicaDL. "
            "Please use the same version as for training or rename manually the folders "
            "'fold-*' in 'split-*' to respect the new MAPS convention."
        )

    split_list = sorted(
        [
            int(split.split("-")[1])
            for split in os.listdir(model_path)
            if split[:4:] == "split"
        ]
    )
    stopped_splits = [
        split
        for split in split_list
        if "tmp" in os.listdir(path.join(model_path, f"split-{split}"))
    ]
    finished_splits = [split for split in split_list if split not in stopped_splits]

    split_manager = maps_manager._init_split_manager()
    split_iterator = split_manager.split_iterator()

    absent_splits = [
        split
        for split in split_iterator
        if split not in finished_splits and split not in stopped_splits
    ]

    logger.info(f"List of finished splits {finished_splits}")
    logger.info(f"List of stopped splits {stopped_splits}")
    logger.info(f"List of absent splits {absent_splits}")
    # To ensure retro-compatibility with random search
    logger.info(
        f"Finished splits {finished_splits}\n"
        f"Stopped splits {stopped_splits}\n"
        f"Absent splits {absent_splits}"
    )
    maps_manager.resume(stopped_splits)
    maps_manager.train(absent_splits)
