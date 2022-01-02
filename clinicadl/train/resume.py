"""
Automatic relaunch of jobs that were stopped before the end of training.
Unfinished splits are detected as they do not contain a "performances" sub-folder
"""
import os
from logging import getLogger
from os import path

from clinicadl import MapsManager

logger = getLogger("clinicadl")


def replace_arg(options, key_name, value):
    if value is not None:
        setattr(options, key_name, value)


def automatic_resume(model_path, user_split_list=None, verbose=0):
    logger = getLogger("clinicadl")

    verbose_list = ["warning", "info", "debug"]
    maps_manager = MapsManager(model_path, verbose=verbose_list[verbose])

    existing_split_list = maps_manager._find_splits()
    stopped_splits = [
        split
        for split in existing_split_list
        if "tmp" in os.listdir(path.join(model_path, f"split-{split}"))
    ]
    finished_splits = [
        split for split in existing_split_list if split not in stopped_splits
    ]

    split_manager = maps_manager._init_split_manager(split_list=user_split_list)
    split_iterator = split_manager.split_iterator()

    absent_splits = [
        split
        for split in split_iterator
        if split not in finished_splits and split not in stopped_splits
    ]

    # To ensure retro-compatibility with random search
    print(
        f"Finished splits {finished_splits}\n"
        f"Stopped splits {stopped_splits}\n"
        f"Absent splits {absent_splits}"
    )
    if len(stopped_splits) > 0:
        maps_manager.resume(stopped_splits)
    if len(absent_splits) > 0:
        maps_manager.train(absent_splits)
