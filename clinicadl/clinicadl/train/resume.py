"""
Automatic relaunch of jobs that were stopped before the end of training.
Unfinished folds are detected as they do not contain a "performances" sub-folder
"""
import os
from os import path

from clinicadl import MapsManager


def replace_arg(options, key_name, value):
    if value is not None:
        setattr(options, key_name, value)


def automatic_resume(model_path, verbose=0):
    verbose_list = ["warning", "info", "debug"]
    maps_manager = MapsManager(model_path, verbose=verbose_list[verbose])

    fold_list = sorted(
        [
            int(fold.split("-")[1])
            for fold in os.listdir(model_path)
            if fold[:4:] == "fold"
        ]
    )
    stopped_folds = [
        fold
        for fold in fold_list
        if "tmp" in os.listdir(path.join(model_path, f"fold-{fold}"))
    ]
    finished_folds = [fold for fold in fold_list if fold not in stopped_folds]

    split_manager = maps_manager._init_split_manager()
    fold_iterator = split_manager.fold_iterator()

    absent_folds = [
        fold
        for fold in fold_iterator
        if fold not in finished_folds and fold not in stopped_folds
    ]

    # To ensure retro-compatibility with random search
    maps_manager.resume(stopped_folds)
    maps_manager.train(absent_folds)
