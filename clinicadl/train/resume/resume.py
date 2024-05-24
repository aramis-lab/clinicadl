"""
Automatic relaunch of jobs that were stopped before the end of training.
Unfinished splits are detected as they do not contain a "performances" sub-folder
"""

from logging import getLogger
from pathlib import Path

from clinicadl import MapsManager
from clinicadl.train.tasks import create_training_config
from clinicadl.train.trainer import Trainer


def replace_arg(options, key_name, value):
    if value is not None:
        setattr(options, key_name, value)


def automatic_resume(model_path: Path, user_split_list=None, verbose=0):
    logger = getLogger("clinicadl")

    verbose_list = ["warning", "info", "debug"]
    maps_manager = MapsManager(model_path, verbose=verbose_list[verbose])
    config_dict = maps_manager.get_parameters()
    # temporary, TODO
    config_dict["tsv_directory"] = config_dict["tsv_path"]
    if config_dict["track_exp"] == "":
        config_dict["track_exp"] = None
    if not config_dict["label_code"]:
        config_dict["label_code"] = {}
    # if not config_dict["preprocessing_json"]
    # = config_dict["extract_json"]
    config_dict["maps_dir"] = config_dict["output_maps_dir"]

    ###
    config = create_training_config(config_dict["network_task"])(
        output_maps_directory=model_path, **config_dict
    )
    trainer = Trainer(config, maps_manager=maps_manager)

    existing_split_list = maps_manager._find_splits()
    stopped_splits = [
        split
        for split in existing_split_list
        if (model_path / f"{maps_manager.split_name}-{split}" / "tmp")
        in list((model_path / f"{maps_manager.split_name}-{split}").iterdir())
    ]

    # Find finished split
    finished_splits = list()
    for split in existing_split_list:
        if split not in stopped_splits:
            performance_dir_list = [
                performance_dir
                for performance_dir in list(
                    (model_path / f"{maps_manager.split_name}-{split}").iterdir()
                )
                if "best-" in performance_dir.name
            ]
            if len(performance_dir_list) > 0:
                finished_splits.append(split)

    split_manager = maps_manager._init_split_manager(split_list=user_split_list)
    split_iterator = split_manager.split_iterator()

    absent_splits = [
        split
        for split in split_iterator
        if split not in finished_splits and split not in stopped_splits
    ]

    # To ensure retro-compatibility with random search
    logger.info(
        f"Finished splits {finished_splits}\n"
        f"Stopped splits {stopped_splits}\n"
        f"Absent splits {absent_splits}"
    )
    if len(stopped_splits) > 0:
        trainer.resume(stopped_splits)
    if len(absent_splits) > 0:
        trainer.train(absent_splits, overwrite=True)
