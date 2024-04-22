from logging import getLogger
from pathlib import Path

from clinicadl.train.train import train
from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.preprocessing import read_preprocessing

from .base_training_config import BaseTaskConfig

logger = getLogger("clinicadl.task_manager")


def task_launcher(config: BaseTaskConfig) -> None:
    """
    Common training framework for all tasks.

    Adds private attributes to the Config object and launches training.

    Parameters
    ----------
    config : BaseTaskConfig
        Configuration object with all the parameters.

    Raises
    ------
    ValueError
        If the <preprocessing_json> parameter doesn't match any existing file.
    ValueError
        If the <preprocessing_dict_target> parameter doesn't match any existing file.
    """
    if not config.multi_cohort:
        preprocessing_json = (
            config.caps_directory / "tensor_extraction" / config.preprocessing_json
        )

        if config.ssda_network:
            preprocessing_json_target = (
                config.caps_target
                / "tensor_extraction"
                / config.preprocessing_dict_target
            )
    else:
        caps_dict = CapsDataset.create_caps_dict(
            config.caps_directory, config.multi_cohort
        )
        json_found = False
        for caps_name, caps_path in caps_dict.items():
            preprocessing_json = (
                caps_path / "tensor_extraction" / config.preprocessing_json
            )
            if preprocessing_json.is_file():
                logger.info(
                    f"Preprocessing JSON {preprocessing_json} found in CAPS {caps_name}."
                )
                json_found = True
        if not json_found:
            raise ValueError(
                f"Preprocessing JSON {config.preprocessing_json} was not found for any CAPS "
                f"in {caps_dict}."
            )
        # To CHECK AND CHANGE
        if config.ssda_network:
            caps_target = config.caps_target
            preprocessing_json_target = (
                caps_target / "tensor_extraction" / config.preprocessing_dict_target
            )

            if preprocessing_json_target.is_file():
                logger.info(
                    f"Preprocessing JSON {preprocessing_json_target} found in CAPS {caps_target}."
                )
                json_found = True
            if not json_found:
                raise ValueError(
                    f"Preprocessing JSON {preprocessing_json_target} was not found for any CAPS "
                    f"in {caps_target}."
                )

    # Mode and preprocessing
    preprocessing_dict = read_preprocessing(preprocessing_json)
    config._preprocessing_dict = preprocessing_dict
    config._mode = preprocessing_dict["mode"]

    if config.ssda_network:
        config._preprocessing_dict_target = read_preprocessing(
            preprocessing_json_target
        )

    # Add default values if missing
    if (
        preprocessing_dict["mode"] == "roi"
        and "roi_background_value" not in preprocessing_dict
    ):
        config._preprocessing_dict["roi_background_value"] = 0

    # temporary # TODO : change train function to give it a config object
    maps_dir = config.output_maps_directory
    train_dict = config.model_dump(
        exclude=["output_maps_directory", "preprocessing_json", "tsv_directory"]
    )
    train_dict["tsv_path"] = config.tsv_directory
    train_dict[
        "preprocessing_dict"
    ] = config._preprocessing_dict  # private attributes are not dumped
    train_dict["mode"] = config._mode
    if config.ssda_network:
        train_dict["preprocessing_dict_target"] = config._preprocessing_dict_target
    train_dict["network_task"] = config._network_task
    if train_dict["transfer_path"] is None:
        train_dict["transfer_path"] = False
    split_list = config.split
    #############

    train(maps_dir, train_dict, split_list)
