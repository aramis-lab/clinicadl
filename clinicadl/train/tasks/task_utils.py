from logging import getLogger
from typing import List

from clinicadl.utils.caps_dataset.data import CapsDataset
from clinicadl.utils.preprocessing import read_preprocessing


def task_launcher(network_task: str, task_options_list: List[str], **kwargs):
    """
    Common training framework for all tasks

    Args:
        network_task: task learnt by the network.
        task_options_list: list of options specific to the task.
        kwargs: other arguments and options for network training.
    """
    from pathlib import Path

    from clinicadl.train.train import train
    from clinicadl.train.train_utils import build_train_dict

    logger = getLogger("clinicadl.task_manager")

    config_file_name = None
    if kwargs["config_file"]:
        config_file_name = Path(kwargs["config_file"])
    train_dict = build_train_dict(config_file_name, network_task)

    # Add arguments
    train_dict["network_task"] = network_task
    train_dict["caps_directory"] = Path(kwargs["caps_directory"])
    train_dict["tsv_path"] = Path(kwargs["tsv_directory"])

    # Change value in train dict depending on user provided options
    standard_options_list = [
        "accumulation_steps",
        "adaptive_learning_rate",
        "amp",
        "architecture",
        "baseline",
        "batch_size",
        "compensation",
        "data_augmentation",
        "deterministic",
        "diagnoses",
        "dropout",
        "epochs",
        "evaluation_steps",
        "fully_sharded_data_parallel",
        "gpu",
        "learning_rate",
        "multi_cohort",
        "multi_network",
        "ssda_network",
        "n_proc",
        "n_splits",
        "nb_unfrozen_layer",
        "normalize",
        "optimizer",
        "patience",
        "profiler",
        "tolerance",
        "track_exp",
        "transfer_path",
        "transfer_selection_metric",
        "weight_decay",
        "sampler",
        "save_all_models",
        "seed",
        "split",
        "caps_target",
        "tsv_target_lab",
        "tsv_target_unlab",
        "preprocessing_dict_target",
    ]
    all_options_list = standard_options_list + task_options_list

    for option in all_options_list:
        if (kwargs[option] is not None and not isinstance(kwargs[option], tuple)) or (
            isinstance(kwargs[option], tuple) and len(kwargs[option]) != 0
        ):
            train_dict[option] = kwargs[option]
    if not train_dict["multi_cohort"]:
        preprocessing_json = (
            train_dict["caps_directory"]
            / "tensor_extraction"
            / kwargs["preprocessing_json"]
        )

        if train_dict["ssda_network"]:
            preprocessing_json_target = (
                Path(kwargs["caps_target"])
                / "tensor_extraction"
                / kwargs["preprocessing_dict_target"]
            )
    else:
        caps_dict = CapsDataset.create_caps_dict(
            train_dict["caps_directory"], train_dict["multi_cohort"]
        )
        json_found = False
        for caps_name, caps_path in caps_dict.items():
            preprocessing_json = (
                caps_path / "tensor_extraction" / kwargs["preprocessing_json"]
            )
            if preprocessing_json.is_file():
                logger.info(
                    f"Preprocessing JSON {preprocessing_json} found in CAPS {caps_name}."
                )
                json_found = True
        if not json_found:
            raise ValueError(
                f"Preprocessing JSON {kwargs['preprocessing_json']} was not found for any CAPS "
                f"in {caps_dict}."
            )
        # To CHECK AND CHANGE
        if train_dict["ssda_network"]:
            caps_target = Path(kwargs["caps_target"])
            preprocessing_json_target = (
                caps_target / "tensor_extraction" / kwargs["preprocessing_dict_target"]
            )

            if preprocessing_json_target.is_file():
                logger.info(
                    f"Preprocessing JSON {preprocessing_json_target} found in CAPS {caps_target}."
                )
                json_found = True
            if not json_found:
                raise ValueError(
                    f"Preprocessing JSON {kwargs['preprocessing_json_target']} was not found for any CAPS "
                    f"in {caps_target}."
                )

    # Mode and preprocessing
    preprocessing_dict = read_preprocessing(preprocessing_json)
    train_dict["preprocessing_dict"] = preprocessing_dict
    train_dict["mode"] = preprocessing_dict["mode"]

    if train_dict["ssda_network"]:
        preprocessing_dict_target = read_preprocessing(preprocessing_json_target)
        train_dict["preprocessing_dict_target"] = preprocessing_dict_target

    # Add default values if missing
    if (
        preprocessing_dict["mode"] == "roi"
        and "roi_background_value" not in preprocessing_dict
    ):
        preprocessing_dict["roi_background_value"] = 0

    train(Path(kwargs["output_maps_directory"]), train_dict, train_dict.pop("split"))
