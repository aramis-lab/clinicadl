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
        "architecture",
        "baseline",
        "batch_size",
        "data_augmentation",
        "deterministic",
        "diagnoses",
        "dropout",
        "epochs",
        "evaluation_steps",
        "gpu",
        "learning_rate",
        "multi_cohort",
        "multi_network",
        "n_proc",
        "n_splits",
        "normalize",
        "optimizer",
        "patience",
        "profiler",
        "tolerance",
        "transfer_selection_metric",
        "weight_decay",
        "sampler",
        "seed",
        "split",
        "compensation",
        "transfer_path",
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

    # Mode and preprocessing
    preprocessing_dict = read_preprocessing(preprocessing_json)
    train_dict["preprocessing_dict"] = preprocessing_dict
    train_dict["mode"] = preprocessing_dict["mode"]

    # Add default values if missing
    if (
        preprocessing_dict["mode"] == "roi"
        and "roi_background_value" not in preprocessing_dict
    ):
        preprocessing_dict["roi_background_value"] = 0

    train(Path(kwargs["output_maps_directory"]), train_dict, train_dict.pop("split"))
