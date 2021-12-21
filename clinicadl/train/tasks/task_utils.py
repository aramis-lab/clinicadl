import os
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
    from clinicadl.train.train import train
    from clinicadl.train.train_utils import get_user_dict

    logger = getLogger("clinicadl")

    if not kwargs["multi_cohort"]:
        preprocessing_json = os.path.join(
            kwargs["caps_directory"], "tensor_extraction", kwargs["preprocessing_json"]
        )
    else:
        caps_dict = CapsDataset.create_caps_dict(
            kwargs["caps_directory"], kwargs["multi_cohort"]
        )
        json_found = False
        for caps_name, caps_path in caps_dict.items():
            if os.path.exists(
                os.path.join(
                    caps_path, "tensor_extraction", kwargs["preprocessing_json"]
                )
            ):
                preprocessing_json = os.path.join(
                    caps_path, "tensor_extraction", kwargs["preprocessing_json"]
                )
                logger.info(
                    f"Preprocessing JSON {preprocessing_json} found in CAPS {caps_name}."
                )
                json_found = True
        if not json_found:
            raise ValueError(
                f"Preprocessing JSON {kwargs['preprocessing_json']} was not found for any CAPS "
                f"in {caps_dict}."
            )

    if kwargs["config_file"]:
        train_dict = get_user_dict(kwargs["config_file"].name, network_task)
    else:
        train_dict = dict()

    # Mode and preprocessing
    preprocessing_dict = read_preprocessing(preprocessing_json)
    train_dict["preprocessing_dict"] = preprocessing_dict
    train_dict["mode"] = preprocessing_dict["mode"]

    # Add arguments
    train_dict["network_task"] = network_task
    train_dict["caps_directory"] = kwargs["caps_directory"]
    train_dict["tsv_path"] = kwargs["tsv_directory"]

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
        "patience",
        "tolerance",
        "transfer_selection_metric",
        "weight_decay",
        "sampler",
        "seed",
        "split",
        "compensation",
        "transfer_path",
    ]
    standard_options_list = standard_options_list + task_options_list

    for option in standard_options_list:
        if (kwargs[option] is not None and not isinstance(kwargs[option], tuple)) or (
            isinstance(kwargs[option], tuple) and len(kwargs[option]) != 0
        ):
            train_dict[option] = kwargs[option]

    train(kwargs["output_maps_directory"], train_dict, train_dict.pop("split"))
