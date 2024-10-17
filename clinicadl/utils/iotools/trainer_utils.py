from pathlib import Path


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

    del parameters["preprocessing_json"]
    # if "tsv_path" in parameters:
    #     parameters["tsv_path"] = parameters["tsv_path"]
    #     del parameters["tsv_path"]
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
    if config.data.label_code is None or len(config.data.label_code) == 0:
        if "label_code" in parameters:
            del parameters["label_code"]

    # if parameters["selection_threshold"]== 0.0:
    #     parameters["selection_threshold"] = False

    if parameters["config_file"] is None:
        del parameters["config_file"]
    if parameters["data_group"] is None:
        del parameters["data_group"]
    if not parameters["data_tsv"]:
        del parameters["data_tsv"]
    if parameters["n_subjects"] == 300:
        del parameters["n_subjects"]
    if parameters["overwrite"] is False:
        del parameters["overwrite"]
    if parameters["save_nifti"] is False:
        del parameters["save_nifti"]
    if parameters["skip_leak_check"] is False:
        del parameters["skip_leak_check"]
    if "normalization" in parameters and parameters["normalization"] == "BatchNorm":
        parameters["normalization"] = "batch"
    if "caps_dict" in parameters:
        del parameters["caps_dict"]
    if "mask_path" in parameters:  # and parameters["mask_path"] is False:
        del parameters["mask_path"]

    if "data_df" in parameters:
        del parameters["data_df"]
    if "train_transformations" in parameters:
        del parameters["train_transformations"]
    return parameters, maps_path


def patch_to_read_json(config_dict):
    config_dict["tsv_path"] = config_dict["tsv_path"]
    if ("track_exp" in config_dict) and (config_dict["track_exp"] == ""):
        config_dict["track_exp"] = None
    config_dict["preprocessing_json"] = config_dict["preprocessing_dict"][
        "extract_json"
    ]
    if "label_code" not in config_dict or config_dict["label_code"] is None:
        config_dict["label_code"] = {}
    if "preprocessing_json" not in config_dict:
        config_dict["preprocessing_json"] = config_dict["preprocessing_dict"][
            "extract_json"
        ]
    return config_dict
