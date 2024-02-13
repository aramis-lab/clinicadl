from pathlib import Path
from typing import Any, Dict

import toml

from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLConfigurationError,
)
from clinicadl.utils.maps_manager.maps_manager_utils import (
    change_str_to_path,
    read_json,
    remove_unused_tasks,
)


def build_train_dict(config_file: Path, task: str) -> Dict[str, Any]:
    """
    Read the configuration file given by the user.
    If it is a TOML file, ensures that the format corresponds to the one in resources.
    Args:
        config_file: path to a configuration file (JSON of TOML).
        task: task learnt by the network (example: classification, regression, reconstruction...).
    Returns:
        dictionary of values ready to use for the MapsManager
    """
    if config_file is None:
        # read default values
        clinicadl_root_dir = (Path(__file__) / "../..").resolve()
        config_path = (
            Path(clinicadl_root_dir) / "resources" / "config" / "train_config.toml"
        )
        config_dict = toml.load(config_path)
        config_dict = remove_unused_tasks(config_dict, task)
        config_dict = change_str_to_path(config_dict)
        train_dict = dict()
        # Fill train_dict from TOML files arguments
        for config_section in config_dict:
            for key in config_dict[config_section]:
                train_dict[key] = config_dict[config_section][key]

    elif config_file.suffix == ".toml":
        user_dict = toml.load(config_file)
        if "Random_Search" in user_dict:
            del user_dict["Random_Search"]

        # read default values
        clinicadl_root_dir = (Path(__file__) / "../..").resolve()
        config_path = (
            Path(clinicadl_root_dir) / "resources" / "config" / "train_config.toml"
        )
        config_dict = toml.load(config_path)
        # Check that TOML file has the same format as the one in clinicadl/resources/config/train_config.toml
        if user_dict is not None:
            user_dict = change_str_to_path(user_dict)
            for section_name in user_dict:
                if section_name not in config_dict:
                    raise ClinicaDLConfigurationError(
                        f"{section_name} section is not valid in TOML configuration file. "
                        f"Please see the documentation to see the list of option in TOML configuration file."
                    )
                for key in user_dict[section_name]:
                    if key not in config_dict[section_name]:
                        raise ClinicaDLConfigurationError(
                            f"{key} option in {section_name} is not valid in TOML configuration file. "
                            f"Please see the documentation to see the list of option in TOML configuration file."
                        )
                    config_dict[section_name][key] = user_dict[section_name][key]

        train_dict = dict()

        # task dependent
        config_dict = remove_unused_tasks(config_dict, task)

        # Fill train_dict from TOML files arguments
        for config_section in config_dict:
            for key in config_dict[config_section]:
                train_dict[key] = config_dict[config_section][key]

    elif config_file.suffix == ".json":
        train_dict = read_json(config_file)
        train_dict = change_str_to_path(train_dict)

    else:
        raise ClinicaDLConfigurationError(
            f"config_file {config_file} should be a TOML or a JSON file."
        )
    return train_dict


def get_model_list(architecture=None, input_size=None, model_layers=False):
    """
    Print the list of models available in ClinicaDL.
    If --architecture is given, information about how to use this model will be displayed.
    If --model_layers flag is added, this pipeline will show the whole model layers.
    If --input_size is added, it will show the whole model layers with chosen input shape.
    """
    from inspect import getmembers, isclass

    import clinicadl.utils.network as network_package

    if not architecture:
        print("The list of currently available models is:")
        model_list = getmembers(network_package, isclass)
        for model in model_list:
            if model[0] != "RandomArchitecture":
                print(f" - {model[0]}")
        print(
            "To show details of a specific model architecture please use the `--architecture` option"
        )
    else:
        model_class = getattr(network_package, architecture)
        args = list(
            model_class.__init__.__code__.co_varnames[
                : model_class.__init__.__code__.co_argcount
            ]
        )
        if not input_size:
            input_size = model_class.get_input_size()

        title_str = f"\nInformation about '{architecture}' network:\n"

        dimension = model_class.get_dimension()
        dimension_str = f"\n\tThis model can deal with {dimension} inputs.".expandtabs(
            4
        )

        if dimension == "2D":
            shape_str = "C@HxW,"
        elif dimension == "3D":
            shape_str = "C@DxHxW,"
        elif dimension == "2D or 3D":
            shape_str = "C@HxW or C@DxHxW,"

        shape_str = f"\n\tThe input must be in the shape {shape_str}".expandtabs(4)
        input_size_str = f" for example input_size can be {input_size}."

        task_str = f"\n\tThis model can be used for {' or '.join(model_class.get_task())}.\n".expandtabs(
            4
        )

        print(
            title_str
            + model_class.__doc__
            + dimension_str
            + shape_str
            + input_size_str
            + task_str
        )

        if model_layers:
            chanel, shape_list = input_size.split("@")
            args.remove("self")
            kwargs = dict()
            kwargs["input_size"] = [int(chanel)] + [
                int(x) for x in shape_list.split("x")
            ]
            kwargs["gpu"] = False

            model = model_class(**kwargs)

            print(f"Input size: {input_size}")
            print("Model layers:")
            print(model.layers)
