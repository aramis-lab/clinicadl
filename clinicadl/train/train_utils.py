import os
from typing import Any, Dict

import toml

from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.maps_manager.maps_manager_utils import (
    read_json,
    remove_unused_tasks,
)


def build_train_dict(config_file: str, task: str) -> Dict[str, Any]:
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
        clinicadl_root_dir = os.path.abspath(os.path.join(__file__, "../.."))
        config_path = os.path.join(
            clinicadl_root_dir,
            "resources",
            "config",
            "train_config.toml",
        )
        config_dict = toml.load(config_path)
        config_dict = remove_unused_tasks(config_dict, task)

        train_dict = dict()
        # Fill train_dict from TOML files arguments
        for config_section in config_dict:
            for key in config_dict[config_section]:
                train_dict[key] = config_dict[config_section][key]

    elif config_file.endswith(".toml"):
        user_dict = toml.load(config_file)
        if "Random_Search" in user_dict:
            del user_dict["Random_Search"]

        # read default values
        clinicadl_root_dir = os.path.abspath(os.path.join(__file__, "../.."))
        config_path = os.path.join(
            clinicadl_root_dir,
            "resources",
            "config",
            "train_config.toml",
        )
        config_dict = toml.load(config_path)
        # Check that TOML file has the same format as the one in clinicadl/resources/config/train_config.toml
        if user_dict is not None:
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

    elif config_file.endswith(".json"):
        train_dict = read_json(config_file)

    else:
        raise ClinicaDLConfigurationError(
            f"config_file {config_file} should be a TOML or a JSON file."
        )

    return train_dict


def get_model_list(architecture=None, input_size=None, model_layers=None):
    """
    Print the list of models available in ClinicaDL.
    If architecture is given, it prints some informations of the specified model.
    If you add the flag -model_layers, this pipeline will show the whole model layers.
    """
    from inspect import getmembers, isclass

    from click import echo, secho

    import clinicadl.utils.network as network_package

    if not architecture:
        echo("The list of currently available models is:")
        model_list = getmembers(network_package, isclass)
        for model in model_list:
            if model[0] != "RandomArchitecture":
                echo(f" - {model[0]}")
        echo(
            "To show details of a specific model architecture please use the --architecture option"
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

        secho(f"\nInformation for '{architecture}' network:", bold=True)

        if not model_layers:

            print(model_class.__doc__)
            dimension = model_class.get_dimension()
            if dimension == 0:
                print(
                    "\tThis model can only deal with".expandtabs(4),
                    "\033[1m" + "2D input" + "\033[0m",
                    "and it must be in the shape C@HxW.",
                )
            elif dimension == 1:
                print(
                    "\tThis model can only deal with".expandtabs(4),
                    "\033[1m" + "3D input" + "\033[0m",
                    " and it must be in the shape C@DxHxW.",
                )
            elif dimension == 2:
                print(
                    "\tThis model can deal with both".expandtabs(4),
                    "\033[1m" + "2D and 3D input" + "\033[0m",
                    " and it must be in the shape C@HxW if the image is 2D or C@DxHxW if the image is 3D.",
                )

            print(f"\tFor example, input_size can be {input_size}.".expandtabs(4))

            task_list = model_class.get_task()
            task_str = (
                "\tThis model can be used for ".expandtabs(4)
                + "\033[1m"
                + f"{task_list[0]}"
                + "\033[0m"
            )
            list_size = len(task_list)
            if list_size > 0:
                for i in range(1, list_size):
                    task_str = (
                        task_str + " or " + "\033[1m" + f"{task_list[i]}" + "\033[0m"
                    )
            task_str = task_str + ".\n"
            print(task_str)
        else:
            chanel, shape_list = input_size.split("@")
            args.remove("self")
            kwargs = dict()
            kwargs["input_size"] = [int(chanel)] + [
                int(x) for x in shape_list.split("x")
            ]
            kwargs["gpu"] = False

            model = model_class(**kwargs)

            echo(f"Input size: {input_size}")
            echo("Model layers:")
            echo(model.layers)
