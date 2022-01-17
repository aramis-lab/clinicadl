import os
from typing import Any, Dict

import toml

from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.maps_manager.maps_manager_utils import (
    read_json,
    remove_unused_tasks,
)


def get_user_dict(config_file: str, task: str) -> Dict[str, Any]:
    """
    Read the configuration file given by the user.
    If it is a TOML file, ensures that the format corresponds to the one in resources.
    Args:
        config_file: path to a configuration file (JSON of TOML).
        task: task learnt by the network (example: classification, regression, reconstruction...).
    Returns:
        dictionary of values ready to use for the MapsManager
    """
    if config_file.endswith(".toml"):
        toml_dict = toml.load(config_file)
        if "Random_Search" in toml_dict:
            del toml_dict["Random_Search"]

        # read default values
        clinicadl_root_dir = os.path.abspath(os.path.join(__file__, "../.."))
        config_path = os.path.join(
            clinicadl_root_dir,
            "resources",
            "config",
            "train_config.toml",
        )
        config_dict = toml.load(config_path)
        # Check that TOML file has the same format as the one in resources
        if toml_dict is not None:
            for section_name in toml_dict:
                if section_name not in config_dict:
                    raise ClinicaDLConfigurationError(
                        f"{section_name} section is not valid in TOML configuration file. "
                        f"Please see the documentation to see the list of option in TOML configuration file."
                    )
                for key in toml_dict[section_name]:
                    if key not in config_dict[section_name]:
                        raise ClinicaDLConfigurationError(
                            f"{key} option in {section_name} is not valid in TOML configuration file. "
                            f"Please see the documentation to see the list of option in TOML configuration file."
                        )

        train_dict = dict()

        # task dependent
        toml_dict = remove_unused_tasks(toml_dict, task)

        # Standard arguments
        for config_section in toml_dict:
            for key in toml_dict[config_section]:
                train_dict[key] = toml_dict[config_section][key]

    elif config_file.endswith(".json"):
        train_dict = read_json(config_file)
    else:
        raise ClinicaDLConfigurationError(
            f"config_file {config_file} should be a TOML or a JSON file."
        )

    return train_dict


def get_model_list(architecture=None, input_size=(128, 128)):
    """
    Print the list of models available in ClinicaDL.
    If architecture is given, it prints the details of the specified model.
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

        # parse input_size
        chanel, shape_list = input_size.split("@")

        args.remove("self")
        kwargs = dict()
        kwargs["input_size"] = [int(chanel)] + [int(x) for x in shape_list.split("x")]
        kwargs["gpu"] = False

        model = model_class(**kwargs)
        secho(f"Information for {architecture} network", bold=True)
        echo(f"Input size: {input_size}")
        echo("Model layers:")
        echo(model.layers)
