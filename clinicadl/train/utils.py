# TODO: merge with task_utils to create the trainer_utils ?
from pathlib import Path
from typing import Any, Dict

import click
import toml
from click.core import ParameterSource

from clinicadl.preprocessing.preprocessing import path_decoder
from clinicadl.utils.enum import Task
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.maps_manager.maps_manager_utils import remove_unused_tasks


def extract_config_from_toml_file(config_file: Path, task: Task) -> Dict[str, Any]:
    """
    Read the configuration file given by the user.

    Ensures that the format corresponds to the TOML file template.

    Parameters
    ----------
    config_file : Path
        Path to a configuration file (JSON of TOML).
    task : Task
        Task performed by the network (e.g. classification).

    Returns
    -------
    Dict[str, Any]
        Config dictionary with the training parameters extracted from the config file.

    Raises
    ------
    ClinicaDLConfigurationError
        If configuration file is not a TOML file.
    ClinicaDLConfigurationError
        If a section in the TOML file is not valid.
    ClinicaDLConfigurationError
        If an option in the TOML file is not valid.
    """
    if config_file.suffix != ".toml":
        raise ClinicaDLConfigurationError(
            f"Config file {config_file} should be a TOML file."
        )

    user_dict = toml.load(config_file)
    if "Random_Search" in user_dict:
        del user_dict["Random_Search"]

    # get the template
    clinicadl_root_dir = Path(__file__).parents[1]
    config_path = clinicadl_root_dir / "resources" / "config" / "train_config.toml"
    config_dict = toml.load(config_path)
    # Check that TOML file has the same format as the one in clinicadl/resources/config/train_config.toml
    user_dict = path_decoder(user_dict)
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

    # task dependent
    user_dict = remove_unused_tasks(
        user_dict, task.value
    )  # TODO : change remove_unused_tasks so that it accepts Task objects

    train_dict = dict()
    # Fill train_dict from TOML files arguments
    for config_section in user_dict:
        for key in user_dict[config_section]:
            train_dict[key] = user_dict[config_section][key]

    return train_dict


def get_model_list(architecture=None, input_size=None, model_layers=False):
    """
    Print the list of models available in ClinicaDL.
    If --architecture is given, information about how to use this model will be displayed.
    If --model_layers flag is added, this pipeline will show the whole model layers.
    If --input_size is added, it will show the whole model layers with chosen input shape.
    """
    from inspect import getmembers, isclass

    import clinicadl.network as network_package

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

        shape_str = f"\n\t The input must be in the shape {shape_str}".expandtabs(4)
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
            channel, shape_list = input_size.split("@")
            args.remove("self")
            kwargs = dict()
            kwargs["input_size"] = [int(channel)] + [
                int(x) for x in shape_list.split("x")
            ]
            kwargs["gpu"] = False

            model = model_class(**kwargs)

            print(f"Input size: {input_size}")
            print("Model layers:")
            print(model.layers)


def merge_cli_and_config_file_options(task: Task, **kwargs) -> Dict[str, Any]:
    """
    Merges options from the CLI (passed by the user) and from the config file
    (if it exists).

    Priority is given to options passed by the user via the CLI. If it is not
    provided, it will look for the option in the possible config file.
    If an option is not passed by the user and not found in the config file, it will
    not be in the output.

    Parameters
    ----------
    task : Task
        The task that is performed (e.g. classification).

    Returns
    -------
    Dict[str, Any]
        A dictionary with training options.
    """
    options = {}
    if kwargs["config_file"]:
        options = extract_config_from_toml_file(
            Path(kwargs["config_file"]),
            task,
        )
    del kwargs["config_file"]
    for arg in kwargs:
        if (
            click.get_current_context().get_parameter_source(arg)
            == ParameterSource.COMMANDLINE
        ):
            options[arg] = kwargs[arg]

    return options
