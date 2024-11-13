import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

from pydantic import BaseModel


class DefaultFromLibrary(str, Enum):
    YES = "DefaultFromLibrary"


def get_args_and_defaults(func: Callable) -> Tuple[List[str], Dict[str, Any]]:
    """
    Gets the arguments of a function, as well as the default
    values possibly attached to them.

    Parameters
    ----------
    func : Callable
        The function.

    Returns
    -------
    List[str]
        The names of the arguments.
    Dict[str, Any]
        The default values in a dict.
    """
    signature = inspect.signature(func)
    args = list(signature.parameters.keys())
    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    return args, defaults


def update_config_with_defaults(config: BaseModel, function: Callable) -> None:
    """
    Updates a configuration object with the default values from a function.

    Parameters
    ----------
    config : BaseModel
        the configuration object.
    function : Callable
        the function from which the defaults are fetched.
    """
    _, defaults = get_args_and_defaults(function)
    for arg, value in config:
        if value == DefaultFromLibrary.YES and arg in defaults:
            setattr(config, arg, defaults[arg])
