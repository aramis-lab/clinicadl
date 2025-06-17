import inspect
from enum import Enum
from inspect import signature
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
    args = list(signature(func).parameters.keys())
    defaults = get_defaults_from(func=func)
    return args, defaults


def get_defaults_from(func: Callable) -> Dict[str, Any]:
    """
    Gets the default values of a function's parameters.

    Parameters
    ----------
    func : Callable
        The functiin

    Returns
    -------
    Dict[str, Any]
        The default values in a dict.
    """
    return {
        k: v.default
        for k, v in signature(func).parameters.items()
        if v.default is not v.empty
    }


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
