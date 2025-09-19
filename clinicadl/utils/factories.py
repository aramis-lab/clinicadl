from inspect import signature
from typing import Any, Callable, Dict, List


def get_args_from(func: Callable) -> List[str]:
    """
    Gets the arguments of a function.

    Parameters
    ----------
    func : Callable
        The function.

    Returns
    -------
    List[str]
        The names of the arguments.
    """
    return list(signature(func).parameters.keys())


def get_defaults_from(func: Callable) -> Dict[str, Any]:
    """
    Gets the default values of a function's parameters.

    Parameters
    ----------
    func : Callable
        The function.

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
