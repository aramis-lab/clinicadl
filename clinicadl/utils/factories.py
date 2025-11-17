from enum import Enum
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, TypeVar

from clinicadl.dictionary.words import NAME

from .exceptions import (
    MissingFieldsError,
    MissingFieldsJsonError,
    NotInterpretableJsonError,
)
from .json import read_json
from .objects import JsonReaderWriter, Serializable

__all__ = [
    "get_args_from",
    "get_defaults_from",
    "factory_from_json",
    "factory_from_dict",
]


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


Context = dict[str, Any]
T = TypeVar("T")
R = TypeVar("R")


def _build_factory_decorator(
    object_type: type[R],
    enum: type[Enum],
    context: Context,
    config: bool,
    base_factory: Callable[..., R],
) -> Callable[[Callable[[T], R]], Callable[[T], R]]:
    """
    To build decorators that creates factories from
    dict or json file.
    """

    def decorator(
        factory: Callable[[T], R],
    ) -> Callable[[T], R]:
        @wraps(factory)
        def wrapper(data: T) -> R:
            return base_factory(data, object_type, enum, context, config)

        return wrapper

    return decorator


def factory_from_json(
    object_type: type[R], enum: type[Enum], context: Context, config: bool = False
) -> Callable[[Callable[[Path], R]], Callable[[Path], R]]:
    """
    Builds a decorator to create a factory
    function from a list of implemented objects (i.e. the potential
    outputs of the factory). It is mandatory that these objects
    are in the Python context where the factory function is
    created.

    The factory functions created by the decorators works
    with a json file.

    Parameters
    ----------
    enum : type[Enum]
        The :py:class:`enum.Enum` that defines the implemented
        objects.
    object_type : type[R]
        The type of the output objects.
    context : Context
        The Python context where the factory is created.
    config : bool, default=False
        Whether the objects are config classes.

    Returns
    -------
    Callable[[Callable[[Path], R]], Callable[[Path], R]]
        The decorator.
    """
    return _build_factory_decorator(
        object_type, enum, context, config, _base_factory_from_json
    )


def factory_from_dict(
    object_type: type[R], enum: type[Enum], context: Context, config: bool = False
) -> Callable[[Callable[[dict[str, Any]], R]], Callable[[dict[str, Any]], R]]:
    """
    Builds a decorator to create a factory
    function from a list of implemented objects (i.e. the potential
    outputs of the factory). It is mandatory that these objects
    are in the Python context where the factory function is
    created.

    The factory functions created by the decorators works
    with an input dictionary.

    Parameters
    ----------
    enum : type[Enum]
        The :py:class:`enum.Enum` that defines the implemented
        objects.
    object_type : type[S]
        The type of the output objects.
    context : Context
        The Python context where the factory is created.
    config : bool, default=False
        Whether the objects are config classes.

    Returns
    -------
    Callable[[Callable[[dict[str, Any]], R]], Callable[[dict[str, Any]], R]]
        The decorator.
    """

    return _build_factory_decorator(
        object_type, enum, context, config, _base_factory_from_dict
    )


def _base_factory_from_json(
    json_path: Path,
    object_type: type[R],
    enum: type[Enum],
    context: Context,
    config: bool,
) -> R:
    """
    Factory function to create an object from a json file.
    It will check that the object associated
    to the json is implemented and accessible (i.e. in the context).
    Then, it will create the object with parameters inside the json.
    """
    dict_ = read_json(json_path)

    if not isinstance(dict_, dict):
        raise NotInterpretableJsonError(json_path, object_type.__name__)

    if NAME not in dict_:
        err = MissingFieldsError(fields=[NAME])
        raise MissingFieldsJsonError(err, json_path)

    name = enum(dict_[NAME]).value
    if config:
        name += "Config"
    obj: JsonReaderWriter = context[name]

    return obj.from_json(json_path)


def _base_factory_from_dict(
    config_dict: dict[str, Any],
    object_type: type[R],
    enum: type[Enum],
    context: Context,
    config: bool,
) -> R:
    """
    Factory function to create an object from a dict.
    It will check that the object associated
    to the dict is implemented and accessible (i.e. in the context).
    Then, it will create the object with parameters inside the dict.
    """
    if NAME not in config_dict:
        raise MissingFieldsError(fields=[NAME])

    name = enum(config_dict[NAME]).value
    if config:
        name += "Config"
    obj: Serializable = context[name]

    return obj.from_dict(config_dict)
