from enum import Enum
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

from clinicadl.utils.dictionary.words import NAME
from clinicadl.utils.exceptions import CannotReadJsonFieldError

from .config import ClinicaDLConfig
from .exceptions import (
    MissingFieldsError,
    MissingFieldsJsonError,
    NotInterpretableJsonError,
    WrongFieldsError,
)
from .json import read_json
from .objects import HasConfig, JsonReaderWriter, Serializable

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


class FactoryFromDict(Protocol[R]):
    def __call__(self, data: dict[str, Any], **kwargs: Any) -> R:
        ...


class FactoryFromJson(Protocol[R]):
    def __call__(self, data: Path, **kwargs: Any) -> R:
        ...


class SafeFactoryFromJson(Protocol[R]):
    def __call__(
        self, data: Path, default: Optional[R]
    ) -> tuple[Optional[R], list[str]]:
        ...


def safe_factory_from_json(
    factory: FactoryFromJson[R],
) -> Callable[[Callable[..., Any]], SafeFactoryFromJson[R]]:
    """
    Builds a decorator to build a factory function to deserialize
    an object serialized in a json file. This factory will not raised errors.

    If some fields of the serialized object cannot be read, they will be reported, and
    the field of the ``default`` argument of the factory will be used to override them.

    If it was impossible to read the serialized object, the factory returns ``None``.

    Parameters
    ----------
    factory : FactoryFromJson[R]
        The factory function to make safe. It must work with json files.

    Returns
    -------
    Callable[[Callable[..., Any]], SafeFactoryFromJson[R]]
        The decorator.
    """

    def decorator(
        func: Callable[..., Any],
    ) -> SafeFactoryFromJson[R]:
        @wraps(func)
        def wrapper(
            data: Path, default: Optional[R] = None
        ) -> tuple[Optional[R], list[str]]:
            if default:
                assert isinstance(default, (HasConfig, ClinicaDLConfig))
            if isinstance(default, HasConfig):
                default = default.config

            try:
                return _recursively_read_json(factory, data, default, [])
            except (
                ValueError,
                NotInterpretableJsonError,
                MissingFieldsJsonError,
                WrongFieldsError,
            ):
                pass

            return None, []

        return wrapper

    return decorator


def _recursively_read_json(
    factory: Callable,
    data: Path,
    default: Optional[ClinicaDLConfig],
    problematic_fields: list[str],
) -> Optional[tuple[R, list[str]]]:
    try:
        return factory(
            data, **{arg: getattr(default, arg) for arg in problematic_fields}
        ), problematic_fields
    except CannotReadJsonFieldError as e:
        if not default:
            return None, []

        problematic_fields += e.error.field_names
        return _recursively_read_json(factory, data, default, problematic_fields)


def factory_from_json(
    object_type: type[R], enum: type[Enum], context: Context, config: bool = False
) -> Callable[[Callable[..., Any]], FactoryFromJson[R]]:
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
    Callable[[Callable[..., Any]], FactoryFromJson[R]]
        The decorator.
    """
    return _build_factory_decorator(
        object_type, enum, context, config, _base_factory_from_json
    )


def factory_from_dict(
    object_type: type[R], enum: type[Enum], context: Context, config: bool = False
) -> Callable[[Callable[..., Any]], FactoryFromDict[R]]:
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
    Callable[[Callable[..., Any]], R]], FactoryFromDict[R]]
        The decorator.
    """

    return _build_factory_decorator(
        object_type, enum, context, config, _base_factory_from_dict
    )


def _build_factory_decorator(
    object_type: type[R],
    enum: type[Enum],
    context: Context,
    config: bool,
    base_factory: Callable[..., R],
) -> Callable[[Callable[..., Any]], Callable[[T], R]]:
    """
    To build decorators that creates factories from
    dict or json file.
    """

    def decorator(
        func: Callable[..., Any],
    ) -> Callable[[T], R]:
        @wraps(func)
        def wrapper(data: T, **kwargs) -> R:
            return base_factory(data, object_type, enum, context, config, kwargs)

        return wrapper

    return decorator


def _base_factory_from_json(
    json_path: Path,
    object_type: type[R],
    enum: type[Enum],
    context: Context,
    config: bool,
    kwargs: dict[str, Any],
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

    return obj.from_json(json_path, **kwargs)


def _base_factory_from_dict(
    config_dict: dict[str, Any],
    object_type: type[R],
    enum: type[Enum],
    context: Context,
    config: bool,
    kwargs: dict[str, Any],
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

    return obj.from_dict(config_dict, **kwargs)
