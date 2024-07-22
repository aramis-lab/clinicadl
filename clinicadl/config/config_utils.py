import typing
from enum import Enum
from typing import Any, get_args

import click
from pydantic import BaseModel


def get_default_from_config_class(arg: str, config: BaseModel) -> Any:
    """
    Gets default value for a parameter of a config class.

    Parameters
    ----------
    arg : str
        The name of the parameter.
    config : BaseModel
        The config class.

    Returns
    -------
    Any
        The default value of the parameter.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class ConfigClass(BaseModel):
    ...     parameter: str = "a string"
    >>> config = ConfigClass()
    >>> get_default_from_config_class("parameter", config)
    "a string"

    >>> from pydantic import BaseModel
    >>> class EnumClass(str, Enum):
    ...     OPTION1 = "option1"
    >>> class ConfigClass(BaseModel):
    ...     parameter: EnumClass = EnumClass.OPTION1
    >>> config = ConfigClass()
    >>> get_default_from_config_class("parameter", config)
    "option1"

    >>> from pydantic import BaseModel
    >>> class EnumClass(str, Enum):
    ...     OPTION1 = "option1"
    >>> class ConfigClass(BaseModel):
    ...     parameter: Tuple[EnumClass] = (EnumClass.OPTION1,)
    >>> config = ConfigClass()
    >>> get_default_from_config_class("parameter", config)
    ('option1',)
    """
    default = config.model_fields[arg].default
    if isinstance(default, Enum):
        return default.value
    if isinstance(default, tuple):
        default_ = []
        for d in default:
            if isinstance(d, Enum):
                default_.append(d.value)
            else:
                default_.append(d)
        return tuple(default_)
    return default


def get_type_from_config_class(arg: str, config: BaseModel) -> Any:
    """
    Gets the type of a parameter of a config class.

    If it is a nested type (e.g. List[str]), it will return
    th underlying type (e.g. str). If the parameter is an Enum
    object, it will return the enumeration as a list (see examples).

    Parameters
    ----------
    arg : str
        The name of the parameter.
    config : BaseModel
        The config class.

    Returns
    -------
    Any
        The type of the parameter.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> class ConfigClass(BaseModel):
    ...     parameter: str = "a string"
    >>> config = ConfigClass()
    >>> get_type_from_config_class("parameter", config)
    str

    >>> from pydantic import BaseModel
    >>> from typing import Tuple
    >>> class ConfigClass(BaseModel):
    ...     parameter: Tuple[str] = ["a string"]
    >>> config = ConfigClass()
    >>> get_type_from_config_class("parameter", config)
    str

    >>> from pydantic import BaseModel
    >>> from typing import Optional
    >>> class ConfigClass(BaseModel):
    ...     parameter: Optional[str] = None
    >>> config = ConfigClass()
    >>> get_type_from_config_class("parameter", config)
    str

    >>> from pydantic import BaseModel
    >>> class EnumClass(str, Enum):
    ...     OPTION1 = "option1"
    ...     OPTION2 = "option2"
    >>> class ConfigClass(BaseModel):
    ...     parameter: EnumClass = EnumClass.OPTION1
    >>> config = ConfigClass()
    >>> get_type_from_config_class("parameter", config)
    cllick.Choice(['option1', 'option2'])

    >>> from pydantic import BaseModel
    >>> class EnumClass(str, Enum):
    ...     OPTION1 = "option1"
    ...     OPTION2 = "option2"
    >>> class ConfigClass(BaseModel):
    ...     parameter: Optional[Tuple[EnumClass]] = None
    >>> config = ConfigClass()
    >>> get_type_from_config_class("parameter", config)
    click.Choice(['option1', 'option2'])
    """
    type_ = config.model_fields[arg].annotation
    if isinstance(type_, typing._GenericAlias):  # Tuple or Optional
        type_ = get_args(type_)[0]
        if isinstance(
            type_, typing._GenericAlias
        ):  # original type is something like Tuple[Optional[...]]
            type_ = get_args(type_)[0]
    if issubclass(type_, Enum):
        type_ = click.Choice(list([option.value for option in type_]))

    return type_
