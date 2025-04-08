from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict

from pydantic import BaseModel, ConfigDict, computed_field

from clinicadl.dictionary.words import NAME
from clinicadl.utils.iotools.utils import path_decoder, path_encoder
from clinicadl.utils.json import read_json, update_json, write_json


class DefaultFromLibrary(str, Enum):
    YES = "DefaultFromLibrary"


class ClinicaDLConfig(BaseModel):
    """Base pydantic dataclass."""

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_json(cls, json_path: Path):
        """
        Reads the serialized config class from a JSON file.
        """
        json_path = Path(json_path)
        dict_ = read_json(json_path=json_path)
        return cls(**dict_)

    def to_dict(self) -> Dict[str, Any]:
        """
        Customized version of 'model_dump'.

        Returns the serialized config class.
        """
        return _order_dict(self.model_dump())

    def write_json(self, json_path: Path, overwrite: bool = False) -> None:
        """
        Writes the serialized config class to a JSON file.
        """
        write_json(json_path=json_path, data=self.to_dict(), overwrite=overwrite)

    def read_json(self, json_path: Path) -> Dict[str, Any]:
        """
        Reads the serialized config class from a JSON file.
        """
        return read_json(json_path=json_path)

    def update_json(self, json_path: Path) -> None:
        """
        Updates the JSON file with the serialized config class.
        """
        update_json(json_path=json_path, new_data=self.to_dict())


class NewClinicaDLConfig(ClinicaDLConfig, ABC):
    """
    Base config class associated to a Python object.

    The config class will get the default parameters
    of the associated object to complete the arguments
    passed by the user.

    The user can then get the parametrized object with
    the method `get_object`.
    """

    def __init__(self, **kwargs):
        associated_class = self._get_class()
        kwargs = _update_kwargs_with_defaults(
            kwargs, function=associated_class.__init__
        )
        super().__init__(**kwargs)

    @computed_field
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the object associated to this config class.
        """

    @abstractmethod
    def _get_class(self) -> Any:
        """Returns the class associated to this config class."""

    def get_object(self) -> Any:
        """
        Returns the object associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        Any
            The parametrized object.
        """
        associated_class = self._get_class()
        return associated_class(**self.model_dump(exclude={"name"}))


def _order_dict(model_or_field: Any) -> Any:
    """
    To always have the field 'name' at the beginning.

    Recursive function to handle fields that themeselves
    contain 'ClinicaDLConfig' instances.
    """
    if isinstance(model_or_field, dict):
        ordered_dict = OrderedDict(**model_or_field)
        if NAME in ordered_dict:  # always 'name' at the beginning
            ordered_dict.move_to_end(NAME, last=False)

        for key, value in ordered_dict.items():
            ordered_dict[key] = _order_dict(value)

        return ordered_dict

    elif isinstance(model_or_field, (tuple, list)):
        ordered_sequence = []
        for v in model_or_field:
            ordered_sequence.append(_order_dict(v))
        if isinstance(model_or_field, tuple):
            ordered_sequence = tuple(ordered_sequence)

        return ordered_sequence

    return model_or_field


def _update_kwargs_with_defaults(
    config: Dict[str, Any], function: Callable
) -> Dict[str, Any]:
    """
    Updates arguments with the default values from a function.
    """
    defaults = _get_defaults(function)
    for arg, value in config.items():
        if value == DefaultFromLibrary.YES and arg in defaults:
            config[arg] = defaults[arg]

    return config


def _get_defaults(func: Callable) -> Dict[str, Any]:
    """
    Gets the default values of a function's arguments.
    """
    signature = inspect.signature(func)
    defaults = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    return defaults
