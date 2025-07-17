from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from pydantic import BaseModel, ConfigDict, computed_field

from clinicadl.dictionary.words import NAME
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.json import read_json, update_json, write_json

CONFIG = "Config"


class ClinicaDLConfig(BaseModel):
    """Base pydantic dataclass."""

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, **kwargs):
        """Useless method but needed for the doc (typing)."""
        super().__init__(**kwargs)

    @classmethod
    def from_json(cls, json_path: Path, **kwargs):
        """
        Reads the serialized config class from a JSON file.
        """
        json_path = Path(json_path)
        dict_ = read_json(json_path=json_path)
        dict_.update(kwargs)
        return cls(**dict_)

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """
        Customized version of 'model_dump'.

        Returns the serialized config class.
        """
        return _order_dict(self.model_dump(**kwargs))

    def write_json(self, json_path: Path, overwrite: bool = False, **kwargs) -> None:
        """
        Writes the serialized config class to a JSON file.
        """
        write_json(
            json_path=json_path, data=self.to_dict(**kwargs), overwrite=overwrite
        )

    @classmethod
    def read_json(cls, json_path: Path) -> Dict[str, Any]:
        """
        Reads the serialized config class from a JSON file.
        """
        config_dict = read_json(json_path=json_path)

        if set(config_dict.keys()) != set(cls.model_fields.keys()):
            raise ClinicaDLArgumentError(
                f"{json_path} is not a valid json file for {cls.__name__}. "
                f"A valid file should contain the keys {list(cls.model_fields.keys())}."
            )

        return config_dict

    def update_json(self, json_path: Path) -> None:
        """
        Updates the JSON file with the serialized config class.
        """
        update_json(json_path=json_path, new_data=self.to_dict())


class ObjectConfig(ClinicaDLConfig, ABC):
    """
    Base config class associated to a Python object.

    The config class will get the default parameters
    of the associated object to complete the arguments
    passed by the user.

    The user can then get the parametrized object with
    the method 'get_object'.
    """

    @computed_field
    @property
    def name(self) -> str:
        """The name of the class associated to this config class."""
        return self._get_name()

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

    @classmethod
    @abstractmethod
    def _get_class(cls) -> Any:
        """Returns the class associated to this config class."""

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return cls.__name__.replace(CONFIG, "")


def update_kwargs_with_defaults(
    config: Dict[str, Any], function: Callable
) -> Dict[str, Any]:
    """
    Updates arguments with the default values from a function.

    Parameters
    ----------
    config : Dict[str, Any]
        The input arguments.
    function : Callable
        The function to retrieve the default values from.

    Returns
    -------
    Dict[str, Any]
        The updated arguments.
    """
    defaults = _get_defaults(function)
    for arg, value in config.items():
        if arg in defaults:
            config[arg] = defaults[arg]

    return config


def _order_dict(model_or_field: Any) -> Any:
    """
    To always have the field 'name' at the beginning.

    Recursive function to handle fields that
    contain themselves 'ClinicaDLConfig' instances.
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
