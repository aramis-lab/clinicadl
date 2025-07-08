from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from pydantic import BaseModel, ConfigDict, PrivateAttr, computed_field, model_validator
from pydantic.fields import ModelPrivateAttr

from clinicadl.dictionary.words import NAME
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.json import read_json, update_json, write_json
from clinicadl.utils.typing import PathType

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

    def to_dict(self, **kwargs) -> Dict[str, Any]:
        """
        Customized version of 'model_dump'.

        Returns the serialized config class.
        """
        return _order_dict(self.model_dump(**kwargs, serialize_as_any=True))

    def write_json(
        self, json_path: PathType, overwrite: bool = False, **kwargs
    ) -> None:
        """
        Writes the serialized config class to a JSON file.
        """
        json_path = Path(json_path)
        write_json(
            json_path=json_path, data=self.to_dict(**kwargs), overwrite=overwrite
        )

    @classmethod
    def from_json(cls, json_path: PathType, **kwargs):
        """
        Reads the serialized config class from a JSON file.
        """
        json_path = Path(json_path)
        dict_ = cls.read_json(json_path=json_path)
        dict_.update(kwargs)
        return cls(**dict_)

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


FieldReaderType = Callable[[dict[str, Any]], ClinicaDLConfig]
FieldReadersType = dict[str, FieldReaderType]


class MultipleConfig(ClinicaDLConfig, ABC):
    """
    Base config class to gather multiple config classes.
    Each field must be of type ClinicaDLConfig and associated with a function to get the associated
    config object from a dictionary (e.g. 'get_network_config').
    """

    _FIELD_READERS: FieldReadersType = {}

    @model_validator(mode="after")
    def validate_readers(self):
        """Checks that all fields have an associated reader."""
        for field, _ in self:
            assert field in self._FIELD_READERS, (
                "A reader function must be specified in 'FIELD_READERS' for each field. "
                f"'{field}' doesn't have associated reader."
            )

        return self

    @classmethod
    def from_json(cls, json_path: PathType, **kwargs):
        """
        Reads the serialized config class from a JSON file.
        """
        json_path = Path(json_path)
        dict_ = cls.read_json(json_path=json_path)
        dict_.update(kwargs)

        for field, values in dict_.items():
            if values is None:
                raise ClinicaDLArgumentError(
                    f"No configuration associated to '{field}' in '{str(json_path)}'!"
                )
            dict_[field] = cls._get_reader(field)(**values)

        return cls(**dict_)

    @classmethod
    def _get_reader(cls, field: str) -> FieldReaderType:
        """Gets the reader for a field."""
        cls._FIELD_READERS: ModelPrivateAttr
        return cls._FIELD_READERS.default[field]  # pylint: disable=no-member


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
