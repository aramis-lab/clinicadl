from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.fields import ModelPrivateAttr
from typing_extensions import Self

from clinicadl.dictionary.words import NAME, READER
from clinicadl.utils.exceptions import (
    CannotReadFieldError,
    CannotReadJsonFieldError,
    MissingFieldsError,
    MissingFieldsJsonError,
    NotInterpretableJsonError,
    WrongFieldsError,
    WrongFieldsJsonError,
)
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.typing import PathType

CONFIG = "Config"

FieldReaderType = Callable[[dict[str, Any]], Any]
FieldReadersType = dict[str, FieldReaderType]


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
    def get_fields(cls, computed: bool = False) -> list[str]:
        """
        Gets the list of the fields in the config class.

        Parameters
        ----------
        computed : bool, default=False
            Whether to also return computed fields.

        Returns
        -------
        list[str]
            The list of the field names.
        """
        fields = list(cls.model_fields.keys())
        if computed:
            fields += list(cls.model_computed_fields.keys())

        return fields

    def to_raw_dict(self, exclude: Optional[Sequence[str]] = None) -> dict[str, Any]:
        """
        Returns the config class as a dictionary, but **not** serialized.

        Parameters
        ----------
        exclude : Optional[Sequence[str]], default=None
            Potential fields to exclude.

        Returns
        -------
        dict[str, Any]
            The raw config class as a dict.
        """
        d = self.__dict__.copy()

        if exclude:
            for name in exclude:
                del d[name]

        for name, value in d.items():
            if hasattr(value, "to_raw"):
                to_raw_dict = getattr(value, "to_raw")
                if callable(to_raw_dict):
                    d[name] = to_raw_dict()

        return d

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """
        Customized version of :py:method:`pydantic.BaseModel.model_dump`.

        Returns
        -------
        dict[str, Any]
            The serialized config class.
        kwargs
            Any argument accepted by :py:method:`pydantic.BaseModel.model_dump`.
        """
        return _order_dict(self.model_dump(**kwargs, serialize_as_any=True))

    def to_json(self, json_path: PathType, overwrite: bool = False, **kwargs) -> None:
        """
        Writes the serialized config class to a ``JSON`` file.

        Parameters
        ----------
        json_path : PathType
            Path to the ``JSON`` file.
        overwrite : bool
            Whether to overwrite the file if it exists.
        """
        write_json(
            json_path=json_path, data=self.to_dict(**kwargs), overwrite=overwrite
        )

    @classmethod
    def from_dict(cls, dict_: dict[str, Any], **kwargs) -> Self:
        """
        To create a ``ClinicaDLConfig`` from a dictionary.

        Parameters
        ----------
        dict_ : dict[str, Any]
            The dictionary.
        kwargs
            Any field to overwrite.

        Returns
        -------
        ClinicaDLConfig
            The config class.
        """
        dict_ = cls._check_dict(dict_)

        for field, value in dict_.items():
            if field not in kwargs:
                reader = cls._get_reader(field)
                kwargs[field] = cls._read_anything(value, field, reader=reader)

        try:
            return cls(**kwargs)
        except ValidationError as e:
            raise CannotReadFieldError(error=e, object_name=cls._get_name()) from e

    @classmethod
    def from_json(cls, json_path: PathType, **kwargs) -> Self:
        """
        To create a ``ClinicaDLConfig`` from a ``JSON`` file.

        Parameters
        ----------
        json_path : PathType
            Path to the ``JSON`` file.
        kwargs
            Any field to overwrite.

        Returns
        -------
        ClinicaDLConfig
            The config class.
        """
        json_path = Path(json_path)
        dict_ = cls._read_json(json_path=json_path)

        try:
            return cls.from_dict(dict_, **kwargs)
        except MissingFieldsError as exc:
            raise MissingFieldsJsonError(exc, json_path) from exc
        except WrongFieldsError as exc:
            raise WrongFieldsJsonError(exc, json_path) from exc
        except CannotReadFieldError as exc:
            raise CannotReadJsonFieldError(exc, json_path) from exc

    @staticmethod
    def serialize_anything(value: Any) -> Any:
        """
        To serialize any object in ``ClinicaDL``.
        Serialization of an object can be customized with a method
        ``to_dict``. Otherwise, the raw object
        will be returned and serialization will be done by pydantic.

        Parameters
        ----------
        value : Any
            The value of the field.

        Returns
        -------
            The serialized field.
        """
        if hasattr(value, "to_dict"):
            to_dict = getattr(value, "to_dict")
            if callable(to_dict):
                return _order_dict(to_dict())

        return value

    @classmethod
    def _check_dict(cls, dict_: dict[str, Any]) -> dict[str, Any]:
        """
        Checks the input of :py:meth:`from_dict`.
        """
        fields_in_dict = set(dict_)
        expected_fields = set(cls.get_fields())

        diff = expected_fields.difference(fields_in_dict)
        if len(diff) > 0:
            raise MissingFieldsError(fields=list(diff))

        diff = fields_in_dict.difference(expected_fields)
        if len(diff) > 0:
            raise WrongFieldsError(fields=list(diff), object_name=cls._get_name())

        return dict_

    @classmethod
    def _read_anything(
        cls, value: Any, field: str, reader: Optional[FieldReaderType] = None
    ) -> Any:
        """
        To read any serialized object.
        It will try to use the reader.
        """
        if reader:
            try:
                return reader(value)
            except Exception as e:
                raise CannotReadFieldError(
                    field_names=[field], object_name=cls._get_name()
                ) from e
        else:
            return value

    @classmethod
    def _read_json(cls, json_path: PathType) -> dict[str, Any]:
        """
        Reads the serialized config class from a JSON file.
        """
        config_dict = read_json(json_path=json_path)

        if not isinstance(config_dict, dict):
            raise NotInterpretableJsonError(json_path, cls._get_name())

        return config_dict

    @field_serializer("*", when_used="always")
    def _field_serializer(self, value: Any) -> Any:
        """
        To serialize a fields.
        """
        return self.serialize_anything(value)

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class."""
        return cls.__name__

    @classmethod
    def _get_reader(cls, field: str) -> Optional[FieldReaderType]:
        """Gets the reader for a field."""
        cls._FIELD_READERS: ModelPrivateAttr
        if cls.model_fields[field].json_schema_extra:  # pylint: disable=unsubscriptable-object
            return cls.model_fields[field].json_schema_extra.get(READER, None)  # pylint: disable=no-member, disable=unsubscriptable-object

        return None


T = TypeVar("T")


class ObjectConfig(ClinicaDLConfig, ABC, Generic[T]):
    """
    Base config class associated to a Python object.

    The user can then get the parametrized object with
    the method :py:meth:`get_object`.
    """

    @computed_field
    @property
    def name(self) -> str:
        """The name of the class associated to this config class."""
        return self._get_name()

    def get_object(self, **kwargs: Any) -> T:
        """
        Returns the object associated to this configuration,
        parametrized with the parameters passed by the user.

        Returns
        -------
        T
            The parametrized object.
        """
        associated_class = self._get_class()
        parameters = self._get_parameters(**kwargs)
        return associated_class(**parameters)

    @classmethod
    def _check_dict(cls, dict_: dict[str, Any]) -> dict[str, Any]:
        """
        Checks the input of :py:meth:`from_dict`.
        """
        dict_ = deepcopy(dict_)
        if NAME in dict_:
            assert (
                dict_[NAME] == cls._get_name()
            ), f"The input dictionary is associated to {dict_[NAME]}, not to {cls._get_name()}."
            del dict_[NAME]
        return super()._check_dict(dict_)

    @classmethod
    @abstractmethod
    def _get_class(cls) -> type[T]:
        """Returns the class associated to this config class."""

    def _get_parameters(self, **kwargs: Any) -> dict[str, Any]:
        """
        Gets the parameters of the class associated class.
        If some parameters has been passed via ``ObjectConfig``
        (or any class with a ``get_object`` method) they are converted
        to the underlying objects.
        """
        params = {}
        for field, value in self:
            if hasattr(value, "get_object") and callable(
                get_object := getattr(value, "get_object")
            ):
                params[field] = get_object(**kwargs)
            else:
                params[field] = value

        return params

    @classmethod
    def _get_name(cls) -> str:
        """Returns the name of the class associated to this config class."""
        return cls.__name__.replace(CONFIG, "")


TConfig = TypeVar("TConfig", bound=ObjectConfig)


class ObjectOrConfig(BaseModel, Generic[T, TConfig]):
    """
    To handle fields that accept an object or the associated config.
    """

    value: Union[T, TConfig]

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def __init__(self, value: Union[T, TConfig]):
        super().__init__(value=value)

    def get_object(self, **kwargs: Any) -> T:
        """
        Returns the object. If a config class was passed,
        it is converted to the underlying object.

        Returns
        -------
        T
            The parametrized object.
        """
        if isinstance(self.value, ObjectConfig):
            return self.value.get_object(**kwargs)
        else:
            return self.value

    def to_raw(self) -> Union[T, TConfig]:
        """
        Unwraps the value of the field.

        Returns
        -------
        Union[T, TConfig]
            The raw value of the field.
        """
        return self.value

    def to_dict(self) -> Union[T, dict[str, Any]]:
        """
        To serialize the current field.

        Returns
        -------
        Union[T, dict[str, Any]]
            The serialized field.
        """
        return ClinicaDLConfig.serialize_anything(self.value)

    @classmethod
    def from_value(cls, value: Union[Self, Union[T, TConfig]]) -> Self:
        """
        To safely create an ``ObjectOrConfig`` from the value of the underlying
        field.

        Useful for pydantic field validators.

        Parameters
        ----------
        value : Union[Self, Union[T, TConfig]]
            The field value or an ``ObjectOrConfig``.

        Returns
        -------
        Self
            The ``ObjectOrConfig``.
        """
        if isinstance(value, cls):
            return value
        return cls(value)

    @classmethod
    def build_reader(
        cls,
        config_reader: Callable[[dict[str, Any]], TConfig],
    ) -> Callable[[Any], Self]:
        """
        To build a reader to deserialize the serialized object/config.

        If a dict is passed, the reader will use ``config_reader``,
        otherwise it will not modify the input.

        Parameters
        ----------
        config_reader : Callable[[dict[str, Any]], TConfig]
            The function to read the serialized config class.

        Returns
        -------
        Callable[[Any], Self]
            The reader to read the serialized object/config.
        """

        def reader(serialized_obj: Any) -> Self:
            if isinstance(serialized_obj, dict):
                obj = config_reader(serialized_obj)
            else:
                obj = serialized_obj
            return cls(obj)

        return reader


class SequenceOfObjects(BaseModel, Generic[T, TConfig]):
    """
    To handle fields that are sequences of objects or the associated configs.
    """

    values: list[ObjectOrConfig[T, TConfig]]

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def __init__(
        self, values: Sequence[Union[Union[T, TConfig], ObjectOrConfig[T, TConfig]]]
    ):
        super().__init__(
            values=[
                ObjectOrConfig(obj) if not isinstance(obj, ObjectOrConfig) else obj
                for obj in values
            ]
        )

    def get_object(self, **kwargs: Any) -> list[T]:
        """
        Returns the list of objects. If config classes were passed,
        they are converted to the underlying object.

        Returns
        -------
        list[T]
            The parametrized objects.
        """
        return [obj.get_object(**kwargs) for obj in self.values]

    def to_raw(self) -> list[Union[T, TConfig]]:
        """
        Unwraps the value of the field.

        Returns
        -------
        list[Union[T, TConfig]]
            The raw value of the field.
        """
        return [value.to_raw() for value in self.values]

    def to_dict(self) -> list[Union[T, dict[str, Any]]]:
        """
        To serialize the current field.

        Returns
        -------
        list[Union[T, dict[str, Any]]]
            The serialized sequence of objects/configs.
        """
        return [obj.to_dict() for obj in self.values]

    @classmethod
    def from_sequence(cls, value: Any, field_name: str) -> Self:
        """
        Checks if the input is a sequence and build a ``SequenceOfConfigs``.

        Useful for pydantic field validators.

        Parameters
        ----------
        value : Any
            The field value.
        field_name : str
            The field name

        Returns
        -------
            The instantiated object.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, Sequence):
            raise ValueError(f"'{field_name}' must be a sequence. Got: {value}")
        return cls(value)

    @classmethod
    def build_reader(
        cls,
        config_reader: Callable[[dict[str, Any]], TConfig],
    ) -> Callable[[list[Any]], Self]:
        """
        To build a reader to deserialize the sequence of
        serialized objects/configs.

        Parameters
        ----------
        reader : Callable[[dict[str, Any]], TConfig]
            The function to read the serialized config class.

        Returns
        -------
        Callable[[list[Any]], Self]
            The reader to read the serialized list of objects/configs.
        """

        def list_reader(serialized_objects: list[Any]) -> Self:
            obj_reader = ObjectOrConfig.build_reader(config_reader)
            return cls([obj_reader(obj) for obj in serialized_objects])

        return list_reader


class DictOfObjects(BaseModel, Generic[T, TConfig]):
    """
    To handle fields that are dictionaries of objects/configs.
    """

    values: dict[str, ObjectOrConfig[T, TConfig]]

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def __init__(
        self, values: dict[str, Union[Union[T, TConfig], ObjectOrConfig[T, TConfig]]]
    ):
        super().__init__(
            values={
                name: ObjectOrConfig(obj)
                if not isinstance(obj, ObjectOrConfig)
                else obj
                for name, obj in values.items()
            }
        )

    def get_object(self, **kwargs: Any) -> dict[str, T]:
        """
        Returns the dictionary of objects. If config classes were passed,
        they are converted to the underlying object.

        Returns
        -------
        dict[str, T]
            The parametrized objects.
        """
        return {name: obj.get_object(**kwargs) for name, obj in self.values.items()}

    def to_raw(self) -> dict[str, Union[T, TConfig]]:
        """
        Unwraps the value of the field.

        Returns
        -------
        dict[str, Union[T, TConfig]]
            The raw value of the field.
        """
        return {name: obj.to_raw() for name, obj in self.values.items()}

    def to_dict(self) -> dict[str, Union[T, dict[str, Any]]]:
        """
        To serialize the current field.

        Returns
        -------
        dict[str, dict[str, Union[T, dict[str, Any]]]]
            The serialized dictionary of objects/configs.
        """
        return {name: obj.to_dict() for name, obj in self.values.items()}

    @classmethod
    def from_dict(cls, value: Any, field_name: str) -> Self:
        """
        Checks if the input is a sequence and build a ``DictOfObjects``.

        Useful for pydantic field validators.

        Parameters
        ----------
        value : Any
            The field value.
        field_name : str
            The field name

        Returns
        -------
            The instantiated object.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError(f"'{field_name}' must be a dict. Got: {value}")
        return cls(value)

    @classmethod
    def build_reader(
        cls,
        config_reader: Callable[[dict[str, Any]], TConfig],
    ) -> Callable[[dict[str, Any]], Self]:
        """
        To build a reader to deserialize the dictionary of
        serialized objects/configs.

        Parameters
        ----------
        reader : Callable[[dict[str, Any]], TConfig]
            The function to read the serialized config class.

        Returns
        -------
        Callable[[dict[str, Any]], Self]
            The reader to read the serialized dictionary of objects/configs.
        """

        def dict_reader(serialized_objects: dict[str, Any]) -> Self:
            obj_reader = ObjectOrConfig.build_reader(config_reader)
            return cls(
                {name: obj_reader(obj) for name, obj in serialized_objects.items()}
            )

        return dict_reader


class KwargsConfig(ObjectConfig[T]):
    """
    Config class to handle kwargs.
    It accepts only ONE field, which must be a :py:class:`DictOfObjects`.
    """

    @field_validator("*", mode="before")
    @classmethod
    def _handle_dict(cls, v: Any, info: ValidationInfo) -> DictOfObjects:
        return DictOfObjects.from_dict(v, field_name=info.field_name)

    @model_validator(mode="after")
    def _count_fields(self) -> Self:
        """
        Check that a the KwargsConfig contain only one field.
        """
        fields = self.get_fields()
        assert (
            len(fields) == 1
        ), f"KwargsConfig should contain only one field, found {fields} here."

        return self

    @classmethod
    def from_dict(cls, dict_: dict[str, Any], **kwargs) -> Self:
        dict_ = cls._check_dict(dict_)

        main_field_name = list(dict_.keys())[0]  # only one field in KwargsConfig

        values: dict = dict_[list(dict_.keys())[0]]
        values.update(kwargs)

        dict_[main_field_name] = values

        try:
            return super().from_dict(dict_)
        except CannotReadFieldError as e:
            wrong_metrics = cls._read_pydantic_error(e.error)
            raise CannotReadFieldError(
                field_names=wrong_metrics, object_name=cls._get_name()
            ) from e

    def to_raw_dict(self, exclude: Optional[Sequence[str]] = None) -> dict[str, Any]:
        dict_ = super().to_raw_dict(exclude)
        main_field_name = self.get_fields()[0]

        return dict_[main_field_name]

    @staticmethod
    def _read_pydantic_error(error: ValidationError) -> list[str]:
        """
        To read a pydantic validation error and determine
        what key of the dict is failing validation.
        """
        list_errors = error.errors()
        wrong_keys = []
        for e in list_errors:
            key = e["loc"][2]
            wrong_keys.append(key)

        return list(set(wrong_keys))


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

    return model_or_field
