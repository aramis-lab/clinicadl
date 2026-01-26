from typing import Any, Generic, TypeVar

from typing_extensions import Self

from .config import ObjectConfig
from .json import write_json
from .typing import PathType


class JsonReaderWriter:
    """
    Base class for ``ClinicaDL`` objects that are saved in ``json`` files
    and can be instantiated from ``json`` file.
    """

    def to_json(self, json_path: PathType, overwrite: bool = False) -> None:
        """
        To save the current object in a ``json`` file.

        The file must be legible to :py:meth:`from_json`.

        Parameters
        ----------
        json_path : PathType
            Path to the ``json`` file.
        overwrite : bool, default=False
            Whether to overwrite the file if it exists.
        """
        raise NotImplementedError(
            "Overwrite 'to_json' to save instances in a json file."
        )

    @classmethod
    def from_json(cls, json_path: PathType, **kwargs: Any) -> Self:
        """
        To create the object from a ``json`` file saved with
        :py:meth:`to_json`.

        Parameters
        ----------
        json_path : PathType
            Path to the ``json`` file.
        kwargs : Any
            Any field of the ``json`` to overwrite.

        Returns
        -------
        Self
            The object instantiated from the file.
        """
        raise NotImplementedError(
            "Overwrite 'from_json' to create instance from a json file."
        )


class Serializable:
    """
    Base class for ``ClinicaDL`` objects that must be serializable, i.e. that will
    be converted to dictionaries to be saved.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        To convert the current object to a dictionary.

        The dictionary must be legible to :py:meth:`from_dict`.

        Returns
        -------
        dict[str, Any]
            The converted object.
        """
        raise NotImplementedError(
            "Overwrite 'to_dict' to convert instances to dictionaries."
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> Self:
        """
        To create the object from a dictionary returned by
        :py:meth:`to_dict`.

        Parameters
        ----------
        config_dict : dict[str, Any]
            The input dictionary.
        kwargs : Any
            Any field of the dictionary to overwrite.

        Returns
        -------
        Self
            The object instantiated from the dictionary.
        """
        raise NotImplementedError(
            "Overwrite 'from_dict' to create instance from a dictionary."
        )


Config = TypeVar("Config", bound=ObjectConfig)


class HasConfig(JsonReaderWriter, Serializable, Generic[Config]):
    """
    Base class for ``ClinicaDL`` objects associated with config
    classes.
    """

    config: Config
    _config_type: type[Config]

    def to_json(self, json_path: PathType, overwrite: bool = False) -> None:
        self.config.to_json(json_path, overwrite=overwrite)

    @classmethod
    def from_json(cls: type[Self], json_path: PathType, **kwargs: Any) -> Self:
        config = cls._config_type.from_json(json_path, **kwargs)
        return cls._from_config(config)

    def to_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    @classmethod
    def from_dict(cls: type[Self], config_dict: dict[str, Any], **kwargs: Any) -> Self:
        config = cls._config_type.from_dict(config_dict, **kwargs)
        return cls._from_config(config)

    @classmethod
    def _from_config(cls: type[Self], config: Config) -> Self:
        """To create the object from the associated config."""
        return cls(
            **config.to_raw_dict()
        )  # not get_object here because we want to keep config classes as config classes


def to_json_safe(
    obj: JsonReaderWriter, json_path: PathType, overwrite: bool = False
) -> None:
    """
    To safely save a :py:class:`JsonReaderWriter` object in a ``json`` file.

    If the method ``to_json`` is not implemented, the raw object representation
    will be saved.

    Parameters
    ---------
    obj : JsonReaderWriter
        The object to save.
    json_path : PathType
        The path to the ``json`` file.
    overwrite : bool, default=False
        Whether to overwrite the file if it exists.
    """
    try:
        obj.to_json(json_path, overwrite=overwrite)
    except NotImplementedError:
        write_json(json_path, data=repr(obj), overwrite=overwrite)


C = TypeVar("C", bound=HasConfig)


def equal_if_config_equal(cls: type[C]) -> type[C]:
    """
    Decorator to define the equality of two objects as the
    equality of their configuration class.

    Parameters
    ----------
    cls : type[C]
        The class to decorate.

    Returns
    -------
    type[C]
        The decorated class, with its '__eq__' method overridden.
    """

    def _config_equal(self: C, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.config == other.config

    cls.__eq__ = _config_equal
    return cls
