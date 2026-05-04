import re
from pathlib import Path

from typing_extensions import Self

from .typing import PathType


class BidsFile:
    """
    A class modelling a :term:`BIDS` file.

    This class will parse the filename and extract the suffix, the extension and the entities

    Parameters
    ----------
    path : str | Path
        A file that respects the :bids:`BIDS file naming conventions <common-principles.html#filenames>`.
    """

    def __init__(self, path: PathType):
        self.path = Path(path).resolve()
        self.entities = self._get_entities(self.path.name)
        self.suffix = self._get_suffix(self.path.name)
        self.extension = self._get_extension(self.path.name)
        assert self.suffix

    @staticmethod
    def _get_extension(filename: str) -> str:
        """
        Gets the total extension of a file.
        'file.nii.gz' will return '.nii.gz'.
        """
        extension = "".join(filename.partition(".")[1:])
        assert extension, f"'{filename}' is not a valid file: there is no extension!"
        return extension

    @staticmethod
    def _get_suffix(filename: str) -> str:
        """
        Gets the suffix from a BIDS filename.
        """
        suffix = filename.partition(".")[0].split("_")[-1]
        assert suffix.isalnum(), f"'{filename}' is not a valid BIDS file: the suffix is not alphanumerical (got '{suffix}')!"
        return suffix

    @staticmethod
    def _get_entities(filename: str) -> dict[str, str]:
        """
        Gets all the (key, value) entities from a filename.
        """
        entities = filename.partition(".")[0].split("_")[:-1]
        entities = [BidsEntity(entity) for entity in entities]

        return {entity.key: entity.value for entity in entities}


class BidsEntity(str):
    """
    A :bids:`BIDS entity <common-principles.html#entities>`.

    Examples
    --------
    .. code-block::
        >>> bids = BidsEntity("trc-18FFDG")
        >>> bids.key
        'trc'
        >>> bids.value
        '18FFDG'
    """

    def __init__(self, entity: str):
        if "-" not in entity:
            raise ValueError(
                f"A BIDS entity must be of the form '<key>-<value>'. Got '{entity}'"
            )
        self.key, _, self.value = entity.partition("-")
        assert self.key.isalnum(), f"They key of a BIDS entity must be an alphanumeric string. Got: '{self.key}'"
        assert self.value.isalnum(), f"They value of a BIDS entity must be an alphanumeric string. Got: '{self.value}'"

    @classmethod
    def from_key_value(cls, key: str, value: str | int) -> Self:
        """
        To create a BIDS entity from a key and a value.

        Parameters
        ----------
        key : str
            The key (alphanumerical string).
        value : str | int
            The value (alphanumerical string or int).

        Return
        ------
        BidsEntity
            The BIDS entity.
        """
        return cls("-".join((str(key), str(value))))


class Subject(BidsEntity):
    """
    A :py:class:`BidsEntity` representing a :bids:`subject <appendices/entities.html#sub>` (a.k.a. a
    "participant").
    """

    pattern = re.compile("sub-.*")

    def __init__(self, entity: str):
        super().__init__(entity)
        assert (
            self.key == "sub"
        ), f"A participant id must start with 'sub' (e.g., 'sub-001'). Got '{self.key}'"

    @classmethod
    def from_value(cls, value: str) -> Self:
        """
        To create a ``Subject`` from the value of the participant id.

        Parameters
        ----------
        value : str
            The id of the participant.

        Return
        ------
        Subject
            The associated ``Subject``.
        """
        return cls.from_key_value("sub", value)


class Session(BidsEntity):
    """
    A :py:class:`BidsEntity` representing a :bids:`session <appendices/entities.html#ses>`.
    """

    pattern = re.compile("ses-.*")

    def __init__(self, entity: str):
        super().__init__(entity)
        assert (
            self.key == "ses"
        ), f"A session id must start with 'ses' (e.g., 'ses-M000'). Got '{self.key}'"

    @classmethod
    def from_value(cls, value: str) -> Self:
        """
        To create a ``Session`` from the value of the session id.

        Parameters
        ----------
        value : str
            The id of the session.

        Return
        ------
        Session
            The associated ``Session``.
        """
        return cls.from_key_value("ses", value)
