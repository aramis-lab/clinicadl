import os
import re
from abc import ABC
from typing import Any, Optional, Pattern

from pydantic import computed_field, field_serializer, field_validator
from typing_extensions import Self

from clinicadl.utils.config import ConfigWithName
from clinicadl.utils.dictionary.suffixes import JSON, TSV


class DataType(ConfigWithName, ABC):
    """
    To define the type of data you are interested in within your database.

    It defines the pattern that will be used by ``ClinicaDL`` to get the right files in the
    database.

    As it is common to want to retrieve :term:`NIfTI` files with a specific suffix in a
    specific folder, :py:meth:`from_folder_and_suffix` can help you easily create
    the wanted pattern.

    Parameters
    ----------
    pattern : Union[str, Pattern]
        Anything that can be compiled with :py:class:`re.compile`.
        ``ClinicaDL`` will look for the files like:
        ``*/sub-*/ses-*/{pattern}``.
    key : str
        A name to give to your datatype (e.g. ``T1w``). It must be a **single**
        word (no space). ``ClinicaDL`` will use it to refer to your data.
    description : Union[str], default=None
        A potential description of the data.

    Examples
    --------
    .. code-block::

        import re
        from clinicadl.data.datatypes import DataType

        datatype = DataType(
            pattern="t1/sub-.*_ses-.*_T1w.nii", key="T1", description="T1 weighted MRIs"    # '.*' means 'anything'
        )
        # ClinicaDL will look for all the files like */sub-*/ses-*/t1/sub-.*_ses-.*_T1w.nii

    .. code-block::

        >>> re.match(datatype.pattern, "t1/sub-001_ses-M000_T1w.nii")
        <re.Match object; span=(0, 27), match='t1/sub-001_ses-M000_T1w.nii'>
        >>> re.match(datatype.pattern, "t1/sub-001_ses-M000_pet.nii")
        None

    .. code-block::

        >>> datatype = DataType.from_folder_and_suffix(folder="t1", suffix="T1w", description="T1 weighted MRIs")
        >>> datatype.pattern
        re.compile(r't1/sub-.*_ses-.*_T1w.nii.*', re.UNICODE)
        >>> datatype.key
        'T1w'

    """

    pattern: Pattern
    key: str
    description: Optional[str] = None

    @classmethod
    def from_folder_and_suffix(
        cls, folder: str, suffix: str, description: Optional[str] = None
    ) -> Self:
        """
        To create a ``DataType`` from a folder name and a file suffix.

        This method will automatically create the pattern to retrieve
        the :term:`NIfTI` files like ``*/sub-*/ses-*/{folder}/sub-*_ses-*_{suffix}.nii*``,
        and return the associated ``DataType``.

        Parameters
        ----------
        folder : str
            The name of the folder in ``sub-*/ses-*/`` where to look
            for the data.
        suffix : str
            The suffix of the ``NIfTI`` files to consider. The suffix will also
            be used for the ``key``.
        description : Optional[str], default=None
            A potential description of the data.

        Returns
        -------
        DataType
            The ``DataType``.
        """
        pattern = os.path.join(folder, f"sub-.*_ses-.*_{suffix}.nii.*")
        pattern = re.compile(pattern)

        return cls(pattern=pattern, key=suffix, description=description)

    @computed_field
    @property
    def name(self) -> str:
        """Gets the name of the current class."""
        return type(self).__name__

    @field_validator("key", mode="before")
    @classmethod
    def _validate_name(cls, value: Any) -> Any:
        """Converts strings to regex."""
        if isinstance(value, str) and " " in value:
            raise ValueError(f"No space accepted in 'name'. Got {value}")
        return value

    @field_serializer("pattern")
    def _serialize_pattern(self, pattern: Pattern) -> str:
        """
        Serialize a pattern.
        """
        return pattern.pattern

    @property
    def tsv_filename(self) -> str:
        """
        Builds a filename for a ``tsv`` file saving
        information on this preprocessing.
        """
        return "overview_" + self._filename + TSV

    @property
    def json_filename(self) -> str:
        """
        Builds a filename for a ``json`` file saving
        information on this preprocessing.
        """
        return "default_" + self._filename + JSON

    @property
    def _filename(self) -> str:
        return self.key
