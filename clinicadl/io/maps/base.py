from __future__ import annotations

import inspect
from pathlib import Path

from clinicadl.utils.io import remove_non_empty_dir
from clinicadl.utils.typing import PathType


class Directory:
    """
    Base class representing a directory structure.

    Attributes
    ----------
    path : Path
        The :pathlib.Path:`pathlib.Path <>` representing the directory.
    """

    def __init__(self, path: PathType):
        self.path = Path(path).resolve()

    def is_empty(self) -> bool:
        """
        Checks if the directory is empty.

        Returns
        -------
        bool
        """
        return not any(self.path.iterdir())

    def remove(self, non_empty_ok: bool = False) -> None:
        """
        Removes the directory.

        Parameters
        ----------
        non_empty_ok : bool, default=False
            Whether to remove the directory even if it is non-empty.
        """
        if not self.is_empty() and not non_empty_ok:
            raise FileExistsError(
                f"{str(self.path)} is not empty! To confirm that you want to delete it, pass non_empty_ok=True"
            )
        remove_non_empty_dir(self.path)

    def create(self, overwrite: bool = False, exist_ok: bool = False) -> None:
        """
        Creates the directory if it does not already exist.

        Parameters
        ----------
        overwrite : bool, default=False
            Whether to overwrite the current directory.
        exist_ok : bool, default=False
            If the file already exists and ``overwrite=False``, the function succeeds when ``exist_ok=True``.
        """
        if self.path.exists() and not (exist_ok or overwrite):
            raise FileExistsError(
                f"Directory {str(self.path)} already exists. If it's ok, pass exist_ok=True. To overwrite it, pass overwrite=True."
            )
        elif self.path.exists() and overwrite:
            self.remove(non_empty_ok=True)

        self.path.mkdir(parents=True, exist_ok=exist_ok)

        dirs = self._get_child_directories()
        for dir_ in dirs:
            dir_.create(overwrite=overwrite, exist_ok=exist_ok)

        paths = self._get_child_paths()
        for path in paths:
            if not path.suffix:
                path.mkdir(exist_ok=True)

    def read(self) -> None:
        """
        Checks and reads the directory to find the files inside.

        Raises
        ------
        FileNotFoundError
            If an expected directory is missing.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Directory {str(self.path)} does not exist.")

        dirs = self._get_child_directories()
        for dir_ in dirs:
            dir_.read()

        paths = self._get_child_paths()
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"A directory or a file is missing: {str(path)}"
                )

    def _get_child_directories(self) -> list[Directory]:
        """
        Gets the values of all the properties that are Directory.
        If the property is a dict containing Directory, it also returns
        its values in the output list.
        """
        paths = []
        for name, type_ in inspect.getmembers(type(self)):
            if isinstance(type_, property):
                value = getattr(self, name)

                if isinstance(value, Directory):
                    paths.append(value)
                elif isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, Directory):
                            paths.append(v)

        return paths

    def _get_child_paths(self) -> list[Path]:
        """
        Gets the values of all the properties that are Path.
        """
        paths = []
        for name, type_ in inspect.getmembers(type(self)):
            if isinstance(type_, property):
                value = getattr(self, name)
                if isinstance(value, Path):
                    paths.append(value)

        return paths
