from pathlib import Path

from clinicadl.tsvtools.utils import remove_non_empty_dir
from clinicadl.utils.exceptions import ClinicaDLConfigurationError
from clinicadl.utils.typing import PathType


class Directory:

    """
    Base class representing a directory structure.

    Attributes
    ----------
        path: Path
            The directory path.
    """

    def __init__(self, path: PathType):
        self.path = Path(path)

    def exists(self) -> bool:
        """Check if the directory exists."""
        return self.path.is_dir()

    def is_empty(self) -> bool:
        """Check if the directory is empty."""
        return not any(self.path.iterdir())

    def remove(self) -> None:
        """Remove the directory."""
        remove_non_empty_dir(self.path)

    def _create(self, overwrite: bool = False, _exists_ok: bool = False):
        """Create the directory if it does not already exist."""

        if self.exists() and not (_exists_ok or overwrite):
            raise ClinicaDLConfigurationError(
                f"Directory ({self.path}) already exists."
            )
        elif overwrite and self.exists():
            self.remove()

        self.path.mkdir(parents=True, exist_ok=_exists_ok)

    def load(self):
        if not self.exists():
            raise ClinicaDLConfigurationError(
                f"Directory ({self.path}) does not exist."
            )
