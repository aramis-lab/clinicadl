from pathlib import Path

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
