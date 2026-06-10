from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)


def remove_non_empty_dir(dir_path: Path) -> None:
    """
    Remove a non-empty directory using only pathlib.

    Parameters
    ----------
    dir_path : Path
        Path to the directory to remove.
    """
    if dir_path.exists() and dir_path.is_dir():
        for item in dir_path.iterdir():
            if item.is_dir():
                remove_non_empty_dir(item)
            else:
                item.unlink()  # Remove files
        dir_path.rmdir()  # Remove the now-empty directory
    else:
        logger.log(f"{dir_path} does not exist or is not a directory.")
