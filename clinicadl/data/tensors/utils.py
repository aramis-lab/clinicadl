from pathlib import Path

from clinicadl.dictionary.suffixes import PT
from clinicadl.utils.path import remove_extension


def path_to_tensors(path: Path, tensors_location: Path) -> Path:
    """
    Given a path, returns the location of the associated ``.pt`` file.

    Parameters
    ----------
    path : Path
        The input path.
    tensors_location : Path
        The folder where the tensors are located.

    Returns
    -------
    Path
        The path to the associated ``.pt`` files.
    """
    path = Path(*path.parts[:-1], *tensors_location.parts, *path.parts[-1:])
    path = remove_extension(path)

    return path.with_suffix(PT)
