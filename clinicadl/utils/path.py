from pathlib import Path
from typing import Optional


def remove_extension(path: Path, degree: Optional[int] = None) -> Path:
    """
    Removes the extension from a path.

    path : Path
        The path.
    degree : Optional[int], default=None
        The degree of the extension to remove. If ``0``, only the last extension will be
        removed. If ``None``, all the extensions will be removed.

    Returns
    -------
    Path
        The new path.
    """
    if degree is not None:
        for _ in range(degree + 1):
            path = path.with_suffix("")
    else:
        while path.suffix:
            path = path.with_suffix("")

    return path
