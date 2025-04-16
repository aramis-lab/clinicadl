import re
from pathlib import Path

from clinicadl.dictionary.suffixes import NII
from clinicadl.utils.typing import PathType

from ...enum import ImageModality
from ...file_type import FileType
from .custom import CustomFileType
from .dwi import DWIFileType
from .flair import FlairFileType
from .pet import PETFileType
from .t1w import T1WFileType


def get_file_type(path: PathType) -> FileType:
    """
    Get the file type based on the file extension.

    Parameters
    ----------
    path : PathType
        The path to the file.

    Returns
    -------
    str
        The file type based on the file extension.
    """

    container, filename = FileType.get_container_and_filename_from_path(path)

    if filename.endswith(ImageModality.T1W.value):
        return T1WFileType()
    elif filename.endswith(ImageModality.FLAIR.value):
        return FlairFileType()
    elif filename.endswith(ImageModality.DWI.value):
        return DWIFileType()
    elif filename.endswith(ImageModality.PET.value):
        match = re.match(r".*trc-(?P<tracer>[^_]+)_rec-(?P<recon>[^_]+)_", filename)
        if match:
            tracer = match.group("tracer")
            recon = match.group("recon")
            return PETFileType(tracer=tracer, reconstruction=recon)
        raise ValueError(
            f"Filename '{filename}' does not match expected PET format with tracer and reconstruction."
        )
    elif filename.endswith(ImageModality.CUSTOM.value):
        return CustomFileType(custom_suffix=filename.split("_")[-1])
    else:
        raise ValueError(f"Unknown file type for file: {filename}")
