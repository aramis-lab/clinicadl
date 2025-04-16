import re
from pathlib import Path

from ...enum import ImageModality
from ...file_type import FileType
from .custom import CustomFileType
from .dwi import DWIFileType
from .flair import FlairFileType
from .pet import PETFileType
from .t1w import T1WFileType


def get_file_type(filename: str) -> FileType:
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

    if filename.endswith(ImageModality.T1W):
        return T1WFileType()
    elif filename.endswith(ImageModality.FLAIR):
        return FlairFileType()
    elif filename.endswith(ImageModality.DWI):
        return DWIFileType()
    elif filename.endswith(ImageModality.PET):
        match_trc = re.match(r".*_trc-(?P<tracer>[^_]+)_", filename)
        if match_trc:
            tracer = match_trc.group("tracer")
        else:
            raise ValueError(
                f"Filename '{filename}' does not match expected PET format with tracer."
            )

        match_rec = re.match(r".*_rec-(?P<recon>[^_]+)_", filename)
        if match_rec:
            recon = match_rec.group("recon")
        else:
            recon = None

        return PETFileType(tracer=tracer, reconstruction=recon)

    elif filename.endswith(ImageModality.CUSTOM):
        return CustomFileType(custom_suffix=filename.split("_")[-1])
    else:
        raise ValueError(f"Unknown file type for file: {filename}")
