# coding: utf8
# TODO: create a folder for generate/ prepare_data/ data to deal with capsDataset objects ?
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from clinicadl.dataset.transforms import extraction as extraction
from clinicadl.utils.enum import ExtractionMethod
from clinicadl.utils.exceptions import ClinicaDLTSVError

logger = getLogger("clinicadl")

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"


class CapsDatasetOutput(BaseModel):
    """
    Output from the CapsDataset.

    Args:
        image (torch.Tensor): Image tensor.
    """

    # equivalenbt to the futur BidsPath from clinicaIO??
    elem: torch.Tensor
    participant_id: Union[int, str]
    session_id: Union[int, str]
    label: Optional[Union[float, int]] = None
    img_idx: Optional[Union[int, str]] = None
    elem_idx: Optional[Union[int, str]] = None
    image_path: Optional[Path] = None
    # domain: Optional[int]=None
    mode: ExtractionMethod

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)


def tsv_to_df(tsv_path: Path):
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except FileNotFoundError:
        raise ClinicaDLTSVError(
            "The TSV file you gave is not a file.\nError explanations:\n"
            f"\t- Clinica expected the following path to be a file: {tsv_path}\n"
            "\t- If you gave relative path, did you run Clinica on the good folder?"
        )
    df = check_df(df)
    return df


def check_df(df: pd.DataFrame):
    if not {PARTICIPANT_ID, SESSION_ID}.issubset(set(df.columns.values)):
        raise ClinicaDLTSVError(
            f"the data file is not in the correct format."
            f"Columns should include {PARTICIPANT_ID, SESSION_ID}"
        )
    df.reset_index(inplace=True)
    return df


def insensitive_glob(pattern_glob: str, recursive: bool = False) -> List[str]:
    """This function is the glob.glob() function that is insensitive to the case.

    Parameters
    ----------
    pattern_glob : str
        Sensitive-to-the-case pattern.

    recursive : bool, optional
        Recursive parameter for `glob.glob()`.
        Default=False.

    Returns
    -------
    List[str] :
        Insensitive-to-the-case pattern.
    """

    def make_case_insensitive_pattern(c: str) -> str:
        return "[%s%s]" % (c.lower(), c.upper()) if c.isalpha() else c

    insensitive_pattern = "".join(map(make_case_insensitive_pattern, pattern_glob))
    return glob(insensitive_pattern, recursive=recursive)
