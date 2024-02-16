import hashlib
import os
import shutil
import ssl
import tempfile
from collections import namedtuple
from enum import Enum
from functools import partial
from glob import glob
from pathlib import Path, PurePath
from time import localtime, strftime, time
from typing import Callable, Dict, List, Optional, Tuple, Union
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLBIDSError,
    ClinicaDLCAPSError,
)
from clinicadl.utils.logger import cprint

RemoteFileStructure = namedtuple("RemoteFileStructure", ["filename", "url", "checksum"])


def bids_nii(
    modality: str = "t1",
    tracer: str = None,
    reconstruction: str = None,
) -> dict:
    """Return the query dict required to capture PET scans.

    Parameters
    ----------
    tracer : Tracer, optional
        If specified, the query will only match PET scans acquired
        with the requested tracer.
        If None, the query will match all PET sans independently of
        the tracer used.

    reconstruction : ReconstructionMethod, optional
        If specified, the query will only match PET scans reconstructed
        with the specified method.
        If None, the query will match all PET scans independently of the
        reconstruction method used.

    Returns
    -------
    dict :
        The query dictionary to get PET scans.
    """
    import os

    modalities = ("t1", "dwi", "pet", "flair")
    if modality not in modalities:
        raise ClinicaDLArgumentError(
            f"ClinicaDL is Unable to read this modality ({modality}) of images, please chose one from this list: {modalities}"
        )
    elif modality == "pet":
        trc = "" if tracer is None else f"_trc-{tracer}"
        rec = "" if reconstruction is None else f"_rec-{reconstruction}"
        description = f"PET data"
        if tracer:
            description += f" with {tracer} tracer"
        if reconstruction:
            description += f" and reconstruction method {reconstruction}"

        return {
            "pattern": os.path.join("pet", f"*{trc}{rec}_pet.nii*"),
            "description": description,
        }
    elif modality == "t1":
        return {"pattern": "anat/sub-*_ses-*_T1w.nii*", "description": "T1w MRI"}
    elif modality == "flair":
        return {"pattern": "sub-*_ses-*_flair.nii*", "description": "FLAIR T2w MRI"}
    elif modality == "dwi":
        return {"pattern": "dwi/sub-*_ses-*_dwi.nii*", "description": "DWI NIfTI"}


def linear_nii(modality: str, uncropped_image: bool) -> dict:

    if modality not in ("T1w", "T2w", "flair"):
        raise ClinicaDLArgumentError(
            f"ClinicaDL is Unable to read this modality ({modality}) of images"
        )
    elif modality == "T1w":
        needed_pipeline = "t1-linear"
    elif modality == "T2w":
        needed_pipeline = "t2-linear"
    elif modality == "flair":
        needed_pipeline = "flair-linear"

    if uncropped_image:
        desc_crop = ""
    else:
        desc_crop = "_desc-Crop"

    information = {
        "pattern": f"*space-MNI152NLin2009cSym{desc_crop}_res-1x1x1_{modality}.nii.gz",
        "description": f"{modality} Image registred in MNI152NLin2009cSym space using {needed_pipeline} pipeline "
        + (
            ""
            if uncropped_image
            else "and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
        ),
        "needed_pipeline": needed_pipeline,
    }
    return information


class DTIBasedMeasure(str, Enum):
    """Possible DTI measures."""

    FRACTIONAL_ANISOTROPY = "FA"
    MEAN_DIFFUSIVITY = "MD"
    AXIAL_DIFFUSIVITY = "AD"
    RADIAL_DIFFUSIVITY = "RD"


def dwi_dti(measure: Union[str, DTIBasedMeasure], space: Optional[str] = None) -> dict:
    """Return the query dict required to capture DWI DTI images.

    Parameters
    ----------
    measure : DTIBasedMeasure or str
        The DTI based measure to consider.

    space : str, optional
        The space to consider.
        By default, all spaces are considered (i.e. '*' is used in regexp).

    Returns
    -------
    dict :
        The query dictionary to get DWI DTI images.
    """
    measure = DTIBasedMeasure(measure)
    space = space or "*"

    return {
        "pattern": f"dwi/dti_based_processing/*/*_space-{space}_{measure.value}.nii.gz",
        "description": f"DTI-based {measure.value} in space {space}.",
        "needed_pipeline": "dwi_dti",
    }


def pet_linear_nii(
    acq_label: str, suvr_reference_region: str, uncropped_image: bool
) -> dict:

    if uncropped_image:
        description = ""
    else:
        description = "_desc-Crop"

    information = {
        "pattern": str(
            Path("pet_linear")
            / f"*_trc-{acq_label}_pet_space-MNI152NLin2009cSym{description}_res-1x1x1_suvr-{suvr_reference_region}_pet.nii.gz"
        ),
        "description": "",
        "needed_pipeline": "pet-linear",
    }
    return information


def container_from_filename(bids_or_caps_filename: Path) -> Path:
    """Extract container from BIDS or CAPS file.

    Parameters
    ----------
    bids_or_caps_filename : str
        Full path to BIDS or CAPS filename.

    Returns
    -------
    str :
        Container path of the form "subjects/<participant_id>/<session_id>".

    Examples
    --------
    >>> from clinica.utils.nipype import container_from_filename
    >>> container_from_filename('/path/to/bids/sub-CLNC01/ses-M000/anat/sub-CLNC01_ses-M000_T1w.nii.gz')
    'subjects/sub-CLNC01/ses-M000'
    >>> container_from_filename('caps/subjects/sub-CLNC01/ses-M000/dwi/preprocessing/sub-CLNC01_ses-M000_preproc.nii')
    'subjects/sub-CLNC01/ses-M000'
    """
    import os
    import re

    m = re.search(r"(sub-[a-zA-Z0-9]+)/(ses-[a-zA-Z0-9]+)", bids_or_caps_filename)
    if not m:
        raise ValueError(
            f"Input filename {bids_or_caps_filename} is not in a BIDS or CAPS compliant format."
            "It does not contain the participant and session ID."
        )
    subject = m.group(1)
    session = m.group(2)
    return Path("subjects") / subject / session


def read_participant_tsv(tsv_file: Path) -> Tuple[List[str], List[str]]:
    """Extract participant IDs and session IDs from TSV file.

    Parameters
    ----------
    tsv_file: str
        Participant TSV file from which to extract the participant and session IDs.

    Returns
    -------
    participants: List[str]
        List of participant IDs.

    sessions: List[str]
        List of session IDs.

    Raises
    ------
    ClinicaException
        If `tsv_file` is not a file.
        If `participant_id` or `session_id` column is missing from TSV file.

    Examples
    --------
    >>> dframe = pd.DataFrame({
    ...     "participant_id": ["sub-01", "sub-01", "sub-02"],
    ...     "session_id": ["ses-M000", "ses-M006", "ses-M000"],
    ...})
    >>> dframe.to_csv("participants.tsv", sep="\t")
    >>> read_participant_tsv("participant.tsv")
    (["sub-01", "sub-01", "sub-02"], ["ses-M000", "ses-M006", "ses-M000"])
    """
    import pandas as pd

    from clinicadl.utils.exceptions import ClinicaDLException

    try:
        df = pd.read_csv(tsv_file, sep="\t")
    except FileNotFoundError:
        raise ClinicaDLException(
            "The TSV file you gave is not a file.\nError explanations:\n"
            f"\t- Clinica expected the following path to be a file: {tsv_file}\n"
            "\t- If you gave relative path, did you run Clinica on the good folder?"
        )

    for column in ("participant_id", "session_id"):
        if column not in list(df.columns.values):
            raise ClinicaDLException(
                f"The TSV file does not contain {column} column (path: {tsv_file})"
            )

    return (
        [sub.strip(" ") for sub in list(df.participant_id)],
        [ses.strip(" ") for ses in list(df.session_id)],
    )


def get_subject_session_list(
    input_dir: Path,
    subject_session_file: Optional[Path] = None,
    is_bids_dir: bool = True,
    use_session_tsv: bool = False,
    tsv_dir: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """Parse a BIDS or CAPS directory to get the subjects and sessions.

    This function lists all the subjects and sessions based on the content of
    the BIDS or CAPS directory or (if specified) on the provided
    subject-sessions TSV file.

    Parameters
    ----------
    input_dir : PathLike
        A BIDS or CAPS directory path.

    subject_session_file : PathLike, optional
        A subjects-sessions file in TSV format.

    is_bids_dir : bool, optional
        Indicates if input_dir is a BIDS or CAPS directory.
        Default=True.

    use_session_tsv : bool, optional
        Specify if the list uses the sessions listed in the sessions.tsv files.
        Default=False.

    tsv_dir : PathLike, optional
        If TSV file does not exist, it will be created in output_dir.
        If not specified, output_dir will be in <tmp> folder

    Returns
    -------
    subjects : list of str
        Subjects list.

    sessions : list of str
        Sessions list.

    Notes
    -----
    This is a generic method based on folder names. If your <BIDS> dataset contains e.g.:
        - sub-CLNC01/ses-M000/anat/sub-CLNC01_ses-M000_T1w.nii
        - sub-CLNC02/ses-M000/dwi/sub-CLNC02_ses-M000_dwi.{bval|bvec|json|nii}
        - sub-CLNC02/ses-M000/anat/sub-CLNC02_ses-M000_T1w.nii
        get_subject_session_list(<BIDS>, None, True) will return
        ['ses-M000', 'ses-M000'], ['sub-CLNC01', 'sub-CLNC02'].

    However, if your pipeline needs both T1w and DWI files, you will need to check
    with e.g. clinicadl_file_reader_function.
    """

    if not subject_session_file:
        output_dir = tsv_dir if tsv_dir else Path(tempfile.mkdtemp())
        timestamp = strftime("%Y%m%d_%H%M%S", localtime(time()))
        tsv_file = f"subjects_sessions_list_{timestamp}.tsv"
        subject_session_file = output_dir / tsv_file
        create_subs_sess_list(
            input_dir=input_dir,
            output_dir=output_dir,
            file_name=tsv_file,
            is_bids_dir=is_bids_dir,
            use_session_tsv=use_session_tsv,
        )

    return read_participant_tsv(subject_session_file)


def create_subs_sess_list(
    input_dir: Path,
    output_dir: Path,
    file_name: str = None,
    is_bids_dir: bool = True,
    use_session_tsv: bool = False,
):
    """Create the file subject_session_list.tsv that contains the list of the visits for each subject for a BIDS or CAPS compliant dataset.

    Args:
        input_dir (str): Path to the BIDS or CAPS directory.
        output_dir (str): Path to the output directory
        file_name: name of the output file
        is_bids_dir (boolean): Specify if input_dir is a BIDS directory or
            not (i.e. a CAPS directory)
        use_session_tsv (boolean): Specify if the list uses the sessions listed in the sessions.tsv files
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    if not file_name:
        file_name = "subjects_sessions_list.tsv"
    subjs_sess_tsv = open(output_dir / file_name, "w")
    subjs_sess_tsv.write("participant_id" + "\t" + "session_id" + "\n")

    if is_bids_dir:
        path_to_search = input_dir
    else:
        path_to_search = input_dir / "subjects"
    subjects_paths = list(path_to_search.glob("*sub-*"))

    # Sort the subjects list
    subjects_paths.sort()

    if len(subjects_paths) == 0:
        raise IOError("Dataset empty or not BIDS/CAPS compliant.")

    for sub_path in subjects_paths:
        subj_id = sub_path.name

        if use_session_tsv:
            session_df = pd.read_csv(sub_path / subj_id + "_sessions.tsv", sep="\t")
            session_df.dropna(how="all", inplace=True)
            session_list = sorted(list(session_df["session_id"].to_numpy()))
            for session in session_list:
                subjs_sess_tsv.write(subj_id + "\t" + session + "\n")

        else:
            sess_list = list(sub_path.glob("*ses-*"))

            for ses_path in sorted(sess_list):
                session_name = ses_path.name
                subjs_sess_tsv.write(subj_id + "\t" + session_name + "\n")

    subjs_sess_tsv.close()


def insensitive_glob(pattern_glob: str, recursive: Optional[bool] = False) -> List[str]:
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


def determine_caps_or_bids(input_dir: Path) -> bool:
    """Determine if the `input_dir` is a CAPS or a BIDS folder.

    Parameters
    ----------
    input_dir : Path
        The input folder.

    Returns
    -------
    bool :
        True if `input_dir` is a BIDS folder, False if `input_dir`
        is a CAPS folder or could not be determined.
    """
    subjects_dir = input_dir / "subjects"
    groups_dir = input_dir / "groups"

    dir_to_look = subjects_dir if subjects_dir.is_dir() else input_dir
    subjects_sub_folders = _list_subjects_sub_folders(dir_to_look, groups_dir)
    if subjects_dir.is_dir():
        return False
    return len(subjects_sub_folders) > 0


def _list_subjects_sub_folders(root_dir: Path, groups_dir: Path) -> List[Path]:

    warning_msg = (
        f"Could not determine if {groups_dir.parent} is a CAPS or BIDS directory. "
        "Clinica will assume this is a CAPS directory."
    )
    folder_content = [f for f in root_dir.iterdir()]
    subjects_sub_folders = [
        sub for sub in folder_content if (sub.name.startswith("sub-") and sub.is_dir())
    ]
    if len(subjects_sub_folders) == 0 and not groups_dir.is_dir():
        cprint(msg=warning_msg, lvl="warning")
    return subjects_sub_folders


def _common_checks(directory: Path, folder_type: str) -> None:
    """Utility function which performs checks common to BIDS and CAPS folder structures.

    Parameters
    ----------
    directory : PathLike
        Directory to check.

    folder_type : {"BIDS", "CAPS"}
        The type of directory.
    """

    if not isinstance(directory, (Path, str)):
        raise ValueError(
            f"Argument you provided to check_{folder_type.lower()}_folder() is not a string."
        )

    error = ClinicaDLBIDSError if folder_type == "BIDS" else ClinicaDLCAPSError

    if not directory.is_dir():
        raise error(
            f"The {folder_type} directory you gave is not a folder.\n"
            "Error explanations:\n"
            f"\t- Clinica expected the following path to be a folder: {directory}\n"
            "\t- If you gave relative path, did you run Clinica on the good folder?"
        )


def check_bids_folder(bids_directory: Path) -> None:
    """Check if provided `bids_directory` is a BIDS folder.

    Parameters
    ----------
    bids_directory : PathLike
        The input folder to check.

    Raises
    ------
    ValueError :
        If `bids_directory` is not a string.

    ClinicaDLBIDSError :
        If the provided path does not exist, or is not a directory.
        If the provided path is a CAPS folder (BIDS and CAPS could
        be swapped by user). We simply check that there is not a folder
        called 'subjects' in the provided path (that exists in CAPS hierarchy).
        If the provided folder is empty.
        If the provided folder does not contain at least one directory whose
        name starts with 'sub-'.
    """

    _common_checks(bids_directory, "BIDS")

    if (bids_directory / "subjects").is_dir():
        raise ClinicaDLBIDSError(
            f"The BIDS directory ({bids_directory}) you provided seems to "
            "be a CAPS directory due to the presence of a 'subjects' folder."
        )

    if len([f for f in bids_directory.iterdir()]) == 0:
        raise ClinicaDLBIDSError(
            f"The BIDS directory you provided is empty. ({bids_directory})."
        )

    subj = [f for f in bids_directory.iterdir() if f.name.startswith("sub-")]
    if len(subj) == 0:
        raise ClinicaDLBIDSError(
            "Your BIDS directory does not contains a single folder whose name "
            "starts with 'sub-'. Check that your folder follow BIDS standard."
        )


def check_caps_folder(caps_directory: Path) -> None:
    """Check if provided `caps_directory`is a CAPS folder.

    Parameters
    ----------
    caps_directory : Path
        The input folder to check.

    Raises
    ------
    ValueError :
        If `caps_directory` is not a string.

    ClinicaCAPSError :
        If the provided path does not exist, or is not a directory.
        If the provided path is a BIDS folder (BIDS and CAPS could be
        swapped by user). We simply check that there is not a folder
        whose name starts with 'sub-' in the provided path (that exists
        in BIDS hierarchy).

    Notes
    -----
    Keep in mind that a CAPS folder can be empty.
    """
    from clinicadl.utils.exceptions import ClinicaDLCAPSError

    _common_checks(caps_directory, "CAPS")

    sub_folders = [f for f in caps_directory.iterdir() if f.name.startswith("sub-")]
    if len(sub_folders) > 0:
        error_string = (
            "Your CAPS directory contains at least one folder whose name "
            "starts with 'sub-'. Check that you did not swap BIDS and CAPS folders.\n"
            "Folder(s) found that match(es) BIDS architecture:\n"
        )
        for directory in sub_folders:
            error_string += f"\t{directory}\n"
        error_string += (
            "A CAPS directory has a folder 'subjects' at its root, in which "
            "are stored the output of the pipeline for each subject."
        )
        raise ClinicaDLCAPSError(error_string)


def find_sub_ses_pattern_path(
    input_directory: Path,
    subject: str,
    session: str,
    error_encountered: list,
    results: list,
    is_bids: bool,
    pattern: str,
) -> None:
    """Appends the output path corresponding to subject, session and pattern in results.

    If an error is encountered, its corresponding message is added to the list `error_encountered`.

    Parameters
    ----------
    input_directory : str
        Path to the root of the input directory (BIDS or CAPS).

        .. warning::
            This function does not perform any check on `input_directory`.
            It is assumed that it has been previously checked by either
            `check_bids_directory` or `check_caps_directory`, and that
            the flag `is_bids` has been set accordingly.

    subject : str
        Name given to the folder of a participant (ex: sub-ADNI002S0295).

    session : str
        Name given to the folder of a session (ex: ses-M00).

    error_encountered : List
        List to which errors encountered in this function are added.

    results : List
        List to which the output path corresponding to subject, session
        and pattern is added.

    is_bids : bool
        True if `input_dir` is a BIDS folder, False if `input_dir` is a
        CAPS folder.

    pattern : str
        Define the pattern of the final file.
    """

    input_directory = Path(input_directory)
    if is_bids:
        origin_pattern = input_directory / subject / session
    else:
        origin_pattern = input_directory / "subjects" / subject / session

    current_pattern = origin_pattern / "**" / pattern
    current_glob_found = insensitive_glob(str(current_pattern), recursive=True)
    if len(current_glob_found) > 1:
        # If we have more than one file at this point, there are two possibilities:
        #   - there is a problem somewhere which made us catch too many files
        #           --> In this case, we raise an error.
        #   - we have captured multiple runs for the same subject and session
        #           --> In this case, we need to select one of these runs to proceed.
        #               Ideally, this should be done via QC but for now, we simply
        #               select the latest run and warn the user about it.
        if _are_multiple_runs(current_glob_found):
            selected = _select_run(current_glob_found)
            list_of_found_files_for_reporting = ""
            for filename in current_glob_found:
                list_of_found_files_for_reporting += f"- {filename}\n"
            cprint(
                f"More than one run were found for subject {subject} and session {session} : "
                f"\n\n{list_of_found_files_for_reporting}\n"
                f"Clinica will proceed with the latest run available, that is \n\n-{selected}.",
                lvl="warning",
            )
            results.append(selected)
        else:
            error_str = f"\t*  ({subject} | {session}): More than 1 file found:\n"
            for found_file in current_glob_found:
                error_str += f"\t\t{found_file}\n"
            error_encountered.append(error_str)
    elif len(current_glob_found) == 0:
        error_encountered.append(f"\t* ({subject} | {session}): No file found\n")
    # Otherwise the file found is added to the result
    else:
        results.append(current_glob_found[0])


def _are_multiple_runs(files: List[Path]) -> bool:
    """Returns whether the files in the provided list only differ through their run number.

    The provided files must have exactly the same parent paths, extensions, and BIDS entities
    excepted for the 'run' entity which must be different.

    Parameters
    ----------
    files : List of str
        The files to analyze.

    Returns
    -------
    bool :
        True if the provided files only differ through their run number, False otherwise.
    """

    # Exit quickly if less than one file or if at least one file does not have the entity run
    if len(files) < 2 or any(["_run-" not in f.name for f in files]):
        return False
    try:
        _check_common_parent_path(files)
        _check_common_extension(files)
        common_suffix = _check_common_suffix(files)
    except ValueError:
        return False
    found_entities = _get_entities(files, common_suffix)
    for entity_name, entity_values in found_entities.items():
        if entity_name != "run":
            # All entities except run numbers should be the same
            if len(entity_values) != 1:
                return False
        else:
            # Run numbers should differ otherwise this is a BIDS violation at this point
            if len(entity_values) != len(files):
                return False
    return True


def get_filename_no_ext(filename: str) -> str:
    """Get the filename without the extension.

    Parameters
    ----------
    filename: str
        The full filename from which to extract the extension out.

    Returns
    -------
    filename_no_ext: str
        The filename with extension removed.

    Examples
    --------
    >>> get_filename_no_ext("foo.nii.gz")
    'foo'
    >>> get_filename_no_ext("sub-01/ses-M000/sub-01_ses-M000.tar.gz")
    'sub-01_ses-M000'
    """

    stem = PurePath(filename).stem
    while "." in stem:
        stem = PurePath(stem).stem

    return stem


def _get_entities(files: List[Path], common_suffix: str) -> dict:
    """Compute a dictionary where the keys are entity names and the values
    are sets of all the corresponding entity values found while iterating over
    the provided files.

    Parameters
    ----------
    files : List of Path
        List of paths to get entities of.

    common_suffix : str
        The suffix common to all the files. This suffix will be stripped
        from the file names in order to only analyze BIDS entities.

    Returns
    -------
    dict :
        The entities dictionary.
    """

    from collections import defaultdict

    found_entities = defaultdict(set)
    # found_entities = dict()
    for f in files:
        entities = get_filename_no_ext(f.name).rstrip(common_suffix).split("_")
        for entity in entities:
            entity_name, entity_value = entity.split("-")
            found_entities[entity_name].add(entity_value)

    return found_entities


def _check_common_properties_of_files(
    files: List[Path],
    property_name: str,
    property_extractor: Callable,
) -> str:
    """Verify that all provided files share the same property and return its value.

    Parameters
    ----------
    files : List of Paths
        List of file paths for which to verify common property.

    property_name : str
        The name of the property to verify.

    property_extractor : Callable
        The function which is responsible for the property extraction.
        It must implement the interface `property_extractor(filename: Path) -> str`

    Returns
    -------
    str :
        The value of the common property.

    Raises
    ------
    ValueError :
        If the provided files do not have the same given property.
    """
    extracted_properties = {property_extractor(f) for f in files}
    if len(extracted_properties) != 1:
        raise ValueError(
            f"The provided files do not share the same {property_name}."
            f"The following {property_name}s were found: {extracted_properties}"
        )
    return extracted_properties.pop()


def _get_parent_path(filename: Path) -> str:
    return str(filename.parent)


def _get_extension(filename: Path) -> str:
    return "".join(filename.suffixes)


def _get_suffix(filename: Path) -> str:

    return f"_{get_filename_no_ext(filename.name).split('_')[-1]}"


_check_common_parent_path = partial(
    _check_common_properties_of_files,
    property_name="parent path",
    property_extractor=_get_parent_path,
)


_check_common_extension = partial(
    _check_common_properties_of_files,
    property_name="extension",
    property_extractor=_get_extension,
)


_check_common_suffix = partial(
    _check_common_properties_of_files,
    property_name="suffix",
    property_extractor=_get_suffix,
)


def _select_run(files: List[str]) -> str:
    import numpy as np

    runs = [int(_get_run_number(f)) for f in files]
    return files[np.argmax(runs)]


def _get_run_number(filename: str) -> str:
    import re

    matches = re.match(r".*_run-(\d+).*", filename)
    if matches:
        return matches[1]
    raise ValueError(f"Filename {filename} should contain one and only one run entity.")


def _check_information(information: Dict) -> None:
    if not isinstance(information, (dict, list)):
        raise TypeError(
            "A dict or list of dicts must be provided for the argument 'information'"
        )

    if isinstance(information, list):
        for item in information:
            if not all(elem in item for elem in ["pattern", "description"]):
                raise ValueError(
                    "'information' must contain the keys 'pattern' and 'description'"
                )

            if not all(
                elem in ["pattern", "description", "needed_pipeline"]
                for elem in item.keys()
            ):
                raise ValueError(
                    "'information' can only contain the keys 'pattern', 'description' and 'needed_pipeline'"
                )

            if item["pattern"][0] == "/":
                raise ValueError(
                    "pattern argument cannot start with char: / (does not work in os.path.join function). "
                    "If you want to indicate the exact name of the file, use the format "
                    "directory_name/filename.extension or filename.extension in the pattern argument."
                )
    else:
        if not all(elem in information for elem in ["pattern", "description"]):
            raise ValueError(
                "'information' must contain the keys 'pattern' and 'description'"
            )

        if not all(
            elem in ["pattern", "description", "needed_pipeline"]
            for elem in information.keys()
        ):
            raise ValueError(
                "'information' can only contain the keys 'pattern', 'description' and 'needed_pipeline'"
            )

        if information["pattern"][0] == "/":
            raise ValueError(
                "pattern argument cannot start with char: / (does not work in os.path.join function). "
                "If you want to indicate the exact name of the file, use the format "
                "directory_name/filename.extension or filename.extension in the pattern argument."
            )


def _format_errors(errors: List, information: Dict) -> str:
    error_message = (
        f"Clinica encountered {len(errors)} "
        f"problem(s) while getting {information['description']}:\n"
    )
    if "needed_pipeline" in information and information["needed_pipeline"]:
        error_message += (
            "Please note that the following clinica pipeline(s) must "
            f"have run to obtain these files: {information['needed_pipeline']}\n"
        )
    error_message += "\n".join(errors)

    return error_message


def clinicadl_file_reader(
    subjects: List[str],
    sessions: List[str],
    input_directory: Path,
    information: Dict,
    raise_exception: Optional[bool] = True,
    n_procs: Optional[int] = 1,
):
    """Read files in BIDS or CAPS directory based on participant ID(s).

    This function grabs files relative to a subject and session list according to a glob pattern (using *)

    Parameters
    ----------
    subjects : List[str]
        List of subjects.

    sessions : List[str]
        List of sessions. Must be same size as `subjects` and must correspond.

    input_directory : PathLike
        Path to the BIDS or CAPS directory to read from.

    information : Dict
        Dictionary containing all the relevant information to look for the files.
        The possible keys are:

            - `pattern`: Required. Define the pattern of the final file.
            - `description`: Required. String to describe what the file is.
            - `needed_pipeline` : Optional. String describing the pipeline(s)
              needed to obtain the related file.

    raise_exception : bool, optional
        If True, an exception is raised if errors happen. If not, we return the file
        list as it is. Default=True.

    n_procs : int, optional
        Number of cores used to fetch files in parallel.
        If set to 1, subjects and sessions will be processed sequentially.
        Default=1.

    Returns
    -------
    results : List[str]
        List of files respecting the subject/session order provided in input.

    error_message : str
        Error message which contains all errors encountered while reading the files.

    Raises
    ------
    TypeError
        If `information` is not a dictionary.

    ValueError
        If `information` is not formatted correctly. See function `_check_information`
        for more details.
        If the length of `subjects` is different from the length of `sessions`.

    ClinicaDLCAPSError or ClinicaDLBIDSError
        If multiples files are found for 1 subject/session, or if no file is found.

        .. note::
            If `raise_exception` is False, no exception is raised.

    Notes
    -----
    This function is case-insensitive, meaning that the pattern argument can, for example,
    contain upper case letters that do not exist in the existing file path.

    You should always use `clinicadl_file_reader` in the following manner:

    .. code-block:: python

         try:
            file_list = clinicadl_file_reader(...)
         except ClinicaException as e:
            # Deal with the error

    Examples
    --------
    The paths are shortened for readability.

    You have the full name of a file.

    File `orig_nu.mgz` from FreeSurfer of subject `sub-ADNI011S4105`, session `ses-M00`
    located in mri folder of FreeSurfer output :

    >>> clinicadl_file_reader(
            ['sub-ADNI011S4105'],
            ['ses-M00'],
            caps_directory,
            {
                'pattern': 'freesurfer_cross_sectional/sub-*_ses-*/mri/orig_nu.mgz',
                'description': 'freesurfer file orig_nu.mgz',
                'needed_pipeline': 't1-freesurfer'
            }
        )
    ['/caps/subjects/sub-ADNI011S4105/ses-M00/t1/freesurfer_cross_sectional/sub-ADNI011S4105_ses-M00/mri/orig_nu.mgz']

    You have a partial name of the file.

    File `sub-ADNI011S4105_ses-M00_trc-18FFDG_pet.nii.gz` in BIDS directory.
    Here, filename depends on subject and session name :

    >>> clinicadl_file_reader(
            ['sub-ADNI011S4105'],
            ['ses-M00'],
            bids_directory,
            {
                'pattern': '*18FFDG_pet.nii*',
                'description': 'FDG PET data'
            }
        )
    ['/bids/sub-ADNI011S4105/ses-M00/pet/sub-ADNI011S4105_ses-M00_trc-18FFDG_pet.nii.gz']

    Tricky example.

    Get the file `rh.white` from FreeSurfer :

    This will fail :

    >>> clinicadl_file_reader(
            ['sub-ADNI011S4105'],
            ['ses-M00'],
            caps,
            {
                'pattern': 'rh.white',
                'description': 'right hemisphere of outer cortical surface.',
                'needed_pipeline': 't1-freesurfer'
            }
        )
    * More than 1 file found::
            /caps/subjects/sub-ADNI011S4105/ses-M00/t1/freesurfer_cross_sectional/fsaverage/surf/rh.white
            /caps/subjects/sub-ADNI011S4105/ses-M00/t1/freesurfer_cross_sectional/rh.EC_average/surf/rh.white
            /caps/subjects/sub-ADNI011S4105/ses-M00/t1/freesurfer_cross_sectional/sub-ADNI011S4105_ses-M00/surf/rh.white

    Correct usage (e.g. in pet-surface): pattern string must be 'sub-*_ses-*/surf/rh.white',
    or even more precise: 't1/freesurfer_cross_sectional/sub-*_ses-*/surf/rh.white'
    It then gives: ['/caps/subjects/sub-ADNI011S4105/ses-M00/t1/freesurfer_cross_sectional/sub-ADNI011S4105_ses-M00/surf/rh.white']
    """
    from clinicadl.utils.exceptions import ClinicaDLBIDSError, ClinicaDLCAPSError

    _check_information(information)
    pattern = information["pattern"]
    is_bids = determine_caps_or_bids(input_directory)
    if is_bids:
        check_bids_folder(input_directory)
    else:
        check_caps_folder(input_directory)

    if len(subjects) != len(sessions):
        raise ValueError("Subjects and sessions must have the same length.")
    if len(subjects) == 0:
        return [], ""

    file_reader = _read_files_parallel if n_procs > 1 else _read_files_sequential
    results, errors_encountered = file_reader(
        input_directory,
        subjects,
        sessions,
        is_bids,
        pattern,
        n_procs=n_procs,
    )
    error_message = _format_errors(errors_encountered, information)
    if len(errors_encountered) > 0 and raise_exception:
        if is_bids:
            raise ClinicaDLBIDSError(error_message)
        else:
            raise ClinicaDLCAPSError(error_message)

    return results, error_message


def _read_files_parallel(
    input_directory: Path,
    subjects: List[str],
    sessions: List[str],
    is_bids: bool,
    pattern: str,
    n_procs: int,
) -> Tuple[List[str], List[str]]:
    from multiprocessing import Manager

    from joblib import Parallel, delayed

    manager = Manager()
    shared_results = manager.list()
    shared_errors_encountered = manager.list()
    Parallel(n_jobs=n_procs)(
        delayed(find_sub_ses_pattern_path)(
            input_directory,
            sub,
            ses,
            shared_errors_encountered,
            shared_results,
            is_bids,
            pattern,
        )
        for sub, ses in zip(subjects, sessions)
    )
    results = list(shared_results)
    errors_encountered = list(shared_errors_encountered)
    return results, errors_encountered


def _read_files_sequential(
    input_directory: Path,
    subjects: List[str],
    sessions: List[str],
    is_bids: bool,
    pattern: str,
    **kwargs,
) -> Tuple[List[str], List[str]]:
    errors_encountered, results = [], []
    for sub, ses in zip(subjects, sessions):
        find_sub_ses_pattern_path(
            input_directory, sub, ses, errors_encountered, results, is_bids, pattern
        )
    return results, errors_encountered


def _sha256(path: Path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def fetch_file(remote: RemoteFileStructure, dirname: Optional[Path]) -> Path:
    """Download a specific file and save it into the resources folder of the package.

    Parameters
    ----------
    remote : RemoteFileStructure
        Structure containing url, filename and checksum.

    dirname : str
        Absolute path where the file will be downloaded.

    Returns
    -------
    file_path : str
        Absolute file path.
    """

    if not dirname.exists():
        cprint(msg="Path to the file does not exist", lvl="warning")
        cprint(msg="Stop Clinica and handle this error", lvl="warning")

    file_path = dirname / remote.filename
    # Download the file from `url` and save it locally under `file_name`:
    gcontext = ssl.SSLContext()
    req = Request(remote.url + remote.filename)
    try:
        response = urlopen(req, context=gcontext)
    except URLError as e:
        if hasattr(e, "reason"):
            cprint(msg=f"We failed to reach a server. Reason: {e.reason}", lvl="error")
        elif hasattr(e, "code"):
            cprint(
                msg=f"The server could not fulfill the request. Error code: {e.code}",
                lvl="error",
            )
    else:
        try:
            with open(file_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
        except OSError as err:
            cprint(msg="OS error: {0}".format(err), lvl="error")

    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError(
            f"{file_path} has an SHA256 checksum ({checksum}) from expected "
            f"({remote.checksum}), file may be corrupted."
        )
    return file_path
