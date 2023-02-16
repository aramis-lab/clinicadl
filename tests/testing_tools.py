import pathlib
from os import PathLike
from pathlib import Path
from typing import Dict, List


def ignore_pattern(file_path: pathlib.Path, ignore_pattern_list: List[str]) -> bool:
    if not ignore_pattern_list:
        return False

    for pattern in ignore_pattern_list:
        if pattern in file_path.__str__():
            return True
    return False


def create_hashes_dict(
    path_folder: pathlib.Path, ignore_pattern_list: List[str] = None
) -> Dict[str, str]:
    """
    Computes a dictionary of files with their corresponding hashes

        Args:
            path_folder: starting point for the tree listing.
            ignore_pattern_list: list of patterns to be ignored to create hash dictionary.

        Returns:
            all_files: a dictionary of the form {/path/to/file.extension: hash(file.extension)}
    """
    import hashlib

    def file_as_bytes(input_file):
        with input_file:
            return input_file.read()

    all_files = []
    for file in path_folder.rglob("*"):
        if not ignore_pattern(file, ignore_pattern_list) and file.is_file():
            all_files.append(file)

    dict_hashes = {
        fname.relative_to(path_folder).__str__(): str(
            hashlib.md5(file_as_bytes(open(fname, "rb"))).digest()
        )
        for fname in all_files
    }
    return dict_hashes


def compare_folders_with_hashes(
    path_folder: pathlib.Path,
    hashes_dict: Dict[str, str],
    ignore_pattern_list: List[str] = None,
):
    """
    Compares the files of a folder against a reference

        Args:
            path_folder: starting point for the tree listing.
            hashes_dict: a dictionary of the form {/path/to/file.extension: hash(file.extension)}
            ignore_pattern_list: list of patterns to be ignored to create hash dictionary.
    """
    hashes_new = create_hashes_dict(path_folder, ignore_pattern_list)

    if hashes_dict != hashes_new:
        error_message1 = ""
        error_message2 = ""
        for key in hashes_dict:
            if key not in hashes_new:
                error_message1 += "{0} not found !\n".format(key)
            elif hashes_dict[key] != hashes_new[key]:
                error_message2 += "{0} does not match the reference file !\n".format(
                    key
                )
        raise ValueError(error_message1 + error_message2)


def models_equal(state_dict_1, state_dict_2, epsilon=0):
    import torch

    for key_item_1, key_item_2 in zip(state_dict_1.items(), state_dict_2.items()):
        if torch.mean(torch.abs(key_item_1[1] - key_item_2[1])) > epsilon:
            print(f"Not equivalent: {key_item_1[0]} != {key_item_2[0]}")
            return False
    return True


def tree(dir_: PathLike, file_out: PathLike):
    """Creates a file (file_out) with a visual tree representing the file
    hierarchy at a given directory

    .. note::
        Does not display empty directories.

    """
    from pathlib import Path

    file_content = ""

    for path in sorted(Path(dir_).rglob("*")):
        if path.is_dir() and not any(path.iterdir()):
            continue
        depth = len(path.relative_to(dir_).parts)
        spacer = "    " * depth
        file_content = file_content + f"{spacer}+ {path.name}\n"

    print(file_content)

    Path(file_out).write_text(file_content)


def compare_folders(outdir: PathLike, refdir: PathLike, tmp_path: PathLike) -> bool:
    """
    Compares the file hierarchy of two folders.

        Args:
            outdir: path to the fisrt fodler.
            refdir: path to the second folder.
            tmp_path: path to a temporary folder.
    """

    from filecmp import cmp
    from pathlib import PurePath

    file_out = PurePath(tmp_path) / "file_out.txt"
    file_ref = PurePath(tmp_path) / "file_ref.txt"
    tree(outdir, file_out)
    tree(refdir, file_ref)
    if not cmp(file_out, file_ref):
        with open(file_out, "r") as fin:
            out_message = fin.read()
        with open(file_ref, "r") as fin:
            ref_message = fin.read()
        raise ValueError(
            "Comparison of out and ref directories shows mismatch :\n "
            "OUT :\n" + out_message + "\n REF :\n" + ref_message
        )
    return True


def compare_folder_with_files(folder: str, file_list: List[str]) -> bool:
    """Compare file existing in two folders
    Args:
        folder: path to a folder
        file_list: list of files which must be found in folder
    Returns:
        True if files in file_list were all found in folder.
    """
    import os

    folder_list = []
    for root, dirs, files in os.walk(folder):
        folder_list.extend(files)

    print(f"Missing files {set(file_list) - set(folder_list)}")
    return set(file_list).issubset(set(folder_list))


def clean_folder(path, recreate=True):
    from os import makedirs
    from os.path import abspath, exists
    from shutil import rmtree

    abs_path = abspath(path)
    if exists(abs_path):
        rmtree(abs_path)
    if recreate:
        makedirs(abs_path)
