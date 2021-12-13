from typing import Dict, List


def ignore_pattern(file_path: str, ignore_pattern_list: List[str]) -> bool:
    if not ignore_pattern_list:
        return False

    for pattern in ignore_pattern_list:
        if pattern in file_path:
            return True
    return False


def create_hashes_dict(
    path_folder: str, ignore_pattern_list: List[str] = None
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
    import os

    def file_as_bytes(input_file):
        with input_file:
            return input_file.read()

    all_files = []
    for subdir, dirs, files in os.walk(path_folder):
        files.sort()
        for file in files:
            if not ignore_pattern(file, ignore_pattern_list) and not ignore_pattern(
                subdir, ignore_pattern_list
            ):
                all_files.append(os.path.join(subdir, file))

    dict_hashes = {
        fname[len(path_folder) :]: str(
            hashlib.md5(file_as_bytes(open(fname, "rb"))).digest()
        )
        for fname in all_files
    }
    return dict_hashes


def compare_folders_with_hashes(
    path_folder: str, hashes_dict: Dict[str, str], ignore_pattern_list: List[str] = None
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
