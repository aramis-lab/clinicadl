# coding: utf8

import os
from os.path import abspath, join
from typing import List

import pytest

# root="/network/lustre/iss02/aramis/projects/clinicadl/data"
root = "/mnt/data/data_CI"


@pytest.fixture(params=["generate_trivial", "generate_random", "generate_shepplogan"])
def generate_commands(request):
    data_caps_folder = join(root, "generate/in/caps")
    if request.param == "generate_trivial":
        output_folder = join(root, "generate/out/trivial_example")
        test_input = [
            "generate",
            "trivial",
            data_caps_folder,
            output_folder,
            "--n_subjects",
            "4",
            "--preprocessing",
            "t1-linear",
        ]
        output_reference = [
            "data.tsv",
            "missing_mods_ses-M00.tsv",
            "sub-TRIV0_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV1_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV2_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV3_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV4_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV5_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV6_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV7_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
        ]
        # output_reference = join(root,"generate/ref/trivial_example")

    elif request.param == "generate_random":
        output_folder = join(root, "generate/out/random_example")
        test_input = [
            "generate",
            "random",
            data_caps_folder,
            output_folder,
            "--n_subjects",
            "10",
            "--mean",
            "4000",
            "--sigma",
            "1000",
            "--preprocessing",
            "t1-linear",
        ]
        output_reference = [
            "data.tsv",
            "missing_mods_ses-M00.tsv",
            "sub-RAND0_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND1_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND10_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND11_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND12_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND13_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND14_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND15_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND16_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND17_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND18_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND19_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND2_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND3_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND4_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND5_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND6_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND7_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND8_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-RAND9_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
        ]
        # output_reference = join(root,"generate/ref/random_example")

    elif request.param == "generate_shepplogan":
        n_subjects = 10
        output_folder = join(root, "generate/out/shepplogan_example")
        test_input = [
            "generate",
            "shepplogan",
            output_folder,
            "--n_subjects",
            f"{n_subjects}",
        ]
        output_reference = (
            ["data.tsv", "missing_mods_ses-M00.tsv"]
            + [
                f"sub-CLNC{i}{j:04d}_ses-M00_space-SheppLogan_axis-axi_channel-single_slice-0_phantom.pt"
                for i in range(2)
                for j in range(n_subjects)
            ]
            + [
                f"sub-CLNC{i}{j:04d}_ses-M00_space-SheppLogan_phantom.nii.gz"
                for i in range(2)
                for j in range(n_subjects)
            ]
        )
        # output_reference = join(root,"generate/ref/shepplogan_example")
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return test_input, output_folder, output_reference


def test_generate(generate_commands):

    test_input, output_folder, output_ref = generate_commands

    flag_error = not os.system("clinicadl " + " ".join(test_input))

    assert flag_error
    assert compare_folder_with_files(abspath(output_folder), output_ref)

    # diff_path = join(root, "generate/diff.txt")
    # if os.path.exists(diff_path):
    #     os.remove(diff_path)
    # os.system(f"diff -r {output_folder} {output_ref} > {diff_path}")
    # filesize = os.path.getsize(diff_path)
    # assert filesize == 0

    # os.remove(diff_path)


def compare_folder_with_files(folder: str, file_list: List[str]) -> bool:
    """Compare file existing in two folders
    Args:
        folder: path to a folder
        file_list: list of files which must be found in folder
    Returns:
        True if files in file_list were all found in folder.
    """

    folder_list = []
    for root, dirs, files in os.walk(folder):
        folder_list.extend(files)

    print(f"Missing files {set(file_list) - set(folder_list)}")
    return set(file_list).issubset(set(folder_list))
