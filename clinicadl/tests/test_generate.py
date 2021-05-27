# coding: utf8

import pytest
import os
from os.path import join, exists, abspath


@pytest.fixture(params=["generate_trivial", "generate_random"])
def generate_commands(request):
    if request.param == "generate_trivial":
        data_caps_folder = "data/dataset/OasisCaps_example/"
        output_folder = "data/dataset/trivial_example"
        test_input = [
            "generate",
            "trivial",
            data_caps_folder,
            "t1-linear",
            output_folder,
            "--n_subjects",
            "4",
        ]
        output_reference = [
            "data.tsv",
            "subjects_sessions_list.tsv",
            "missing_mods_ses-M00.tsv",
            "subjects_sessions_list.tsv",
            "sub-TRIV0_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV1_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV2_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV3_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV4_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV5_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV6_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
            "sub-TRIV7_ses-M00_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz",
        ]

    elif request.param == "generate_random":
        data_caps_folder = "data/dataset/OasisCaps_example/"
        output_folder = "data/dataset/random_example"
        test_input = [
            "generate",
            "random",
            data_caps_folder,
            "t1-linear",
            output_folder,
            "--n_subjects",
            "10",
            "--mean",
            "4000",
            "--sigma",
            "1000",
        ]
        output_reference = [
            "data.tsv",
            "subjects_sessions_list.tsv",
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
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return test_input, output_folder, output_reference


def test_generate(generate_commands):

    test_input = generate_commands[0]
    output_folder = generate_commands[1]
    output_ref = generate_commands[2]

    flag_error = not os.system("clinicadl " + " ".join(test_input))

    assert flag_error
    # assert exists(output_folder)
    assert not compare_folder_with_files(abspath(output_folder), output_ref)


def compare_folder_with_files(folder1, list_of_files):
    """Compare file existing in two folders

    Args:
        folder1: first folder to compare
        list_of_files: list of files in a second folder to compare

    Output: list of files not present in folder1
    """

    files1 = []
    for root, dirs, files in os.walk(folder1):
        files1.extend(files)

    return [f for f in files if f not in files1]
