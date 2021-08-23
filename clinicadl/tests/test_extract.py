# coding: utf8

import os

import pytest


@pytest.fixture(
    params=[
        "extract_t1_image",
        "extract_t1_patch",
        "extract_t1_slice",
        "extract_t1_roi",
    ]
)
def cli_extract_t1(request):
    preprocessing = "t1-linear"
    caps_directory = ""
    if request.param == "extract_t1_image":
        test_input = [
            "extract",
            "image",
            caps_directory,
            preprocessing,
        ]
    elif request.param == "extract_t1_patch":
        test_input = [
            "extract",
            "patch",
            caps_directory,
            preprocessing,
            "--patch_size",
            "50",
            "--stride_size",
            "50",
        ]
    elif request.param == "extract_t1_slice":
        test_input = [
            "extract",
            "slice",
            caps_directory,
            preprocessing,
            "slice_mode",
            "rgb",
            "slice_direction",
            "0",
        ]
    elif request.param == "extract_t1_roi":
        test_input = [
            "extract",
            "roi",
            caps_directory,
            preprocessing,
            "--roi_list",
            "['rightHippocampusBox', 'leftHippocampusBox']",
            "--roi_uncropped_image",
            "True",
        ]


def test_extract_t1(cli_extract_t1):
    test_input = cli_extract_t1

    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error


@pytest.fixture(
    params=[
        "extract_pet_image",
        "extract_pet_patch",
        "extract_pet_slice",
        "extract_pet_roi",
    ]
)
def cli_extract_pet(request):
    preprocessing = "pet-linear"
    caps_directory = ""
    if request.param == "extract_t1_image":
        test_input = [
            "extract",
            "image",
            caps_directory,
            preprocessing,
            "--acq_label",
            "av45",
            "--suvr_reference_region",
            "pons2",
        ]
    elif request.param == "extract_t1_patch":
        test_input = [
            "extract",
            "patch",
            caps_directory,
            preprocessing,
            "--patch_size",
            "50",
            "--stride_size",
            "50",
            "--acq_label",
            "av45",
            "--suvr_reference_region",
            "pons2",
        ]
    elif request.param == "extract_t1_slice":
        test_input = [
            "extract",
            "slice",
            caps_directory,
            preprocessing,
            "slice_mode",
            "rgb",
            "slice_direction",
            "0",
            "--acq_label",
            "av45",
            "--suvr_reference_region",
            "pons2",
        ]
    elif request.param == "extract_t1_roi":
        test_input = [
            "extract",
            "roi",
            caps_directory,
            preprocessing,
            "--roi_list",
            "['rightHippocampusBox', 'leftHippocampusBox']",
            "--roi_uncropped_image",
            "True",
            "--acq_label",
            "av45",
            "--suvr_reference_region",
            "pons2",
        ]


def test_extract_pet(cli_extract_pet):
    test_input = cli_extract_pet

    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error
