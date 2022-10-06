# coding: utf8

import os
import shutil
import warnings
from os import PathLike
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.testing_tools import clean_folder, compare_folders, tree

warnings.filterwarnings("ignore")


@pytest.fixture(
    params=[
        "image",
        "patch",
        "slice",
        "roi",
    ]
)
def test_name(request):
    return request.param


def extract_commands(cmdopt, tmp_path, test_name):

    # base_dir = Path(cmdopt["input"])
    # input_dir = base_dir / "extract" / "in"
    # ref_dir = base_dir / "extract" / "ref"
    # tmp_out_dir = tmp_path / "extract" / "out"
    # tmp_out_dir.mkdir(parents=True)

    input_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/extract/in"
    )
    ref_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/extract/ref"
    )
    tmp_out_dir = Path(
        "/network/lustre/iss02/aramis/projects/clinicadl/data/dvc/extract/out"
    )
    clean_folder(tmp_out_dir, recreate=True)

    input_caps_directory = input_dir / "caps"
    if test_name == "image":
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_image")
        parameters = {"mode": "image"}

    elif test_name == "patch":
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_patch")
        parameters = {"mode": "patch", "patch_size": 50, "stride_size": 50}

    elif test_name == "slice":
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_slice")
        parameters = {
            "mode": "slice",
            "slice_mode": "rgb",
            "slice_direction": 0,
            "discarded_slices": [0, 0],
        }

    elif test_name == "roi":
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_roi")
        parameters = {
            "mode": "roi",
            "roi_list": ["rightHippocampusBox", "leftHippocampusBox"],
            "uncropped_roi": False,
            "roi_custom_template": "",
            "roi_custom_mask_pattern": "",
        }
    else:
        print(f"Test {test_name} not available.")
        assert 0

    test_extract(input_dir, ref_dir, tmp_out_dir, parameters)


def test_extract(input_dir, ref_dir, out_dir, parameters):

    modalities = ["t1-linear", "pet-linear"]  # , "custom"]
    uncropped_image = [True, False]
    acquisition_label = ["18FAV45", "11CPIB"]
    parameters["save_features"] = True
    parameters["prepare_dl"] = True

    for modality in modalities:
        parameters["preprocessing"] = modality
        if modality == "pet-linear":
            for acq in acquisition_label:
                parameters["acq_label"] = acq
                parameters["suvr_reference_region"] = "pons2"
                parameters["use_uncropped_image"] = False
                parameters[
                    "extract_json"
                ] = f"{modality}-{acq}_mode-{parameters['mode']}.json"
                tsv_file = join(input_dir, f"pet_{acq}.tsv")
                mode = parameters["mode"]
                extract_generic(out_dir, mode, tsv_file, parameters)

        elif modality == "custom":
            parameters["use_uncropped_image"] = True
            parameters[
                "custom_suffix"
            ] = "graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
            parameters["roi_custom_template"] = "Ixi549Space"
            parameters["extract_json"] = f"{modality}_mode-{parameters['mode']}.json"
            tsv_file = join(input_dir, "subjects.tsv")
            mode = parameters["mode"]
            extract_generic(out_dir, mode, tsv_file, parameters)

        elif modality == "t1-linear":
            for flag in uncropped_image:
                parameters["use_uncropped_image"] = flag
                parameters[
                    "extract_json"
                ] = f"{modality}_crop-{not flag}_mode-{parameters['mode']}.json"

                tsv_file = input_dir / "subjects.tsv"
                mode = parameters["mode"]
                extract_generic(out_dir, mode, tsv_file, parameters)
        else:
            raise NotImplementedError(
                f"Test for modality {modality} was not implemented."
            )

    compare_folders(out_dir / f"caps_{mode}", ref_dir / f"caps_{mode}", out_dir)


def extract_generic(out_dir, mode, tsv_file, parameters):

    from os.path import join

    from clinicadl.prepare_data.prepare_data import DeepLearningPrepareData

    DeepLearningPrepareData(
        caps_directory=join(out_dir, f"caps_{mode}"),
        tsv_file=tsv_file,
        n_proc=2,
        parameters=parameters,
    )
