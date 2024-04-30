# coding: utf8

import os
import shutil
import warnings
from os import PathLike
from os.path import join
from pathlib import Path
from typing import Any, Dict, List

import pytest

from clinicadl.prepare_data.prepare_data_config import (
    PrepareDataConfig,
    PrepareDataImageConfig,
    PrepareDataPatchConfig,
    PrepareDataROIConfig,
    PrepareDataSliceConfig,
)
from clinicadl.utils.enum import (
    ExtractionMethod,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from tests.testing_tools import clean_folder, compare_folders

warnings.filterwarnings("ignore")


@pytest.fixture(
    params=[
        "slice",
        "patch",
        "image",
        "roi",
    ]
)
def test_name(request):
    return request.param


def test_prepare_data(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "prepare_data" / "in"
    ref_dir = base_dir / "prepare_data" / "ref"
    tmp_out_dir = tmp_path / "prepare_data" / "out"
    tmp_out_dir.mkdir(parents=True)

    clean_folder(tmp_out_dir, recreate=True)

    input_caps_directory = input_dir / "caps"
    input_caps_flair_directory = input_dir / "caps_flair"
    if test_name == "image":
        if (tmp_out_dir / "caps_image").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_image")
            shutil.rmtree(tmp_out_dir / "caps_image_flair")
        shutil.copytree(input_caps_flair_directory, tmp_out_dir / "caps_image_flair")
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_image")
        config = PrepareDataImageConfig(
            caps_directory=input_caps_directory,
            preprocessing_cls=Preprocessing.T1_LINEAR,
        )

    elif test_name == "patch":
        if (tmp_out_dir / "caps_patch").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_patch")
            shutil.rmtree(tmp_out_dir / "caps_patch_flair")
        shutil.copytree(input_caps_flair_directory, tmp_out_dir / "caps_patch_flair")
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_patch")
        config = PrepareDataPatchConfig(
            caps_directory=input_caps_directory,
            preprocessing_cls=Preprocessing.T1_LINEAR,
        )

    elif test_name == "slice":
        if (tmp_out_dir / "caps_slice").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_slice")
            shutil.rmtree(tmp_out_dir / "caps_slice_flair")
        shutil.copytree(input_caps_flair_directory, tmp_out_dir / "caps_slice_flair")
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_slice")
        config = PrepareDataSliceConfig(
            caps_directory=input_caps_directory,
            preprocessing_cls=Preprocessing.T1_LINEAR,
        )

    elif test_name == "roi":
        if (tmp_out_dir / "caps_roi").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_roi")
            shutil.rmtree(tmp_out_dir / "caps_roi_flair")
        shutil.copytree(input_caps_flair_directory, tmp_out_dir / "caps_roi_flair")
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_roi")
        config = PrepareDataROIConfig(
            caps_directory=input_caps_directory,
            preprocessing_cls=Preprocessing.T1_LINEAR,
            roi_list=["rightHippocampusBox", "leftHippocampusBox"],
        )

    else:
        print(f"Test {test_name} not available.")
        assert 0

    run_test_prepare_data(input_dir, ref_dir, tmp_out_dir, test_name, config)


def run_test_prepare_data(
    input_dir, ref_dir, out_dir, test_name: str, config: PrepareDataConfig
):
    modalities = ["t1-linear", "pet-linear", "flair-linear"]
    uncropped_image = [True, False]
    acquisition_label = ["18FAV45", "11CPIB"]
    config.save_features = True

    for modality in modalities:
        config.preprocessing = Preprocessing(modality)
        if modality == "pet-linear":
            for acq in acquisition_label:
                config.tracer = Tracer(acq)
                config.suvr_reference_region = SUVRReferenceRegions("pons2")
                config.use_uncropped_image = False
                config.extract_json = f"{modality}-{acq}_mode-{test_name}.json"
                tsv_file = join(input_dir, f"pet_{acq}.tsv")
                mode = test_name
                extract_generic(out_dir, mode, tsv_file, config)

        elif modality == "custom":
            config.use_uncropped_image = True
            config.custom_suffix = (
                "graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
            )
            if isinstance(config, PrepareDataROIConfig):
                config.roi_custom_template = "Ixi549Space"
            config.extract_json = f"{modality}_mode-{test_name}.json"
            tsv_file = input_dir / "subjects.tsv"
            mode = test_name
            extract_generic(out_dir, mode, tsv_file, config)

        elif modality == "t1-linear":
            for flag in uncropped_image:
                config.use_uncropped_image = flag
                config.extract_json = (
                    f"{modality}_crop-{not flag}_mode-{test_name}.json"
                )
                mode = test_name
                extract_generic(out_dir, mode, None, config)

        elif modality == "flair-linear":
            config.save_features = False
            for flag in uncropped_image:
                config.use_uncropped_image = flag
                config.extract_json = (
                    f"{modality}_crop-{not flag}_mode-{test_name}.json"
                )
                mode = f"{test_name}_flair"
                extract_generic(out_dir, mode, None, config)
        else:
            raise NotImplementedError(
                f"Test for modality {modality} was not implemented."
            )
    assert compare_folders(
        out_dir / f"caps_{test_name}", ref_dir / f"caps_{test_name}", out_dir
    )


def extract_generic(out_dir, mode, tsv_file, config: PrepareDataConfig):
    from clinicadl.prepare_data.prepare_data import DeepLearningPrepareData

    config.caps_directory = out_dir / f"caps_{mode}"
    config.tsv_file = tsv_file
    config.n_proc = 1
    DeepLearningPrepareData(config)
