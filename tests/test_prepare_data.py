# coding: utf8

import os
import shutil
import warnings
from os import PathLike
from os.path import join
from pathlib import Path
from typing import Any, Dict, List

import pytest

from clinicadl.prepare_data.prepare_data_config import PrepareDataConfig
from clinicadl.utils.caps_dataset.data_config import DataConfig
from clinicadl.utils.enum import (
    ExtractionMethod,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.mode.mode_config import ModeConfig, return_mode_config
from clinicadl.utils.preprocessing.preprocessing_config import (
    PreprocessingImageConfig,
    PreprocessingPatchConfig,
    PreprocessingROIConfig,
    PreprocessingSliceConfig,
    return_preprocessing_config,
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
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_image")

        if (tmp_out_dir / "caps_image_flair").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_image_flair")
        shutil.copytree(input_caps_flair_directory, tmp_out_dir / "caps_image_flair")

        config = PrepareDataConfig(
            preprocessing=PreprocessingImageConfig(
                preprocessing_cls=Preprocessing.T1_LINEAR,
            ),
            mode=return_mode_config(Preprocessing.T1_LINEAR)(),
            data=DataConfig(caps_directory=tmp_out_dir / "caps_image"),
        )

    elif test_name == "patch":
        if (tmp_out_dir / "caps_patch").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_patch")
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_patch")

        if (tmp_out_dir / "caps_patch_flair").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_patch_flair")
        shutil.copytree(input_caps_flair_directory, tmp_out_dir / "caps_patch_flair")

        config = PrepareDataConfig(
            preprocessing=PreprocessingPatchConfig(
                preprocessing_cls=Preprocessing.T1_LINEAR,
            ),
            mode=return_mode_config(Preprocessing.T1_LINEAR)(),
            data=DataConfig(caps_directory=tmp_out_dir / "caps_patch"),
        )

    elif test_name == "slice":
        if (tmp_out_dir / "caps_slice").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_slice")
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_slice")

        if (tmp_out_dir / "caps_slice_flair").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_slice_flair")
        shutil.copytree(input_caps_flair_directory, tmp_out_dir / "caps_slice_flair")

        config = PrepareDataConfig(
            preprocessing=PreprocessingSliceConfig(
                preprocessing_cls=Preprocessing.T1_LINEAR,
            ),
            mode=return_mode_config(Preprocessing.T1_LINEAR)(),
            data=DataConfig(caps_directory=tmp_out_dir / "caps_slice"),
        )
    elif test_name == "roi":
        if (tmp_out_dir / "caps_roi").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_roi")
        shutil.copytree(input_caps_directory, tmp_out_dir / "caps_roi")

        if (tmp_out_dir / "caps_roi_flair").is_dir():
            shutil.rmtree(tmp_out_dir / "caps_roi_flair")
        shutil.copytree(input_caps_flair_directory, tmp_out_dir / "caps_roi_flair")
        config = PrepareDataConfig(
            preprocessing=PreprocessingROIConfig(
                preprocessing_cls=Preprocessing.T1_LINEAR,
                roi_list=["rightHippocampusBox", "leftHippocampusBox"],
            ),
            mode=return_mode_config(Preprocessing.T1_LINEAR)(),
            data=DataConfig(caps_directory=tmp_out_dir / "caps_roi"),
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
    config.preprocessing.save_features = True

    for modality in modalities:
        config.preprocessing.preprocessing = Preprocessing(modality)
        config.mode = return_mode_config(Preprocessing(modality))()
        if modality == "pet-linear":
            for acq in acquisition_label:
                config.mode.tracer = Tracer(acq)
                config.mode.suvr_reference_region = SUVRReferenceRegions("pons2")
                config.preprocessing.use_uncropped_image = False
                config.preprocessing.extract_json = (
                    f"{modality}-{acq}_mode-{test_name}.json"
                )
                tsv_file = join(input_dir, f"pet_{acq}.tsv")
                mode = test_name
                extract_generic(out_dir, mode, tsv_file, config)

        elif modality == "custom":
            config.preprocessing.use_uncropped_image = True
            config.mode.custom_suffix = (
                "graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
            )
            if isinstance(config.preprocessing, PreprocessingROIConfig):
                config.preprocessing.roi_custom_template = "Ixi549Space"
            config.preprocessing.extract_json = f"{modality}_mode-{test_name}.json"
            tsv_file = input_dir / "subjects.tsv"
            mode = test_name
            extract_generic(out_dir, mode, tsv_file, config)

        elif modality == "t1-linear":
            for flag in uncropped_image:
                config.preprocessing.use_uncropped_image = flag
                config.preprocessing.extract_json = (
                    f"{modality}_crop-{not flag}_mode-{test_name}.json"
                )
                mode = test_name
                extract_generic(out_dir, mode, None, config)

        elif modality == "flair-linear":
            config.data.caps_directory = Path(
                str(config.data.caps_directory) + "_flair"
            )
            config.preprocessing.save_features = False
            for flag in uncropped_image:
                config.preprocessing.use_uncropped_image = flag
                config.preprocessing.extract_json = (
                    f"{modality}_crop-{not flag}_mode-{test_name}.json"
                )
                mode = f"{test_name}_flair"
                extract_generic(out_dir, mode, None, config)
        else:
            raise NotImplementedError(
                f"Test for modality {modality} was not implemented."
            )

    assert compare_folders(
        out_dir / f"caps_{test_name}_flair",
        ref_dir / f"caps_{test_name}_flair",
        out_dir,
    )
    assert compare_folders(
        out_dir / f"caps_{test_name}", ref_dir / f"caps_{test_name}", out_dir
    )


def extract_generic(out_dir, mode, tsv_file, config: PrepareDataConfig):
    from clinicadl.prepare_data.prepare_data import DeepLearningPrepareData

    config.data.caps_directory = out_dir / f"caps_{mode}"
    config.mode.tsv_file = tsv_file
    config.data.n_proc = 1
    DeepLearningPrepareData(config)
