# coding: utf8

import os
import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")


def clean_folder(path, recreate=True):
    from os import makedirs
    from os.path import abspath, exists
    from shutil import rmtree

    abs_path = abspath(path)
    if exists(abs_path):
        rmtree(abs_path)
    if recreate:
        makedirs(abs_path)


def test_extract():
    import shutil
    from os.path import abspath, dirname, join

    # root = "/network/lustre/iss02/aramis/projects/clinicadl/data"
    root = "extract"

    # Remove potential residual of previous UT
    clean_folder(join(root, "out"), recreate=False)

    # Copy necessary data from in to out
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps_image"))
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps_slice"))
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps_roi"))
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps_patch"))
    shutil.copytree(join(root, "in", "caps_T1V"), join(root, "out", "caps_custom"))

    # Prepare test for different parameters

    modalities = ["t1-linear", "pet-linear"]  # , "custom"]

    uncropped_image = [True, False]
    acquisition_label = ["18FAV45", "11CPIB"]

    image_params = {"mode": "image"}
    patch_params = {"mode": "patch", "patch_size": 50, "stride_size": 50}
    slice_params = {
        "mode": "slice",
        "slice_mode": "rgb",
        "slice_direction": 0,
        "discarded_slices": [0, 0],
    }
    roi_params = {
        "mode": "roi",
        "roi_list": ["rightHippocampusBox", "leftHippocampusBox"],
        "uncropped_roi": False,
        "roi_custom_template": "",
        "roi_custom_mask_pattern": "",
    }

    data: List[Dict[str, Any]] = [image_params, slice_params, patch_params, roi_params]

    for parameters in data:

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
                    tsv_file = join(root, "in", f"pet_{acq}.tsv")
                    mode = parameters["mode"]
                    extract_generic(root, mode, tsv_file, parameters)

            elif modality == "custom":
                parameters["use_uncropped_image"] = True
                parameters[
                    "custom_suffix"
                ] = "graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
                parameters["roi_custom_template"] = "Ixi549Space"
                parameters[
                    "extract_json"
                ] = f"{modality}_mode-{parameters['mode']}.json"
                tsv_file = join(root, "in", "subjects.tsv")
                mode = parameters["mode"]
                extract_generic(root, mode, tsv_file, parameters)

            elif modality == "t1-linear":
                for flag in uncropped_image:
                    parameters["use_uncropped_image"] = flag
                    parameters[
                        "extract_json"
                    ] = f"{modality}_crop-{not flag}_mode-{parameters['mode']}.json"

                    tsv_file = join(root, "in", "subjects.tsv")
                    mode = parameters["mode"]
                    extract_generic(root, mode, tsv_file, parameters)
            else:
                raise NotImplementedError(
                    f"Test for modality {modality} was not implemented."
                )
        out_folder = join(root, f"out/caps_{mode}/subjects")
        ref_folder = join(root, f"ref/caps_{mode}/subjects")

        diff_path = join(root, f"diff_{mode}{modality}.txt")
        if os.path.exists(diff_path):
            os.remove(diff_path)
        os.system(f"sort ")
        os.system(f"diff -qr {out_folder} {ref_folder} > {diff_path}")
        filesize = os.path.getsize(diff_path)
        assert filesize == 0
        os.remove(diff_path)

    clean_folder(join(root, "out"), recreate=True)


def extract_generic(root, mode, tsv_file, parameters):

    from os.path import join

    from clinicadl.extract.extract import DeepLearningPrepareData

    DeepLearningPrepareData(
        caps_directory=join(root, "out", f"caps_{mode}"),
        tsv_file=tsv_file,
        n_proc=2,
        parameters=parameters,
    )
