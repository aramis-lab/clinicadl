# coding: utf8

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


def compare_folders(out, ref, shared_folder_name):
    from difflib import unified_diff
    from filecmp import cmp
    from os import remove
    from os.path import join

    out_txt = join(out, "out_folder.txt")
    ref_txt = join(ref, "ref_folder.txt")

    # list_files(join(out, shared_folder_name), filename=out_txt)

    list_files(out, filename=out_txt)
    list_files(ref, filename=ref_txt)

    print(ref_txt)
    # list_files(join(ref, shared_folder_name), filename=ref_txt)

    # Compare them
    if not cmp(out_txt, ref_txt):
        with open(out_txt, "r") as fin:
            out_message = fin.read()
        with open(ref_txt, "r") as fin:
            ref_message = fin.read()
        with open(out_txt, "r") as out:
            with open(ref_txt, "r") as ref:
                diff = unified_diff(
                    out.readlines(),
                    ref.readlines(),
                    fromfile="output",
                    tofile="reference",
                )
        diff_text = ""
        for line in diff:
            diff_text = diff_text + line + "\n"
        # remove(out_txt)
        # remove(ref_txt)
        raise AssertionError(
            "Comparison of out and ref directories shows mismatch :\n "
            "OUT :\n"
            + out_message
            + "\n REF :\n"
            + ref_message
            + "\nDiff :\n"
            + diff_text
        )

    # Clean folders
    # remove(out_txt)
    # remove(ref_txt)


def list_files(startpath, filename=None):
    """
    Args:
        startpath: starting point for the tree listing. Does not list hidden
        files (to avoid problems with .DS_store for example
        filename: if None, display to stdout, otherwise write in the file
    Returns:
        void
    """
    from os import remove, sep, walk
    from os.path import abspath, basename, exists, expanduser, expandvars

    if exists(filename):
        remove(filename)

    expanded_path = abspath(expanduser(expandvars(startpath)))
    for root, dirs, files in walk(expanded_path):
        level = root.replace(startpath, "").count(sep)
        indent = " " * 4 * (level)
        rootstring = "{}{}/".format(indent, basename(root))
        # Do not deal with hidden files
        if not basename(root).startswith("."):
            if filename is not None:
                # 'a' stands for 'append' rather than 'w' for 'write'. We must
                # manually jump line with \n otherwise everything is
                # concatenated
                with open(filename, "a") as fin:
                    fin.write(rootstring + "\n")
            else:
                print(rootstring)
            subindent = " " * 4 * (level + 1)
            for f in files:
                filestring = "{}{}".format(subindent, f)
                if not basename(f).startswith("."):
                    if filename is not None:
                        with open(filename, "a") as fin:
                            fin.write(filestring + "\n")
                    else:
                        print(filestring)


def test_extract():
    import shutil
    from os.path import abspath, dirname, join

    # root = "/network/lustre/iss02/aramis/projects/clinicadl/data"
    root = "/mnt/data/data_CI"
    root = join(root, "extract")

    # Remove potential residual of previous UT
    clean_folder(join(root, "out"), recreate=False)

    # Copy necessary data from in to out
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps_image"))
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps_slice"))
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps_roi"))
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps_patch"))

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
        "uncropped_roi": True,
        "roi_custom_template": "",
        "roi_custom_mask_pattern": "",
    }

    data: List[Dict[str, Any]] = [image_params, slice_params, patch_params, roi_params]

    for parameters in data:

        parameters["prepare_dl"] = True
        parameters["save_features"] = True

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

            # Check output vs ref
    import os

    # image
    out_folder = join(root, "out/caps_image")
    ref_folder = join(root, "ref/caps_image")

    diff_path = join(root, "diff.txt")
    if os.path.exists(diff_path):
        os.remove(diff_path)
    os.system(f"diff -Naur {out_folder} {ref_folder} > {diff_path}")
    filesize = os.path.getsize(diff_path)
    assert filesize == 0

    # slice
    out_folder = join(root, "out/caps_slice")
    ref_folder = join(root, "ref/caps_slice")

    diff_path = join(root, "diff.txt")
    if os.path.exists(diff_path):
        os.remove(diff_path)
    os.system(f"diff -Naur {out_folder} {ref_folder} > {diff_path}")
    filesize = os.path.getsize(diff_path)
    assert filesize == 0

    # roi
    out_folder = join(root, "out/caps_roi")
    ref_folder = join(root, "ref/caps_roi")

    diff_path = join(root, "diff.txt")
    if os.path.exists(diff_path):
        os.remove(diff_path)
    os.system(f"diff -Naur {out_folder} {ref_folder} > {diff_path}")
    filesize = os.path.getsize(diff_path)
    assert filesize == 0

    # patch
    out_folder = join(root, "out/caps_patch")
    ref_folder = join(root, "ref/caps_patch")

    diff_path = join(root, "diff.txt")
    if os.path.exists(diff_path):
        os.remove(diff_path)
    os.system(f"diff -Naur {out_folder} {ref_folder} > {diff_path}")
    filesize = os.path.getsize(diff_path)
    assert filesize == 0

    # compare_folders(out_folder, ref_folder, shared_folder_name="caps/subjects")
    os.remove(diff_path)
    clean_folder(join(root, "out"), recreate=False)


def extract_generic(root, mode, tsv_file, parameters):

    from os.path import join

    from clinicadl.extract.extract import DeepLearningPrepareData

    DeepLearningPrepareData(
        caps_directory=join(root, "out", f"caps_{mode}"),
        tsv_file=tsv_file,
        n_proc=2,
        parameters=parameters,
    )
