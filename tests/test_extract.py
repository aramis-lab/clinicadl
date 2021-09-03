# coding: utf8

import warnings
from os import pardir

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

    list_files(join(out, shared_folder_name), filename=out_txt)
    list_files(join(ref, shared_folder_name), filename=ref_txt)

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
        remove(out_txt)
        remove(ref_txt)
        raise ValueError(
            "Comparison of out and ref directories shows mismatch :\n "
            "OUT :\n"
            + out_message
            + "\n REF :\n"
            + ref_message
            + "\nDiff :\n"
            + diff_text
        )

    # Clean folders
    remove(out_txt)
    remove(ref_txt)


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

    root = dirname(abspath(join(abspath(__file__))))
    root = join(root, "data", "dataset", "DeepLearningPrepareData")

    # Remove potential residual of previous UT
    clean_folder(join(root, "out", "caps"), recreate=False)

    # Copy necessary data from in to out
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps"))

    # Prepare test for different parameters

    modalities = ["t1-linear", "pet-linear", "custom"]

    uncropped_image = [True, False]

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

    data = [image_params, slice_params, patch_params, roi_params]

    for parameters in data:

        parameters["prepare_dl"] = True

        for modality in modalities:

            parameters["preprocessing"] = modality

            if modality == "pet-linear":
                parameters["acq_label"] = "av45"
                parameters["suvr_reference_region"] = "pons2"
                parameters["use_uncropped_image"] = False
                extract_generic(root, parameters)

            elif modality == "custom":
                parameters["use_uncropped_image"] = True
                parameters[
                    "custom_suffix"
                ] = "graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
                parameters["roi_custom_template"] = "Ixi549Space"
                extract_generic(root, parameters)

            elif modality == "t1-linear":
                for flag in uncropped_image:
                    parameters["use_uncropped_image"] = flag
                    extract_generic(root, parameters)
            else:
                raise NotImplementedError(
                    f"Test for modality {modality} was not implemented."
                )

    # Check output vs ref
    out_folder = join(root, "out")
    ref_folder = join(root, "ref")

    compare_folders(out_folder, ref_folder, shared_folder_name="caps/subjects")

    clean_folder(join(root, "out", "caps"), recreate=False)


def extract_generic(root, parameters):

    from os.path import join

    from clinicadl.extract.extract import DeepLearningPrepareData

    DeepLearningPrepareData(
        caps_directory=join(root, "out", "caps"),
        tsv_file=join(root, "in", "subjects.tsv"),
        parameters=parameters,
    )
