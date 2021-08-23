# coding: utf8

import warnings
from os import pardir

from clinica.test.nonregression.testing_tools import clean_folder, compare_folders

warnings.filterwarnings("ignore")


def test_extract():
    import shutil
    from os.path import abspath, dirname, join

    root = dirname(abspath(join(abspath(__file__), pardir, pardir)))
    root = join(root, "data", "DeepLearningPrepareData")

    # Remove potential residual of previous UT
    clean_folder(join(root, "out", "caps"), recreate=False)

    # Copy necessary data from in to out
    shutil.copytree(join(root, "in", "caps"), join(root, "out", "caps"))

    # Prepare test for different parameters

    modalities = ["t1-linear", "pet-linear", "custom"]

    uncropped_image = [True, False]

    image_params = {"extract_method": "image"}
    slice_params = {"extract_method": "patch", "patch_size": 50, "stride_size": 50}
    patch_params = {
        "extract_method": "slice",
        "slice_mode": "rgb",
        "slice_direction": 0,
    }
    roi_params = {
        "extract_method": "roi",
        "roi_list": ["rightHippocampusBox", "leftHippocampusBox"],
        "use_uncropped_image": True,
    }

    data = [image_params, slice_params, patch_params, roi_params]

    for parameters in data:

        for modality in modalities:

            parameters["modality"] = modality

            if modality == "pet-linear":
                parameters["acq_label"] = "av45"
                parameters["suvr_reference_region"] = "pons2"
                parameters["use_uncropped_image"] = False
                extract_generic(root, parameters)

            elif modality == "custom":
                parameters["use_uncropped_image"] = True
                parameters["custom_template"] = "Ixi549Space"
                parameters[
                    "custom_suffix"
                ] = "graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
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

    compare_folders(out_folder, ref_folder, shared_folder_name="caps")

    clean_folder(join(root, "out", "caps"), recreate=False)


def extract_generic(root, parameters):

    from os.path import join

    from clinicadl.extract import DeepLearningPrepareData

    DeepLearningPrepareData(
        caps_directory=join(root, "out", "caps"),
        tsv_file=join(root, "in", "subjects.tsv"),
        parameters=parameters,
    )
