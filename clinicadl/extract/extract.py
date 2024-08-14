import os
from logging import getLogger
from pathlib import Path


def DeepLearningPrepareData(
    caps_directory: Path, tsv_file: Path, n_proc, parameters, mood_abdom: bool = False
):
    from joblib import Parallel, delayed
    from torch import save as save_tensor

    from clinicadl.utils.clinica_utils import (
        check_caps_folder,
        clinicadl_file_reader,
        container_from_filename,
        get_subject_session_list,
    )
    from clinicadl.utils.exceptions import ClinicaDLArgumentError
    from clinicadl.utils.preprocessing import write_preprocessing

    from .extract_utils import check_mask_list, compute_folder_and_file_type

    logger = getLogger("clinicadl")

    # Get subject and session list
    check_caps_folder(caps_directory)
    logger.debug(f"CAPS directory : {caps_directory}.")
    is_bids_dir = False
    subjects, sessions = get_subject_session_list(
        caps_directory, tsv_file, is_bids_dir, False, None
    )
    if parameters["prepare_dl"]:
        logger.info(
            f"{parameters['mode']}s will be extracted in Pytorch tensor from {len(sessions)} images."
        )
    else:
        logger.info(
            f"Images will be extracted in Pytorch tensor from {len(sessions)} images."
        )
        logger.info(
            f"Information for {parameters['mode']} will be saved in output JSON file and will be used "
            f"during training for on-the-fly extraction."
        )
    logger.debug(f"List of subjects: \n{subjects}.")
    logger.debug(f"List of sessions: \n{sessions}.")

    # Select the correct filetype corresponding to modality
    # and select the right folder output name corresponding to modality
    logger.debug(
        f"Selected images are preprocessed with {parameters['preprocessing']} pipeline`."
    )
    mod_subfolder, file_type = compute_folder_and_file_type(parameters)
    parameters["file_type"] = file_type

    # Input file:
    input_files = clinicadl_file_reader(subjects, sessions, caps_directory, file_type)[
        0
    ]
    logger.debug(f"Selected image file name list: {input_files}.")

    def write_output_imgs(output_mode, container, subfolder, mood_abdom: bool = False):
        # Write the extracted tensor on a .pt file
        for filename, tensor in output_mode:
            output_file_dir = Path(
                caps_directory,
                container,
                "deeplearning_prepare_data",
                subfolder,
                mod_subfolder,
            )
            if not Path(output_file_dir).is_dir():
                os.makedirs(output_file_dir)

            output_file = Path(output_file_dir, filename)

            if mood_abdom:
                import torch
                from skimage.transform import resize

                output_img = resize(tensor.numpy(), output_shape=(256, 256, 256))
                output_img = torch.tensor(output_img)

            save_tensor(tensor, output_file)

            logger.debug(f"    Output tensor saved at {output_file}")

    if parameters["mode"] == "image" or not parameters["prepare_dl"]:

        def prepare_image(file):
            from .extract_utils import extract_images

            logger.debug(f"  Processing of {file}.")
            container = container_from_filename(file)
            subfolder = "image_based"
            output_mode = extract_images(file)
            logger.debug(f"    Image extracted.")
            write_output_imgs(output_mode, container, subfolder, mood_abdom=mood_abdom)

        Parallel(n_jobs=n_proc)(delayed(prepare_image)(file) for file in input_files)

    elif parameters["prepare_dl"] and parameters["mode"] == "slice":

        def prepare_slice(file):
            from .extract_utils import extract_slices

            logger.debug(f"  Processing of {file}.")
            container = container_from_filename(file)
            subfolder = "slice_based"
            output_mode = extract_slices(
                file,
                slice_direction=parameters["slice_direction"],
                slice_mode=parameters["slice_mode"],
                discarded_slices=parameters["discarded_slices"],
            )
            logger.debug(f"    {len(output_mode)} slices extracted.")
            write_output_imgs(output_mode, container, subfolder)

        Parallel(n_jobs=n_proc)(delayed(prepare_slice)(file) for file in input_files)

    elif parameters["prepare_dl"] and parameters["mode"] == "patch":

        def prepare_patch(file):
            from .extract_utils import extract_patches

            logger.debug(f"  Processing of {file}.")
            container = container_from_filename(file)
            subfolder = "patch_based"
            output_mode = extract_patches(
                file,
                patch_size=parameters["patch_size"],
                stride_size=parameters["stride_size"],
            )
            logger.debug(f"    {len(output_mode)} patches extracted.")
            write_output_imgs(output_mode, container, subfolder)

        Parallel(n_jobs=n_proc)(delayed(prepare_patch)(file) for file in input_files)

    elif parameters["prepare_dl"] and parameters["mode"] == "roi":

        def prepare_roi(file):
            from .extract_utils import extract_roi

            logger.debug(f"  Processing of {file}.")
            container = container_from_filename(file)
            subfolder = "roi_based"
            if parameters["preprocessing"] == "custom":
                if not parameters["roi_custom_template"]:
                    raise ClinicaDLArgumentError(
                        "A custom template must be defined when the modality is set to custom."
                    )
                parameters["roi_template"] = parameters["roi_custom_template"]
                parameters["roi_mask_pattern"] = parameters["roi_custom_mask_pattern"]
            else:
                from .extract_utils import PATTERN_DICT, TEMPLATE_DICT

                parameters["roi_template"] = TEMPLATE_DICT[parameters["preprocessing"]]
                parameters["roi_mask_pattern"] = PATTERN_DICT[
                    parameters["preprocessing"]
                ]

            parameters["masks_location"] = Path(
                caps_directory, "masks", f"tpl-{parameters['roi_template']}"
            )
            if len(parameters["roi_list"]) == 0:
                raise ClinicaDLArgumentError(
                    "A list of regions of interest must be given."
                )
            else:
                check_mask_list(
                    parameters["masks_location"],
                    parameters["roi_list"],
                    parameters["roi_mask_pattern"],
                    (
                        None
                        if parameters["use_uncropped_image"] is None
                        else not parameters["use_uncropped_image"]
                    ),
                )
            output_mode = extract_roi(
                file,
                masks_location=parameters["masks_location"],
                mask_pattern=parameters["roi_mask_pattern"],
                cropped_input=(
                    None
                    if parameters["use_uncropped_image"] is None
                    else not parameters["use_uncropped_image"]
                ),
                roi_names=parameters["roi_list"],
                uncrop_output=parameters["uncropped_roi"],
            )
            logger.debug(f"    ROI extracted.")
            write_output_imgs(output_mode, container, subfolder)

        Parallel(n_jobs=n_proc)(delayed(prepare_roi)(file) for file in input_files)

    else:
        raise NotImplementedError(
            f"Extraction is not implemented for mode {parameters['mode']}."
        )

    # Save parameters dictionary
    preprocessing_json_path = write_preprocessing(parameters, caps_directory)
    logger.info(f"Preprocessing JSON saved at {preprocessing_json_path}.")
