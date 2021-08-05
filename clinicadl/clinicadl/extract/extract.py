from logging import getLogger

def DeepLearningPrepareData(caps_directory, tsv_file, parameters):
    import os
    from os import path
    from torch import save as save_tensor
    from clinica.utils.inputs import check_caps_folder
    from clinica.utils.participant import get_subject_session_list
    from clinica.utils.exceptions import (
        ClinicaBIDSError,
        ClinicaCAPSError,
        ClinicaException,
    )
    from clinica.utils.input_files import (
        T1W_EXTENSIVE,
        T1W_LINEAR,
        T1W_LINEAR_CROPPED,
        pet_linear_nii,
    )
    from clinica.utils.inputs import clinica_file_reader
    from clinica.utils.nipype import container_from_filename
    from clinicadl.utils.preprocessing import write_preprocessing
    from .extract_utils import (
        check_mask_list,
        extract_patches,
        extract_roi,
        extract_slices,
        extract_images,
    )

    logger = getLogger("clinicadl")

    # Get subject and session list
    check_caps_folder(caps_directory)
    input_dir = caps_directory
    logger.debug(f"CAPS directory : {input_dir}.")
    is_bids_dir = False
    sessions, subjects = get_subject_session_list(
        input_dir, tsv_file, is_bids_dir, False, None
    )
    logger.info(f"{parameters['mode']}s will be extracted in Pytorch tensor from {len(sessions)} images.")
    logger.debug(f"List of subjects: \n{subjects}.")
    logger.debug(f"List of sessions: \n{sessions}.")

    # Select the correct filetype corresponding to modality
    # and select the right folder output name corresponding to modality
    logger.debug(f"Selected images are preprocessed with {parameters['preprocessing']} pipeline`.")
    if parameters["preprocessing"] == "t1-linear":
        mod_subfolder = "t1_linear"
        if parameters["use_uncropped_image"]:
            FILE_TYPE = T1W_LINEAR
        else:
            FILE_TYPE = T1W_LINEAR_CROPPED
    if parameters["preprocessing"] == "t1-extensive":
        mod_subfolder = "t1_extensive"
        FILE_TYPE = T1W_EXTENSIVE
        parameters["uncropped_image"] = None
    if parameters["preprocessing"] == "pet-linear":
        mod_subfolder = "pet_linear"
        FILE_TYPE = pet_linear_nii(
            parameters["acq_label"],
            parameters["suvr_reference_region"],
            parameters["use_uncropped_image"],
        )
    if parameters["preprocessing"] == "custom":
        mod_subfolder = "custom"
        FILE_TYPE = {
            "pattern": f"*{parameters['custom_suffix']}",
            "description": "Custom suffix",
        }
        parameters["use_uncropped_image"] = None
    parameters["file_type"] = FILE_TYPE

    # Input file:
    input_files = clinica_file_reader(subjects, sessions, caps_directory, FILE_TYPE)

    # Loop on the images
    for file in input_files:
        logger.debug(f"  Processing of {file}.")
        container = container_from_filename(file)
        # Extract the wanted tensor
        if parameters["mode"] == "image":
            subfolder = "image_based"
            output_mode = extract_images(file)
            logger.debug(f"    Image extracted.")
        elif parameters["mode"] == "slice":
            subfolder = "slice_based"
            output_mode = extract_slices(
                file,
                slice_direction=parameters["slice_direction"],
                slice_mode=parameters["slice_mode"],
            )
            logger.debug(f"    {len(output_mode)} slices extracted.")
        elif parameters["mode"] == "patch":
            subfolder = "patch_based"
            output_mode = extract_patches(
                file,
                patch_size=parameters["patch_size"],
                stride_size=parameters["stride_size"],
            )
            logger.debug(f"    {len(output_mode)} patches extracted.")
        elif parameters["mode"] == "roi":
            subfolder = "roi_based"
            if parameters["preprocessing"] == "custom":
                parameters["roi_template"] = parameters["roi_custom_template"]
                if parameters["roi_custom_template"] is None:
                    raise ValueError(
                        "A custom template must be defined when the modality is set to custom."
                    )
            else:
                from .extract_utils import TEMPLATE_DICT
                parameters["roi_template"] = TEMPLATE_DICT[parameters["preprocessing"]]
            parameters["masks_location"] = path.join(
                caps_directory, "masks", f"tpl-{parameters['roi_template']}"
            )
            if len(parameters["roi_list"])==0:
                raise ValueError("A list of regions must be given.")
            else:
                check_mask_list(
                    parameters["masks_location"],
                    parameters["roi_list"],
                    parameters["roi_custom_mask_pattern"],
                    None
                    if parameters["use_uncropped_image"] is None
                    else not parameters["use_uncropped_image"],
                )
            output_mode = extract_roi(
                file,
                masks_location=parameters["masks_location"],
                mask_pattern=parameters["roi_custom_mask_pattern"],
                cropped_input=None
                if parameters["use_uncropped_image"] is None
                else not parameters["use_uncropped_image"],
                roi_list=parameters["roi_list"],
                uncrop_output=parameters["uncropped_roi"],
            )
            logger.debug(f"    ROI extracted.")
        # Write the extracted tensor on a .pt file
        for tensor in output_mode:
            output_file_dir = path.join(
                caps_directory,
                container,
                "deeplearning_prepare_data",
                subfolder,
                mod_subfolder,
            )
            if not path.exists(output_file_dir):
                os.makedirs(output_file_dir)
            output_file = path.join(output_file_dir, tensor[0])
            save_tensor(tensor[1], output_file)
            logger.debug(f"    Output tensor saved at {output_file}")

    # Save parameters dictionnary
    preprocessing_json_path = write_preprocessing(parameters, caps_directory)
    logger.info(f"Preprocessing JSON saved at {preprocessing_json_path}.")
