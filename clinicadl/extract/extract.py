from logging import getLogger


def DeepLearningPrepareData(caps_directory, tsv_file, parameters):
    import os
    from os import path

    from clinica.utils.inputs import check_caps_folder, clinica_file_reader
    from clinica.utils.nipype import container_from_filename
    from clinica.utils.participant import get_subject_session_list
    from torch import save as save_tensor

    from clinicadl.utils.preprocessing import write_preprocessing

    from .extract_utils import (
        check_mask_list,
        compute_folder_and_file_type,
        extract_images,
        extract_patches,
        extract_roi,
        extract_slices,
    )

    logger = getLogger("clinicadl")

    # Get subject and session list
    check_caps_folder(caps_directory)
    logger.debug(f"CAPS directory : {caps_directory}.")
    is_bids_dir = False
    sessions, subjects = get_subject_session_list(
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
    input_files = clinica_file_reader(subjects, sessions, caps_directory, file_type)

    # Loop on the images
    for file in input_files:
        logger.debug(f"  Processing of {file}.")
        container = container_from_filename(file)
        # Extract the wanted tensor
        if parameters["mode"] == "image" or not parameters["prepare_dl"]:
            subfolder = "image_based"
            output_mode = extract_images(file)
            logger.debug(f"    Image extracted.")
        elif parameters["prepare_dl"] and parameters["mode"] == "slice":
            subfolder = "slice_based"
            output_mode = extract_slices(
                file,
                slice_direction=parameters["slice_direction"],
                slice_mode=parameters["slice_mode"],
                discarded_slices=parameters["discarded_slices"],
            )
            logger.debug(f"    {len(output_mode)} slices extracted.")
        elif parameters["prepare_dl"] and parameters["mode"] == "patch":
            subfolder = "patch_based"
            output_mode = extract_patches(
                file,
                patch_size=parameters["patch_size"],
                stride_size=parameters["stride_size"],
            )
            logger.debug(f"    {len(output_mode)} patches extracted.")
        elif parameters["prepare_dl"] and parameters["mode"] == "roi":
            subfolder = "roi_based"
            if parameters["preprocessing"] == "custom":
                if not parameters["roi_custom_template"]:
                    raise ValueError(
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

            parameters["masks_location"] = path.join(
                caps_directory, "masks", f"tpl-{parameters['roi_template']}"
            )
            if len(parameters["roi_list"]) == 0:
                raise ValueError("A list of regions must be given.")
            else:
                check_mask_list(
                    parameters["masks_location"],
                    parameters["roi_list"],
                    parameters["roi_mask_pattern"],
                    None
                    if parameters["use_uncropped_image"] is None
                    else not parameters["use_uncropped_image"],
                )
            output_mode = extract_roi(
                file,
                masks_location=parameters["masks_location"],
                mask_pattern=parameters["roi_mask_pattern"],
                cropped_input=None
                if parameters["use_uncropped_image"] is None
                else not parameters["use_uncropped_image"],
                roi_names=parameters["roi_list"],
                uncrop_output=parameters["uncropped_roi"],
            )
            logger.debug(f"    ROI extracted.")
        else:
            raise NotImplementedError(
                f"Extraction is not implemented for mode {parameters['mode']}."
            )
        # Write the extracted tensor on a .pt file
        for filename, tensor in output_mode:
            output_file_dir = path.join(
                caps_directory,
                container,
                "deeplearning_prepare_data",
                subfolder,
                mod_subfolder,
            )
            if not path.exists(output_file_dir):
                os.makedirs(output_file_dir)
            output_file = path.join(output_file_dir, filename)
            save_tensor(tensor, output_file)
            logger.debug(f"    Output tensor saved at {output_file}")

    # Save parameters dictionary
    preprocessing_json_path = write_preprocessing(parameters, caps_directory)
    logger.info(f"Preprocessing JSON saved at {preprocessing_json_path}.")
