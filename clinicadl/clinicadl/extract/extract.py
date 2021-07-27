def DeepLearningPrepareData(caps_directory, tsv_file, parameters, preprocessing_path):
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
    from clinicadl.utils.preprossing import write_preprocessing
    from .extract_utils import (
        check_mask_list,
        extract_patches,
        extract_roi,
        extract_slices,
        save_as_pt,
    )

    # Get subject and session list
    check_caps_folder(caps_directory)
    input_dir = caps_directory
    is_bids_dir = False
    subjects, sessions = get_subject_session_list(
        input_dir, tsv_file, is_bids_dir, False, None
    )

    # Select the correct filetype corresponding to modality
    # and select the right folder output name corresponding to modality
    if parameters["modality"] == "t1-linear":
        mod_subfolder = "t1_linear"
        if parameters["use_uncropped_image"]:
            FILE_TYPE = T1W_LINEAR
        else:
            FILE_TYPE = T1W_LINEAR_CROPPED
    if parameters["modality"] == "t1-extensive":
        mod_subfolder = "t1_extensive"
        FILE_TYPE = T1W_EXTENSIVE
        parameters["uncropped_image"] = None
    if parameters["modality"] == "pet-linear":
        mod_subfolder = "pet_linear"
        FILE_TYPE = pet_linear_nii(
            parameters["acq_label"],
            parameters["suvr_reference_region"],
            parameters["use_uncropped_image"],
        )
    if parameters["modality"] == "custom":
        mod_subfolder = "custom"
        FILE_TYPE = {
            "pattern": f"*{parameters['custom_suffix']}",
            "description": "Custom suffix",
        }
        parameters["use_uncropped_image"] = None
    parameters["file_type"] = FILE_TYPE

    # Input file:
    try:
        input_files = clinica_file_reader(subjects, sessions, caps_directory, FILE_TYPE)
    except ClinicaException as e:
        err = (
            "Clinica faced error(s) while trying to read files in your CAPS directory.\n"
            + str(e)
        )
        raise ClinicaBIDSError(err)

    # Loop on the images
    for file in input_files:
        container = container_from_filename(file)
        # Extract the wanted tensor
        if parameters["extract_method"] == "image":
            subfolder = "image_based"
            output_mode = save_as_pt(file)
        elif parameters["extract_method"] == "slice":
            subfolder = "slice_based"
            output_file_rgb, output_file_original = extract_slices(
                file,
                slice_direction=parameters["slice_direction"],
                slice_mode=parameters["slice_mode"],
            )
        elif parameters["extract_method"] == "patch":
            subfolder = "patch_based"
            output_mode = extract_patches(
                file,
                patch_size=parameters["patch_size"],
                stride_size=parameters["stride_size"],
            )
        elif parameters["extract_method"] == "roi":
            subfolder = "roi_based"
            if parameters["modality"] == "custom":
                parameters["roi_template"] = parameters["roi_custom_template"]
                if parameters["roi_custom_template"] is None:
                    raise ValueError(
                        "A custom template must be defined when the modality is set to custom."
                    )
            else:
                from .prepare_data_utils import TEMPLATE_DICT

                parameters["roi_template"] = TEMPLATE_DICT[parameters["modality"]]
            parameters["masks_location"] = path.join(
                caps_directory, "masks", f"tpl-{parameters['roi_template']}"
            )
            if parameters["roi_list"] is None:
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
                uncrop_output=parameters["roi_uncrop_output"],
            )
        # Write the extracted tensor on a .pt file
        for tensor in output_mode:
            output_file_path = path.join(
                caps_directory,
                container,
                "deep_learning_prepare_data",
                subfolder,
                mod_subfolder,
                tensor[0],
            )
            save_tensor(tensor[1], output_file_path)

    # Save parameters dictionnary
    write_preprocessing(parameters, preprocessing_path)
