from logging import getLogger
from pathlib import Path
from typing import Optional

from joblib import Parallel, delayed
from torch import save as save_tensor

from clinicadl.caps_dataset.caps_dataset_config import CapsDatasetConfig
from clinicadl.caps_dataset.caps_dataset_utils import compute_folder_and_file_type
from clinicadl.caps_dataset.extraction.config import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.caps_dataset.extraction.utils import write_preprocessing
from clinicadl.utils.clinica_utils import (
    check_caps_folder,
    clinicadl_file_reader,
    container_from_filename,
    determine_caps_or_bids,
    get_subject_session_list,
)
from clinicadl.utils.enum import ExtractionMethod, Pattern, Preprocessing, Template
from clinicadl.utils.exceptions import ClinicaDLArgumentError

from .prepare_data_utils import check_mask_list


def DeepLearningPrepareData(
    config: CapsDatasetConfig, from_bids: Optional[Path] = None
):
    logger = getLogger("clinicadl.prepare_data")
    # Get subject and session list
    if from_bids is not None:
        try:
            input_directory = Path(from_bids)
        except ClinicaDLArgumentError:
            logger.warning("Your BIDS directory doesn't exist.")
        logger.debug(f"BIDS directory: {input_directory}.")
        is_bids_dir = True
    else:
        input_directory = config.data.caps_directory
        check_caps_folder(input_directory)
        logger.debug(f"CAPS directory: {input_directory}.")
        is_bids_dir = False

    subjects, sessions = get_subject_session_list(
        input_directory, config.data.data_tsv, is_bids_dir, False, None
    )

    if config.extraction.save_features:
        logger.info(
            f"{config.extraction.mode.value}s will be extracted in Pytorch tensor from {len(sessions)} images."
        )
    else:
        logger.info(
            f"Images will be extracted in Pytorch tensor from {len(sessions)} images."
        )
        logger.info(
            f"Information for {config.extraction.mode.value} will be saved in output JSON file and will be used "
            f"during training for on-the-fly extraction."
        )
    logger.debug(f"List of subjects: \n{subjects}.")
    logger.debug(f"List of sessions: \n{sessions}.")

    # Select the correct filetype corresponding to modality
    # and select the right folder output name corresponding to modality
    logger.debug(
        f"Selected images are preprocessed with {config.preprocessing} pipeline`."
    )

    mod_subfolder, file_type = compute_folder_and_file_type(config, from_bids)

    # Input file:
    input_files = clinicadl_file_reader(
        subjects, sessions, input_directory, file_type.model_dump()
    )[0]
    logger.debug(f"Selected image file name list: {input_files}.")

    def write_output_imgs(output_mode, container, subfolder):
        # Write the extracted tensor on a .pt file
        for filename, tensor in output_mode:
            output_file_dir = (
                config.data.caps_directory
                / container
                / "deeplearning_prepare_data"
                / subfolder
                / mod_subfolder
            )
            output_file_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_file_dir / filename
            save_tensor(tensor, output_file)
            logger.debug(f"Output tensor saved at {output_file}")

    if (
        config.extraction.mode == ExtractionMethod.IMAGE
        or not config.extraction.save_features
    ):

        def prepare_image(file):
            from .prepare_data_utils import extract_images

            logger.debug(f"Processing of {file}.")
            container = container_from_filename(file)
            subfolder = "image_based"
            output_mode = extract_images(Path(file))
            logger.debug("Image extracted.")
            write_output_imgs(output_mode, container, subfolder)

        Parallel(n_jobs=config.dataloader.n_proc)(
            delayed(prepare_image)(file) for file in input_files
        )

    elif config.extraction.save_features:
        if config.extraction.mode == ExtractionMethod.SLICE:
            assert isinstance(config.extraction, ExtractionSliceConfig)

            def prepare_slice(file):
                from .prepare_data_utils import extract_slices

                assert isinstance(config.extraction, ExtractionSliceConfig)
                logger.debug(f"  Processing of {file}.")
                container = container_from_filename(file)
                subfolder = "slice_based"
                output_mode = extract_slices(
                    Path(file),
                    slice_direction=config.extraction.slice_direction,
                    slice_mode=config.extraction.slice_mode,
                    discarded_slices=config.extraction.discarded_slices,
                )
                logger.debug(f"    {len(output_mode)} slices extracted.")
                write_output_imgs(output_mode, container, subfolder)

            Parallel(n_jobs=config.dataloader.n_proc)(
                delayed(prepare_slice)(file) for file in input_files
            )

        elif config.extraction.mode == ExtractionMethod.PATCH:
            assert isinstance(config.extraction, ExtractionPatchConfig)

            def prepare_patch(file):
                from .prepare_data_utils import extract_patches

                assert isinstance(config.extraction, ExtractionPatchConfig)
                logger.debug(f"  Processing of {file}.")
                container = container_from_filename(file)
                subfolder = "patch_based"
                output_mode = extract_patches(
                    Path(file),
                    patch_size=config.extraction.patch_size,
                    stride_size=config.extraction.stride_size,
                )
                logger.debug(f"    {len(output_mode)} patches extracted.")
                write_output_imgs(output_mode, container, subfolder)

            Parallel(n_jobs=config.dataloader.n_proc)(
                delayed(prepare_patch)(file) for file in input_files
            )

        elif config.extraction.mode == ExtractionMethod.ROI:
            assert isinstance(config.extraction, ExtractionROIConfig)

            def prepare_roi(file):
                from .prepare_data_utils import extract_roi

                assert isinstance(config.extraction, ExtractionROIConfig)
                logger.debug(f"  Processing of {file}.")
                container = container_from_filename(file)
                subfolder = "roi_based"
                if config.preprocessing == Preprocessing.CUSTOM:
                    if not config.extraction.roi_custom_template:
                        raise ClinicaDLArgumentError(
                            "A custom template must be defined when the modality is set to custom."
                        )
                    roi_template = config.extraction.roi_custom_template
                    roi_mask_pattern = config.extraction.roi_custom_mask_pattern
                else:
                    if config.preprocessing.preprocessing == Preprocessing.T1_LINEAR:
                        roi_template = Template.T1_LINEAR
                        roi_mask_pattern = Pattern.T1_LINEAR
                    elif config.preprocessing.preprocessing == Preprocessing.PET_LINEAR:
                        roi_template = Template.PET_LINEAR
                        roi_mask_pattern = Pattern.PET_LINEAR
                    elif (
                        config.preprocessing.preprocessing == Preprocessing.FLAIR_LINEAR
                    ):
                        roi_template = Template.FLAIR_LINEAR
                        roi_mask_pattern = Pattern.FLAIR_LINEAR

                masks_location = input_directory / "masks" / f"tpl-{roi_template}"

                if len(config.extraction.roi_list) == 0:
                    raise ClinicaDLArgumentError(
                        "A list of regions of interest must be given."
                    )
                else:
                    check_mask_list(
                        masks_location,
                        config.extraction.roi_list,
                        roi_mask_pattern,
                        config.preprocessing.use_uncropped_image,
                    )

                output_mode = extract_roi(
                    Path(file),
                    masks_location=masks_location,
                    mask_pattern=roi_mask_pattern,
                    cropped_input=not config.preprocessing.use_uncropped_image,
                    roi_names=config.extraction.roi_list,
                    uncrop_output=config.extraction.roi_uncrop_output,
                )
                logger.debug("ROI extracted.")
                write_output_imgs(output_mode, container, subfolder)

            Parallel(n_jobs=config.dataloader.n_proc)(
                delayed(prepare_roi)(file) for file in input_files
            )

    else:
        raise NotImplementedError(
            f"Extraction is not implemented for mode {config.extraction.mode.value}."
        )

    # Save parameters dictionary
    preprocessing_json_path = write_preprocessing(
        config.extraction.model_dump(), config.data.caps_directory
    )
    logger.info(f"Preprocessing JSON saved at {preprocessing_json_path}.")
