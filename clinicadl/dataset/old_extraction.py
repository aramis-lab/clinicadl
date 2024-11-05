import json
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch import save as save_tensor

from clinicadl.dataset.caps_dataset import (
    CapsDataset,
    CapsDatasetImage,
    CapsDatasetPatch,
    CapsDatasetRoi,
    CapsDatasetSlice,
)
from clinicadl.dataset.config.extraction import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.dataset.config.preprocessing import (
    CustomPreprocessingConfig,
    DTIPreprocessingConfig,
    FlairPreprocessingConfig,
    PETPreprocessingConfig,
    PreprocessingConfig,
    T1PreprocessingConfig,
    T2PreprocessingConfig,
)
from clinicadl.dataset.config.utils import get_preprocessing
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.transforms.transforms import Transforms
from clinicadl.utils.enum import (
    DTIMeasure,
    DTISpace,
    Preprocessing,
    SliceDirection,
    SliceMode,
    SubFolder,
    Suffix,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.exceptions import ClinicaDLArgumentError
from clinicadl.utils.iotools.clinica_utils import (
    check_caps_folder,
    clinicadl_file_reader,
    container_from_filename,
    create_subs_sess_list,
    determine_caps_or_bids,
    get_subject_session_list,
)
from clinicadl.utils.iotools.utils import path_encoder


def extract_slice(
    self,
    preprocessing: PreprocessingConfig,
    data_tsv: Optional[Path] = None,
    n_proc: int = 2,
    extract_json: Optional[str] = None,
    slice_direction: Optional[SliceDirection] = None,
    slice_mode: Optional[SliceMode] = None,
    discarded_slices: Optional[Union[int, tuple]] = None,
) -> ExtractionSliceConfig:
    """TO COMPLETE"""

    input_files = self.prepare_extraction(preprocessing, data_tsv=data_tsv)
    extraction = ExtractionSliceConfig(
        extract_json=extract_json,
        slice_direction=slice_direction,
        slice_mode=slice_mode,
        discarded_slices=discarded_slices,
    )

    def prepare_slice(file):
        logger.debug(f"  Processing of {file}.")
        output_mode = extraction.extract_slices(file)
        logger.debug(f"{len(output_mode)} slices extracted.")

        self.write_output_imgs(
            output_mode=output_mode,
            file=file,
            subfolder=SubFolder.SLICE,
            preprocessing=preprocessing,
        )

    Parallel(n_jobs=n_proc)(delayed(prepare_slice)(file) for file in input_files)

    return extraction


def extract_patch(
    self,
    preprocessing: PreprocessingConfig,
    data_tsv: Optional[Path] = None,
    n_proc: int = 2,
    extract_json: Optional[str] = None,
    patch_size: Optional[int] = None,
    stride_size: Optional[int] = None,
) -> ExtractionPatchConfig:
    """TO COMPLETE"""

    input_files = self.prepare_extraction(preprocessing, data_tsv=data_tsv)
    extraction = ExtractionPatchConfig(
        extract_json=extract_json, patch_size=patch_size, stride_size=stride_size
    )

    def prepare_patch(file):
        logger.debug(f"  Processing of {file}.")
        output_mode = extraction.extract_patches(file)
        logger.debug(f"{len(output_mode)} patches extracted.")
        self.write_output_imgs(
            output_mode=output_mode,
            file=file,
            subfolder=SubFolder.PATCH,
            preprocessing=preprocessing,
        )

    Parallel(n_jobs=n_proc)(delayed(prepare_patch)(file) for file in input_files)

    return extraction


def extract_roi(
    self,
    preprocessing: PreprocessingConfig,
    data_tsv: Optional[Path] = None,
    n_proc: int = 2,
    extract_json: Optional[str] = None,
    roi_list: Optional[list[str]] = None,
    roi_crop_input: Optional[bool] = None,
    roi_crop_output: Optional[bool] = None,
    roi_custom_template: Optional[str] = None,
    roi_custom_pattern: str = None,
    roi_custom_suffix: Optional[str] = None,
    roi_custom_mask_pattern: Optional[str] = None,
    roi_background_value: Optional[int] = None,
) -> ExtractionROIConfig:
    """TO COMPLETE"""

    input_files = self.prepare_extraction(preprocessing, data_tsv=data_tsv)
    extraction = ExtractionROIConfig(
        extract_json=extract_json,
        roi_list=roi_list,
        roi_crop_input=roi_crop_input,
        roi_crop_output=roi_crop_output,
        roi_custom_template=roi_custom_template,
        roi_custom_mask_pattern=roi_custom_pattern,
        roi_custom_suffix=roi_custom_suffix,
        roi_background_value=roi_background_value,
    )
    extraction.check_with_preprocessing(preprocessing=preprocessing.preprocessing)

    def prepare_roi(file):
        logger.debug(f"  Processing of {file}.")
        masks_location = (
            self.input_directory / "masks" / f"tpl-{extraction.roi_template}"
        )
        extraction.check_mask_list(masks_location=masks_location)
        output_mode = extraction.extract_roi(file, masks_location=masks_location)
        logger.debug(f"{len(output_mode)} patches extracted.")
        self.write_output_imgs(
            output_mode=output_mode,
            file=file,
            subfolder=SubFolder.ROI,
            preprocessing=preprocessing,
        )

    Parallel(n_jobs=n_proc)(delayed(prepare_roi)(file) for file in input_files)

    return extraction
