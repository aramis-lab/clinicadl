from pathlib import Path

from clinicadl.dataset.caps_dataset import CapsDataset
from clinicadl.dataset.config.extraction import (
    ExtractionConfig,
    ExtractionImageConfig,
    ExtractionPatchConfig,
    ExtractionROIConfig,
    ExtractionSliceConfig,
)
from clinicadl.dataset.config.preprocessing import PreprocessingConfig
from clinicadl.experiment_manager.experiment_manager import ExperimentManager
from clinicadl.transforms.config import TransformsConfig


class CapsReader:
    def __init__(self, caps_directory: Path, manager: ExperimentManager):
        """TO COMPLETE"""
        pass

    def get_dataset(
        self,
        extraction: ExtractionConfig,
        preprocessing: PreprocessingConfig,
        sub_ses_tsv: Path,
        transforms: TransformsConfig,
    ) -> CapsDataset:
        return CapsDataset(extraction, preprocessing, sub_ses_tsv, transforms)

    def get_preprocessing(self, preprocessing: str) -> PreprocessingConfig:
        """TO COMPLETE"""

        return PreprocessingConfig()

    def extract_slice(
        self, preprocessing: PreprocessingConfig, arg_slice
    ) -> ExtractionSliceConfig:
        """TO COMPLETE"""

        return ExtractionSliceConfig()

    def extract_patch(
        self, preprocessing: PreprocessingConfig, arg_patch
    ) -> ExtractionPatchConfig:
        """TO COMPLETE"""

        return ExtractionPatchConfig()

    def extract_roi(
        self, preprocessing: PreprocessingConfig, arg_roi
    ) -> ExtractionROIConfig:
        """TO COMPLETE"""

        return ExtractionROIConfig()

    def extract_image(
        self, preprocessing: PreprocessingConfig, arg_image
    ) -> ExtractionImageConfig:
        """TO COMPLETE"""

        return ExtractionImageConfig()
