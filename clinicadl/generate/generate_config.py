from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Annotated, Optional, Union

from pydantic import BaseModel, field_validator

from clinicadl.utils.exceptions import ClinicaDLTSVError

logger = getLogger("clinicadl.predict_config")


class Preprocessing(str, Enum):
    """Possible preprocessing method in clinicaDL."""

    T1_LINEAR = "t1-linear"
    T1_EXTENSIVE = "t1-extensive"
    PET_LINEAR = "pet-linear"


class SUVRReferenceRegions(str, Enum):
    """Possible SUVR reference region for pet images in clinicaDL."""

    PONS = "pons"
    CEREBELLUMPONS = "cerebellumPons"
    PONS2 = "pons2"
    CEREBELLUMPONS2 = "cerebellumPons2"


class Tracer(str, Enum):
    """Possible tracer for pet images in clinicaDL."""

    FFDG = "18FFDG"
    FAV45 = "18FAV45"


class Pathology(str, Enum):
    """Possible pathology for hypometabolic generation of pet images in clinicaDL."""

    AD = "ad"
    BVFTD = "bvftd"
    LVPPA = "lvppa"
    NFVPPA = "nfvppa"
    PCA = "pca"
    SVPPA = "svppa"


class GenerateConfig(BaseModel):
    generated_caps_directory: Path
    n_subjects: int = 300
    n_proc: int = 1

    class ConfigDict:
        validate_assignment = True


class SharedGenerateConfigOne(GenerateConfig):
    caps_directory: Path
    participants_list: Optional[Path] = None
    preprocessing_cls: Preprocessing = Preprocessing.T1_LINEAR
    use_uncropped_image: bool = False

    @field_validator("participants_list", mode="before")
    def check_tsv_file(cls, v):
        if not isinstance(v, Path):
            Path(v)
        return v

    @property
    def preprocessing(self) -> Preprocessing:
        return self.preprocessing_cls.value

    @preprocessing.setter
    def preprocessing(self, value: Union[str, Preprocessing]):
        self.preprocessing_cls = Preprocessing(value)


class SharedGenerateConfigTwo(SharedGenerateConfigOne):
    suvr_reference_region_cls: SUVRReferenceRegions = SUVRReferenceRegions.PONS
    tracer_cls: Tracer = Tracer.FFDG

    @property
    def suvr_reference_region(self) -> SUVRReferenceRegions:
        return self.suvr_reference_region_cls.value

    @suvr_reference_region.setter
    def suvr_reference_region(self, value: Union[str, SUVRReferenceRegions]):
        self.suvr_reference_region_cls = SUVRReferenceRegions(value)

    @property
    def tracer(self) -> Tracer:
        return self.tracer_cls.value

    @tracer.setter
    def tracer(self, value: Union[str, Tracer]):
        self.tracer_cls = Tracer(value)


class GenerateArtifactsConfig(SharedGenerateConfigTwo):
    contrast: bool = False
    gamma: Annotated[list[float], 2] = [-0.2, -0.05]
    motion: bool = False
    num_transforms: int = 2
    noise: bool = False
    noise_std: Annotated[list[float], 2] = [5, 15]
    rotation: Annotated[list[float], 2] = [2, 4]  # float o int ???
    translation: Annotated[list[float], 2] = [2, 4]

    @field_validator("gamma", "noise_std", "rotation", "translation", mode="before")
    def list_to_tuples(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("gamma", mode="before")
    def gamma_validator(cls, v):
        assert len(v) == 2
        if v[0] < -1 or v[0] > v[1] or v[1] > 1:
            raise ValueError(
                f"gamma augmentation must range between -1 and 1, please set other values than {v}."
            )
        return v


class GenerateHypometabolicConfig(SharedGenerateConfigOne):
    anomaly_degree: float = 30.0
    pathology_cls: Pathology = Pathology.AD
    sigma: int = 5

    @property
    def pathology(self) -> Pathology:
        return self.pathology_cls.value

    @pathology.setter
    def pathology(self, value: Union[str, Pathology]):
        self.pathology_cls = Pathology(value)


class GenerateRandomConfig(SharedGenerateConfigTwo):
    mean: float = 0.0
    n_subjects: int = 300
    sigma: float = 0.5


class GenerateTrivialConfig(SharedGenerateConfigTwo):
    atrophy_percent: float = 60.0
    mask_path: Optional[Path] = None

    @field_validator("mask_path", mode="before")
    def check_mask_file(cls, v):
        if v:
            if not isinstance(v, Path):
                Path(v)
            if not v.is_file():
                raise ClinicaDLTSVError(
                    "The participants_list you gave is not a file. Please give an existing file."
                )
            if v.stat().st_size == 0:
                raise ClinicaDLTSVError(
                    "The participants_list you gave is empty. Please give a non-empty file."
                )

        return v


class GenerateSheppLoganConfig(GenerateConfig):
    ad_subtypes_distribution: Annotated[list[float], 3] = [0.05, 0.85, 0.10]
    cn_subtypes_distribution: Annotated[list[float], 3] = [1.0, 0.0, 0.0]
    extract_json: str = ""
    image_size: int = 128
    smoothing: bool = False

    # @field_validator(
    #     "ad_subtypes_distribution", "cn_subtypes_distribution", mode="before"
    # )
    # # def list_to_tuples(cls, v):
    # #     if isinstance(v, list):
    # #         return tuple(v)
    # #     return v
