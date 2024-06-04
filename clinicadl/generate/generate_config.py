import tarfile
from logging import getLogger
from pathlib import Path
from time import time
from typing import Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    computed_field,
    field_validator,
)

from clinicadl.caps_dataset.data_config import DataConfig as DataBaseConfig
from clinicadl.config.config import ModalityConfig
from clinicadl.preprocessing.config import PreprocessingConfig
from clinicadl.utils.clinica_utils import (
    RemoteFileStructure,
    clinicadl_file_reader,
    fetch_file,
)
from clinicadl.utils.enum import (
    Pathology,
    Preprocessing,
    SUVRReferenceRegions,
    Tracer,
)
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLTSVError,
    DownloadError,
)

logger = getLogger("clinicadl.predict_config")


class GenerateConfig(BaseModel):
    generated_caps_directory: Path

    # pydantic config
    model_config = ConfigDict(validate_assignment=True)

    # TODO: The number of subjects cannot be higher than the number of subjects in the baseline caps dataset


class GenerateArtifactsConfig(BaseModel):
    contrast: bool = False
    gamma: Tuple[float, float] = (-0.2, -0.05)
    motion: bool = False
    num_transforms: PositiveInt = 2
    noise: bool = False
    noise_std: Tuple[NonNegativeFloat, NonNegativeFloat] = (5.0, 15.0)
    rotation: Tuple[NonNegativeFloat, NonNegativeFloat] = (2.0, 4.0)  # float o int ???
    translation: Tuple[NonNegativeFloat, NonNegativeFloat] = (2.0, 4.0)

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

    @computed_field
    @property
    def artifacts_list(self) -> list[str]:
        artifacts_list = []
        if self.motion:
            artifacts_list.append("motion")
        if self.contrast:
            artifacts_list.append("contrast")
        if self.noise:
            artifacts_list.append("noise")
        return artifacts_list


class GenerateHypometabolicConfig(BaseModel):
    anomaly_degree: NonNegativeFloat = 30.0
    pathology: Pathology = Pathology.AD
    sigma: NonNegativeFloat = 5


class GenerateRandomConfig(BaseModel):
    mean: NonNegativeFloat = 0.0
    sigma: NonNegativeFloat = 0.5


class GenerateTrivialConfig(BaseModel):
    atrophy_percent: PositiveFloat = 60.0
    mask_path: Optional[Path] = None


class GenerateSheppLoganConfig(BaseModel):
    ad_subtypes_distribution: Tuple[
        NonNegativeFloat, NonNegativeFloat, NonNegativeFloat
    ] = (0.05, 0.85, 0.10)
    cn_subtypes_distribution: Tuple[
        NonNegativeFloat, NonNegativeFloat, NonNegativeFloat
    ] = (1.0, 0.0, 0.0)
    extract_json: Optional[str] = None
    image_size: PositiveInt = 128
    smoothing: bool = False

    @field_validator("extract_json", mode="before")
    def compute_extract_json(cls, v: str):
        if v is None:
            return f"extract_{int(time())}.json"
        elif not v.endswith(".json"):
            return f"{v}.json"
        else:
            return v

    @field_validator(
        "ad_subtypes_distribution", "cn_subtypes_distribution", mode="before"
    )
    def probabilities_validator(
        cls, v: Tuple[NonNegativeFloat, NonNegativeFloat, NonNegativeFloat]
    ):
        for i in v:
            if i > 1 or i < 0:
                raise ClinicaDLArgumentError(
                    f"Probabilities must be between 0 and 1 for {v}"
                )

        if isinstance(v, list):
            return tuple(v)
        return v
