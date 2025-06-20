# from pathlib import Path

# import torchio.transforms as transforms

# from clinicadl.data.dataloader import DataLoaderConfig
# from clinicadl.data.datasets.caps_dataset import CapsDataset
# from clinicadl.data.datatypes.modalities.pet import ReconstructionMethod, Tracer
# from clinicadl.data.datatypes.preprocessing import FlairLinear, PETLinear, T1Linear
# from clinicadl.data.datatypes.preprocessing.pet import SUVRReferenceRegion
# from clinicadl.transforms import Transforms
# from clinicadl.transforms.config.factory import (
#     PadConfig,
#     RandomBlurConfig,
#     RandomFlipConfig,
#     RescaleIntensityConfig,
# )
# from clinicadl.transforms.extraction import Slice

# caps_directory = Path(
#     "/Users/camille.brianceau/aramis/CLINICADL/caps"
# )  # output of clinica pipelines
# sub_ses_t1 = caps_directory / "subjects_t1.tsv"  # 64 subjects

# preprocessing_t1 = T1Linear()

# preprocessing_pet_45 = PETLinear(
#     tracer="18FAV45", suvr_reference_region=SUVRReferenceRegion.CEREBELLUM_PONS2
# )

# preprocessing_pet_fdg = PETLinear(
#     tracer=Tracer.FDG,
#     suvr_reference_region=SUVRReferenceRegion.PONS2,
#     use_uncropped_image=True,
#     reconstruction=ReconstructionMethod.STATIC_ATTENUATION_CORRECTION,
# )

# preprocessing_flair = FlairLinear()


# transforms_image = Transforms(
#     image_transforms=[RescaleIntensityConfig()],
#     sample_transforms=[RandomBlurConfig()],
#     augmentations=[RandomFlipConfig()],
#     extraction=Slice(slices=[24, 25, 26, 27, 56, 57, 58, 78, 96, 97]),
# )

# dataset_t1_image = CapsDataset(
#     caps_directory=caps_directory,
#     data=sub_ses_t1,
#     preprocessing=preprocessing_t1,
#     transforms=transforms_image,
#     label="diagnosis",
# )
# dataset_t1_image.to_tensors(json_name="t1_with_transforms_rescale_blur.json", n_proc=2)


import pytest
import torch
import torchio as tio
from pydantic import BaseModel, ConfigDict, ValidationError

from clinicadl.networks.config import get_network_config
from clinicadl.networks.config.vit import (
    ViTB16Config,
    ViTB32Config,
    ViTConfig,
    ViTL16Config,
    ViTL32Config,
)
from clinicadl.transforms.config.spatial_augmentations import (
    RandomAffineConfig,
    RandomAnisotropyConfig,
    RandomElasticDeformationConfig,
    RandomFlipConfig,
)

args = {
    "num_outputs": 1,
}

name = "ViT-L/32"

c = get_network_config(name=name, **args)
print(c)
