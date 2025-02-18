from pathlib import Path

import torchio.transforms as transforms

from clinicadl.data import tensor_conversion
from clinicadl.data.datasets import CapsDataset, ConcatDataset
from clinicadl.data.datatype.modalities.pet import Tracer
from clinicadl.data.datatype.preprocessing import (
    FlairLinear,
    PETLinear,
    Preprocessing,
    T1Linear,
)
from clinicadl.data.datatype.preprocessing.pet import SUVRReferenceRegion
from clinicadl.experiment_manager import ExperimentManager
from clinicadl.losses.config import CrossEntropyLossConfig
from clinicadl.model.clinicadl_model import ClinicaDLModel
from clinicadl.networks.factory import (
    ConvEncoderOptions,
    create_network_config,
    get_network_from_config,
)
from clinicadl.splitter import KFold, make_kfold, make_split
from clinicadl.transforms import Transforms
from clinicadl.transforms.extraction import Image, Patch, Slice

sub_ses_t1 = Path("/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_t1.tsv")
sub_ses_pet_45 = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_pet_18FAV45.tsv"
)
sub_ses_flair = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_flair.tsv"
)
sub_ses_pet_11 = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps/subjects_pet_11CPIB.tsv"
)

caps_directory = Path(
    "/Users/camille.brianceau/aramis/CLINICADL/caps"
)  # output of clinica pipelines

preprocessing_pet_45 = PETLinear(
    tracer=Tracer.AV45, suvr_reference_region=SUVRReferenceRegion.PONS2
)
preprocessing_pet_11 = PETLinear(
    tracer=Tracer.PIB, suvr_reference_region=SUVRReferenceRegion.PONS2
)

preprocessing_t1 = T1Linear()
preprocessing_flair = FlairLinear()


transforms_patch = Transforms(
    augmentations=[transforms.Ghosting(2, 1, 0.1, 0.1), transforms.RandomMotion()],
    extraction=Patch(patch_size=60),
    image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],
    sample_transforms=[transforms.RandomMotion()],
)  # not mandatory

transforms_slice = Transforms(extraction=Slice())

transforms_image = Transforms(
    augmentations=[transforms.RandomMotion()],
    extraction=Image(),
    image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],
)


print("Pet 45 and Patch ")
dataset_pet_45_patch = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_pet_45,
    preprocessing=preprocessing_pet_45,
    transforms=transforms_patch,
)
dataset_pet_45_patch.to_tensors(json_name="test_pet_45.json", n_proc=2)

print(dataset_pet_45_patch)
print(dataset_pet_45_patch.__len__())
print(dataset_pet_45_patch._get_meta_data(3))
print(dataset_pet_45_patch._get_meta_data(80))
# print(dataset_pet_45_patch._get_full_image())

# dataset_pet_45_patch.caps_reader._write_caps_json(
#     transforms_patch, preprocessing_pet_45, sub_ses_pet_45, name="tfsdklsqfh"
# )


print("Pet 11 and Image ")

dataset_pet_11_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_pet_11,
    preprocessing=preprocessing_pet_11,
    transforms=transforms_image,
)
dataset_pet_11_image.to_tensors(
    json_name="test_pet_11.json"
)  # to extract the tensor of the PET file this time

print(dataset_pet_11_image)
print(dataset_pet_11_image.__len__())
print(dataset_pet_11_image._get_meta_data(0))
print(dataset_pet_11_image._get_meta_data(1))
# print(dataset_pet_11_image._get_full_image())
# print(dataset_pet_11_image.__getitem__(1).elem_idx)
# print(dataset_pet_11_image.elem_per_image)


print("T1 and image ")

dataset_t1_image = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_t1,
    preprocessing=preprocessing_t1,
    transforms=transforms_image,
)
dataset_t1_image.to_tensors(
    json_name="test_t1_image.json"
)  # to extract the tensor of the PET file this time

print(dataset_t1_image)
print(dataset_t1_image.__len__())
print(dataset_t1_image._get_meta_data(3))
print(dataset_t1_image._get_meta_data(5))
# print(dataset_t1_image._get_full_image())
# print(dataset_t1_image.__getitem__(5).elem_idx)
# print(dataset_t1_image.elem_per_image)


print("Flair and slice ")

dataset_flair_slice = CapsDataset(
    caps_directory=caps_directory,
    data=sub_ses_flair,
    preprocessing=preprocessing_flair,
    transforms=transforms_slice,
)

print(dataset_flair_slice)
print(dataset_flair_slice.__len__())
print(dataset_flair_slice._get_meta_data(3))
print(dataset_flair_slice._get_meta_data(80))
# print(dataset_flair_slice._get_full_image())
# print(dataset_flair_slice.__getitem__(80).elem_idx)
# print(dataset_flair_slice.elem_per_image)


lity_multi_extract = ConcatDataset(
    [
        dataset_t1_image,
        dataset_pet_11_image,
    ]
)  # 3 train.tsv en entrée qu'il faut concat et pareil pour les transforms à faire attention
