# from pathlib import Path

# import pytest
# import torchio.transforms as transforms
# from pydantic import ValidationError

# from clinicadl.dataset.config import (
#     PreprocessingConfig,
#     PreprocessingCustom,
#     PreprocessingFlair,
#     PreprocessingPET,
#     PreprocessingT1,
#     PreprocessingT2,
#     get_extraction,
#     get_preprocessing,
# )
# from clinicadl.dataset.datasets.caps_dataset import CapsDataset
# from clinicadl.dataset.transforms.extraction import (
#     ROI,
#     BaseExtraction,
#     Image,
#     Patch,
#     Slice,
# )
# from clinicadl.dataset.transforms.transforms import Transforms
# from clinicadl.utils.enum import (
#     ExtractionMethod,
#     ImageModality,
#     LinearModality,
#     Preprocessing,
# )

# CAPS_PATH = Path(__file__).parents[1] / "ressources" / "caps_example"
# MASK_PATH = CAPS_PATH / "masks"
# TPL_PATH = MASK_PATH / "tpl-MNI152NLin2009cSym"

# sub_ses_t1 = Path(CAPS_PATH / "subjects_t1.tsv")
# sub_ses_pet_45 = Path(CAPS_PATH / "subjects_pet_18FAV45.tsv")
# sub_ses_flair = Path(CAPS_PATH / "subjects_flair.tsv")
# sub_ses_pet_11 = Path(CAPS_PATH / "subjects_pet_11CPIB.tsv")


# transforms_patch = Transforms(
#     object_augmentation=[transforms.Ghosting(2, 1, 0.1, 0.1)],
#     image_augmentation=[transforms.RandomMotion()],
#     extraction=Patch(patch_size=60),
#     image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],
#     object_transforms=[transforms.RandomMotion()],
# )  # not mandatory

# transforms_slice = Transforms(extraction=Slice())

# transforms_roi = Transforms(
#     object_augmentation=[transforms.Ghosting(2, 1, 0.1, 0.1)],
#     object_transforms=[transforms.RandomMotion()],
#     extraction=ROI(
#         roi_list=["leftHippocampusBox", "rightHippocampusBox"],
#         roi_mask_location=TPL_PATH,
#         roi_crop_input=True,
#     ),
# )

# transforms_image = Transforms(
#     image_augmentation=[transforms.RandomMotion()],
#     extraction=Image(),
#     image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],
# )

# preprocessing_t1 = PreprocessingT1()
# preprocessing_flair = PreprocessingFlair()
# preprocessing_45 = PreprocessingPET(tracer="18FAV45", suvr_reference_region="pons2")
# preprocessing_11 = PreprocessingPET(tracer="11CPIB", suvr_reference_region="pons2")


# @pytest.mark.parametrize(
#     "name,config",
#     [
#         ("custom", PreprocessingCustom),
#         ("flair-linear", PreprocessingFlair),
#         ("pet-linear", PreprocessingPET),
#         ("t1-linear", PreprocessingT1),
#         (Preprocessing.CUSTOM, PreprocessingCustom),
#         (Preprocessing.FLAIR_LINEAR, PreprocessingFlair),
#         (Preprocessing.PET_LINEAR, PreprocessingPET),
#         (Preprocessing.T1_LINEAR, PreprocessingT1),
#     ],
# )
# def test_get_preprocessing_config(name, config):
#     assert get_preprocessing(name) == config


# @pytest.mark.parametrize(
#     "name,config",
#     [
#         ("image", Image),
#         ("slice", Slice),
#         ("patch", Patch),
#         ("roi", ROI),
#         (ExtractionMethod.IMAGE, Image),
#         (ExtractionMethod.SLICE, Slice),
#         (ExtractionMethod.PATCH, Patch),
#         (ExtractionMethod.ROI, ROI),
#     ],
# )
# def test_get_extraction_config(name, config):
#     assert get_extraction(name) == config


# @pytest.mark.parametrize(
#     "trc,suvr,patch_size,stride_size,sub_ses",
#     [
#         ("18FAV45", "pons2", 50, 40, sub_ses_pet_45),
#         ("11CPIB", "pons2", 100, 100, sub_ses_pet_11),
#     ],
# )
# def test_pet_patch(trc, suvr, patch_size, stride_size, sub_ses):
#     preprocessing = PreprocessingPET(tracer=trc, suvr_reference_region=suvr)
#     transforms_patch = Transforms(
#         object_augmentation=[transforms.Ghosting(2, 1, 0.1, 0.1)],
#         image_augmentation=[transforms.RandomMotion()],
#         extraction=Patch(patch_size=patch_size, stride_size=stride_size),
#         image_transforms=[transforms.Blur((0.5, 0.6, 0.3))],
#         object_transforms=[transforms.RandomMotion()],
#     )
#     dataset = CapsDataset(
#         caps_directory=CAPS_PATH,
#         data=sub_ses,
#         preprocessing=preprocessing,
#         transforms=transforms_patch,
#     )
#     dataset.prepare_data(n_proc=2)

#     assert dataset._get_meta_data(1)[3] == dataset.__getitem__(1).elem_idx
#     assert dataset._get_full_image()[1] == dataset.__getitem__(0).image_path


# def test_roi():
#     with pytest.raises(ValidationError):
#         ROI()

#     with pytest.raises(ValidationError):
#         ROI(roi_list=["left", "right"])

#     with pytest.raises(NotImplementedError):
#         ROI(roi_list=[], roi_mask_location=Path(""))

#     with pytest.raises(ValidationError):
#         ROI(roi_list=["left", "right"], roi_mask_location=Path(""))

#     with pytest.raises(FileNotFoundError):
#         ROI(roi_list=["left", "right"], roi_mask_location=TPL_PATH)

#     roi = ROI(
#         roi_list=["leftHippocampusBox", "rightHippocampusBox"],
#         roi_mask_location=TPL_PATH,
#         roi_crop_input=True,
#     )
#     roi_bis = ROI(
#         roi_list=["leftHippocampusBox", "rightHippocampusBox"],
#         roi_mask_location=MASK_PATH,
#         roi_crop_input=True,
#     )
#     assert roi == roi_bis


# @pytest.mark.parametrize(
#     "transforms,preprocessing,data_tsv",
#     [
#         (transforms_image, preprocessing_t1, sub_ses_t1),
#         (transforms_roi, preprocessing_45, sub_ses_pet_45),
#         (transforms_slice, preprocessing_45, sub_ses_pet_45),
#         (transforms_patch, preprocessing_flair, sub_ses_flair),
#         (transforms_image, preprocessing_45, sub_ses_pet_45),
#         (transforms_roi, preprocessing_45, sub_ses_pet_45),
#         (transforms_slice, preprocessing_t1, sub_ses_t1),
#         (transforms_patch, preprocessing_flair, sub_ses_flair),
#     ],
# )
# def test(transforms, preprocessing, data_tsv):
#     dataset = CapsDataset(
#         caps_directory=CAPS_PATH,
#         data=data_tsv,
#         preprocessing=preprocessing,
#         transforms=transforms,
#     )
#     dataset.prepare_data(n_proc=2)

#     assert dataset._get_meta_data(2)[3] == dataset.__getitem__(2).elem_idx
#     assert dataset._get_full_image()[1] == dataset.__getitem__(0).image_path
