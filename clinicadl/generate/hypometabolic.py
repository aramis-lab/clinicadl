import os

import nibabel as nib
import numpy as np
from clinica.utils.inputs import clinica_file_reader
from clinica.utils.nipype import container_from_filename
from clinica.utils.participant import get_subject_session_list
from nilearn.image import resample_to_img
from scipy import ndimage

# import argparse


# Take command line inputs
# parser = argparse.ArgumentParser()
# parser.add_argument('pathology', type=str, help='Pathology shortname')
# parser.add_argument('percentage', type=int, help='Percentage of hypometabolism')
# args = parser.parse_args()

percentage = 30
pathologies = ["ad"]


def hypo_synthesis(image, mask):
    return image * mask


# Get files list
tsv_path = "/network/lustre/iss02/aramis/users/ravi.hassanaly/projects/thesis/adni_tsv/pet_uniform/deep_learning_exp/train/validation_splits-8/split-0/CN_baseline.tsv"
input_caps_dir = "/network/lustre/iss02/aramis/datasets/adni/caps/caps_pet_uniform"
print("Load files")
sessions, subjects = get_subject_session_list(
    input_caps_dir, tsv_path, False, False, None
)
file_type = {
    "pattern": os.path.join(
        "pet_linear",
        f"*_trc-18FFDG_rec-uniform_pet_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz",
    ),
    "description": "",
    "needed_pipeline": "pet-linear",
}
input_files = clinica_file_reader(subjects, sessions, input_caps_dir, file_type)[0]
print(input_files)

for pathology in pathologies:

    # Load mask
    mask_path = os.path.join(input_caps_dir, "masks", f"mask_hypo_{pathology}.nii")
    mask_nii = nib.load(mask_path)

    # Create the output CAPS
    output_caps_dir = f"/network/lustre/iss02/aramis/users/ravi.hassanaly/data/hypometabolic_caps/caps_validation_{pathology}_{percentage}"
    os.makedirs(os.path.join(output_caps_dir, "subjects"), exist_ok=True)
    # Set sigma value depending on percentage
    # if percentage >= 70:
    #    sigma = 9
    # elif percentage >= 50:
    #    sigma = 6
    # else:
    #    sigma = 5
    sigma = 6
    # For loop on all the files
    for file in input_files:
        print("Load file: ", file)
        image_nii = nib.load(file)
        image = image_nii.get_fdata()

        print("Making mask")
        mask_nii = resample_to_img(mask_nii, image_nii, interpolation="nearest")
        mask = mask_nii.get_fdata()

        inverse_mask = 1 - mask
        inverse_mask[inverse_mask == 0] = 1 - percentage / 100
        gaussian_mask = ndimage.gaussian_filter(inverse_mask, sigma=sigma)

        # Create outputdir
        container = container_from_filename(file)
        out_image_nii_dir = os.path.join(output_caps_dir, container, "pet_linear")
        out_image_nii_filename = os.path.basename(file)
        os.makedirs(out_image_nii_dir, exist_ok=True)

        print("Processing image")
        out_image = hypo_synthesis(image, gaussian_mask)

        # Save the image
        out_image_nii = nib.Nifti1Image(out_image, affine=image_nii.affine)
        out_image_nii.to_filename(
            os.path.join(out_image_nii_dir, out_image_nii_filename)
        )
        print(f"Saved at {out_image_nii_filename}")
