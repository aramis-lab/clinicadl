from copy import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint, Mask


def test_DataPoint():
    caps_dir = Path(__file__).parents[1] / "ressources" / "caps_example"
    image_tensor = torch.randn(1, 3, 4, 5)
    mask = torch.ones(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)
    image_path = (
        caps_dir
        / "subjects"
        / "sub-002"
        / "ses-M018"
        / "pet_linear"
        / "sub-002_ses-M018_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_label-hippocampus_mask.nii.gz"
    )
    label_path = (
        caps_dir
        / "subjects"
        / "sub-002"
        / "ses-M018"
        / "pet_linear"
        / "sub-002_ses-M018_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_label-hippocampus_mask.nii.gz"
    )
    mask_path = caps_dir / "masks" / "leftHippocampus.nii.gz"

    data_point = DataPoint(
        image_path,
        label,
        participant="sub-002",
        session="ses-M018",
        mask_1=mask,
        mask_2=mask_path,
    )
    assert isinstance(data_point.image, tio.ScalarImage)
    assert (
        data_point.image.tensor == torch.from_numpy(nib.load(image_path).get_fdata())
    ).all()
    assert isinstance(data_point.label, tio.LabelMap)
    assert (data_point.label.tensor == label).all()
    assert isinstance(data_point.mask_1, tio.LabelMap)
    assert (data_point.mask_1.tensor == mask).all()
    assert isinstance(data_point.mask_2, tio.LabelMap)
    assert (
        data_point.mask_2.tensor == torch.from_numpy(nib.load(mask_path).get_fdata())
    ).all()
    assert data_point.participant == "sub-002"
    assert data_point.session == "ses-M018"

    c = copy(data_point)
    assert isinstance(c.image, tio.ScalarImage)
    assert isinstance(c.label, tio.LabelMap)
    assert c.participant == "sub-002"
    assert c.session == "ses-M018"
    assert isinstance(c.mask_1, tio.LabelMap)
    assert isinstance(c.mask_2, tio.LabelMap)

    data_point = DataPoint(
        tio.ScalarImage(tensor=image_tensor),
        label=None,
        mask_1=tio.LabelMap(tensor=mask),
        participant="sub-002",
        session="ses-M018",
    )
    assert (data_point.image.tensor == image_tensor).all()
    assert data_point.label is None
    assert (data_point.mask_1.tensor == mask).all()
    data_point = DataPoint(
        image_tensor,
        label=1,
        participant="sub-002",
        session="ses-M018",
    )
    assert (data_point.image.tensor == image_tensor).all()
    assert data_point.label == 1
    data_point = DataPoint(
        image_tensor,
        label=tio.LabelMap(tensor=label),
        participant="sub-002",
        session="ses-M018",
    )
    assert (data_point.label.tensor == label).all()
    data_point = DataPoint(
        image_tensor,
        label=label_path,
        participant="sub-002",
        session="ses-M018",
    )
    assert (
        data_point.label.tensor == torch.from_numpy(nib.load(label_path).get_fdata())
    ).all()


def test_Mask():
    caps_dir = Path(__file__).parents[1] / "ressources" / "caps_example"
    subject_1_dir = caps_dir / "subjects" / "sub-002" / "ses-M006" / "pet_linear"
    subject_2_dir = caps_dir / "subjects" / "sub-000" / "ses-M006" / "pet_linear"

    #########################################################
    mask = Mask(str(caps_dir / "masks" / "leftHippocampus.nii.gz"))
    assert mask.mask == caps_dir / "masks" / "leftHippocampus.nii.gz"

    mask = Mask(caps_dir / "masks" / "leftHippocampus.nii.gz")
    assert mask.is_common_mask
    assert mask._mask_img is None
    assert mask._mask_pt is None

    associated_mask = mask.get_associated_mask(
        subject_1_dir
        / "sub-002_ses-M006_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.nii.gz"
    )
    assert (
        associated_mask.tensor
        == torch.from_numpy(
            nib.load(caps_dir / "masks" / "leftHippocampus.nii.gz").get_fdata()
        )
    ).all()
    assert np.isclose(associated_mask.spacing, (0.9, 0.9, 0.9)).all()
    assert mask._mask_img is not None
    assert mask._mask_pt is None

    associated_mask = mask.get_associated_mask(
        subject_1_dir
        / "tensors"
        / "sub-002_ses-M006_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.pt"
    )
    assert (
        associated_mask.tensor
        == torch.load(
            caps_dir / "masks" / "tensors" / "leftHippocampus.pt", weights_only=True
        )
    ).all()
    assert mask._mask_img is not None
    assert mask._mask_pt is not None

    #####################################
    mask = Mask("label-hippocampus_mask")
    assert not mask.is_common_mask

    associated_mask = mask.get_associated_mask(
        subject_2_dir
        / "sub-000_ses-M006_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.nii.gz"
    )
    assert (
        associated_mask.tensor
        == torch.from_numpy(
            nib.load(
                subject_2_dir
                / "sub-000_ses-M006_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_label-hippocampus_mask.nii.gz"
            ).get_fdata()
        )
    ).all()
    assert np.isclose(associated_mask.spacing, (1.2, 0.9, 0.9)).all()

    associated_mask = mask.get_associated_mask(
        subject_2_dir
        / "tensors"
        / "sub-000_ses-M006_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.pt"
    )
    assert (
        associated_mask.tensor
        == torch.load(
            subject_2_dir
            / "tensors"
            / "sub-000_ses-M006_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_label-hippocampus_mask.pt",
            weights_only=True,
        )
    ).all()

    assert mask._mask_img is None
    assert mask._mask_pt is None

    ##### errors #####
    with pytest.raises(FileNotFoundError):
        mask = Mask(Path("abc.nii.gz"))

    mask = Mask("label-hippocampus_mask")
    with pytest.raises(FileNotFoundError):
        mask.get_associated_mask(
            subject_1_dir
            / "tensors"
            / "sub-002_ses-M006_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.pt"
        )
