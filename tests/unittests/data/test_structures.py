from pathlib import Path

import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint, Mask


def test_DataPoint():
    image_tensor = torch.randn(1, 3, 4, 5)
    mask_1 = torch.ones(1, 3, 4, 5)
    mask_2 = torch.zeros(1, 3, 4, 5)
    label = torch.ones(1, 3, 4, 5)

    data_point = DataPoint(image_tensor, label, mask_1=mask_1, mask_2=mask_2)
    assert isinstance(data_point.image, tio.ScalarImage)
    assert (data_point.image.tensor == image_tensor).all()
    assert isinstance(data_point.label, tio.LabelMap)
    assert (data_point.label.tensor == label).all()
    assert isinstance(data_point.mask_1, tio.LabelMap)
    assert (data_point.mask_1.tensor == mask_1).all()
    assert isinstance(data_point.mask_2, tio.LabelMap)
    assert (data_point.mask_2.tensor == mask_2).all()

    data_point = DataPoint(image_tensor, label=None, mask_1=tio.LabelMap(tensor=mask_1))
    assert data_point.label is None
    data_point = DataPoint(image_tensor, label=1)
    assert data_point.label == 1
    data_point = DataPoint(image_tensor, label=tio.LabelMap(tensor=label))
    assert isinstance(data_point.label, tio.LabelMap)

    with pytest.raises(AssertionError):
        DataPoint(image_tensor, label=torch.ones(1, 2, 4, 5))
    with pytest.raises(AssertionError):
        DataPoint(image_tensor, label=None, mask_1=torch.ones(1, 2, 4, 5))


def test_Mask():
    caps_dir = Path(__file__).parents[1] / "ressources" / "caps_example"
    mask = Mask(caps_dir / "masks/leftHippocampus.nii.gz")
    assert (
        mask.get_associated_mask("sub-001_ses-M000_T1w.nii.gz")
        == caps_dir / "masks/leftHippocampus.nii.gz"
    )

    mask = Mask("label-hippocampus_mask")
    subject_dir = caps_dir / "subjects" / "sub-000" / "ses-M000" / "t1_linear"
    assert (
        mask.get_associated_mask(
            subject_dir
            / "sub-000_ses-M000_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"
        )
        == subject_dir
        / "sub-000_ses-M000_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_label-hippocampus_mask.nii.gz"
    )
