from copy import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint, Mask


def test_DataPoint():
    caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
    affine = np.diag([1.3, 1.2, 1.1, 1])
    image = tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3), affine=affine)
    mask = tio.LabelMap(tensor=torch.ones(1, 3, 3, 4), affine=np.diag(np.ones(4)))
    label = tio.LabelMap(tensor=torch.ones(1, 3, 3, 3), affine=affine)
    image_path = (
        caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
    )
    label_path = (
        caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_brain.nii.gz"
    )
    mask_path = caps_dir / "masks" / "leftHippocampus.nii.gz"

    # first basic test
    data_point = DataPoint(
        image_path,
        label,
        participant="sub-000",
        session="ses-M000",
        mask_1=mask,
    )
    data_point.add_mask(mask_path, "mask_2")
    data_point.add_mask(mask, "mask_3")
    assert isinstance(data_point.image, tio.ScalarImage)
    assert (
        data_point.image.tensor == torch.from_numpy(nib.load(image_path).get_fdata())
    ).all()
    assert isinstance(data_point.label, tio.LabelMap)
    assert (data_point.label.tensor == label.tensor).all()
    assert isinstance(data_point.mask_1, tio.LabelMap)
    assert (data_point.mask_1.tensor == mask.tensor).all()
    assert isinstance(data_point.mask_2, tio.LabelMap)
    assert (
        data_point.mask_2.tensor == torch.from_numpy(nib.load(mask_path).get_fdata())
    ).all()
    assert isinstance(data_point.mask_3, tio.LabelMap)
    assert (data_point.mask_3.tensor == mask.tensor).all()
    assert data_point.participant == "sub-000"
    assert data_point.session == "ses-M000"

    # affine, voxel spacing and shapes
    with pytest.raises(RuntimeError):
        data_point.affine
    with pytest.raises(RuntimeError):
        data_point.spacing
    with pytest.raises(RuntimeError):
        data_point.spatial_shape

    # test copy
    c = copy(data_point)
    assert isinstance(c.image, tio.ScalarImage)
    assert isinstance(c.label, tio.LabelMap)
    assert c.participant == "sub-000"
    assert c.session == "ses-M000"
    assert isinstance(c.mask_1, tio.LabelMap)
    assert isinstance(c.mask_2, tio.LabelMap)
    assert isinstance(c.mask_2, tio.LabelMap)
    assert isinstance(c.mask_3, tio.LabelMap)

    # other tests
    data_point = DataPoint(
        image=image,
        label=None,
        participant="sub-000",
        session="ses-M000",
        mask=mask_path,
    )
    assert (data_point.image.tensor == image.tensor).all()
    assert data_point.label is None
    assert (
        data_point.mask.tensor == torch.from_numpy(nib.load(mask_path).get_fdata())
    ).all()
    data_point = DataPoint(
        image=image,
        label=1,
        participant="sub-000",
        session="ses-M000",
    )
    assert data_point.label == 1

    data_point = DataPoint(
        image,
        label=label_path,
        participant="sub-000",
        session="ses-M00",
    )
    assert (
        data_point.label.tensor == torch.from_numpy(nib.load(label_path).get_fdata())
    ).all()
    assert data_point.spacing == (1.3, 1.2, 1.1)
    assert (data_point.affine == np.diag([1.3, 1.2, 1.1, 1])).all()
    assert data_point.spatial_shape == (3, 3, 3)


def test_Mask():
    caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
    subject_dir = caps_dir / "subjects" / "sub-000" / "ses-M000" / "t1_linear"

    #########################################################
    mask = Mask(str(caps_dir / "masks" / "leftHippocampus.nii.gz"))
    assert mask.path == caps_dir / "masks" / "leftHippocampus.nii.gz"
    assert mask.name == "leftHippocampus.nii.gz"

    mask = Mask(caps_dir / "masks" / "leftHippocampus.nii.gz")
    assert mask.is_common_mask
    assert mask._mask_img is None
    associated_mask = mask.get_associated_mask()
    assert (
        associated_mask.tensor
        == torch.from_numpy(
            nib.load(caps_dir / "masks" / "leftHippocampus.nii.gz").get_fdata()
        )
    ).all()
    assert np.isclose(associated_mask.affine, np.diag([1.3, 1.2, 1.1, 1])).all()
    assert mask._mask_img is not None

    # tensor mask
    mask = Mask(caps_dir / "masks" / "tensors" / "leftHippocampus.pt")
    associated_mask = mask.get_associated_mask()
    assert (
        associated_mask.tensor
        == torch.load(
            caps_dir / "masks" / "tensors" / "leftHippocampus.pt", weights_only=True
        )["mask"]
    ).all()

    #####################################
    mask = Mask("brain")
    assert not mask.is_common_mask

    associated_mask = mask.get_associated_mask(
        subject_dir / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
    )
    assert (
        associated_mask.tensor
        == torch.from_numpy(
            nib.load(
                subject_dir
                / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_brain.nii.gz"
            ).get_fdata()
        )
    ).all()
    assert np.isclose(associated_mask.affine, np.diag([1.3, 1.2, 1.1, 1])).all()
    assert mask._mask_img is None

    ##### errors #####
    with pytest.raises(FileNotFoundError):
        mask = Mask(Path("abc.nii.gz"))

    mask = Mask("brain")
    with pytest.raises(FileNotFoundError):
        mask.get_associated_mask(
            subject_dir.parent
            / "pet_linear"
            / "tensors"
            / "sub-000_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii"
        )
    with pytest.raises(ValueError):
        mask.get_associated_mask()
