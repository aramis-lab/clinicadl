from copy import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint


def test_DataPoint():
    caps_dir = Path(__file__).parents[2] / "resources" / "caps_example"
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
    data_point.add_image(image_path, "image_2")
    data_point.add_image(image, "image_3")

    data_point.add_mask(str(mask_path), "mask_2")
    data_point.add_mask(mask, "mask_3")

    assert isinstance(data_point.image, tio.ScalarImage)
    assert (
        data_point.image.tensor == torch.from_numpy(nib.load(image_path).get_fdata())
    ).all()

    assert isinstance(data_point.label, tio.LabelMap)
    assert (data_point.label.tensor == label.tensor).all()

    assert isinstance(data_point["image_2"], tio.ScalarImage)
    assert isinstance(data_point["image_3"], tio.ScalarImage)
    assert (
        data_point["image_2"].tensor
        == torch.from_numpy(nib.load(image_path).get_fdata())
    ).all()
    assert (data_point["image_3"].tensor == image.tensor).all()

    assert isinstance(data_point["mask_2"], tio.LabelMap)
    assert isinstance(data_point["mask_3"], tio.LabelMap)
    assert (
        data_point["mask_2"].tensor == torch.from_numpy(nib.load(mask_path).get_fdata())
    ).all()
    assert (data_point["mask_3"].tensor == mask.tensor).all()

    assert data_point.participant == "sub-000"
    assert data_point.session == "ses-M000"

    # affine, voxel spacing and shapes
    with pytest.raises(RuntimeError):
        data_point.affine
    with pytest.raises(RuntimeError):
        data_point.spacing
    with pytest.raises(RuntimeError):
        data_point.spatial_shape
    with pytest.raises(RuntimeError):
        data_point.shape

    # get images
    assert len(data_point.get_images()) == 3
    assert len(data_point.get_images(intensity_only=False)) == 7
    assert len(data_point.get_images(intensity_only=False, include="image")) == 1
    assert len(data_point.get_images(intensity_only=False, exclude="image")) == 6

    assert len(data_point.get_images_dict()) == 3
    assert set(data_point.get_images_dict().keys()) == {"image", "image_2", "image_3"}
    assert len(data_point.get_images_dict(intensity_only=False)) == 7
    assert len(data_point.get_images_dict(intensity_only=False, include="image")) == 1
    assert len(data_point.get_images_dict(intensity_only=False, exclude="image")) == 6

    # test copy
    c = copy(data_point)
    assert isinstance(c.image, tio.ScalarImage)
    assert isinstance(c.label, tio.LabelMap)
    assert c.participant == "sub-000"
    assert c.session == "ses-M000"
    assert isinstance(c["mask_3"], tio.LabelMap)

    # other tests
    data_point = DataPoint(
        image=image,
        label=None,
        participant="sub-000",
        session="ses-M000",
    )
    assert (data_point.image.tensor == image.tensor).all()
    assert data_point.label is None

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
    assert data_point.shape == (1, 3, 3, 3)

    # transforms history
    transform = tio.Clamp(out_min=0, out_max=1)
    transformed_datapoint = transform(data_point)
    assert len(transformed_datapoint.get_applied_transforms()) == 1
    assert isinstance(transformed_datapoint.get_applied_transforms()[0], tio.Clamp)
