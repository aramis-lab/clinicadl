from copy import copy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint


class SubDataPoint(DataPoint):
    pass


def test_DataPoint():
    bids_dir = Path(__file__).parents[2] / "resources" / "bids"
    affine = np.diag([1.3, 1.2, 1.1, 1])
    image = tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3), affine=affine)
    mask = tio.LabelMap(tensor=torch.ones(1, 3, 3, 4), affine=np.diag(np.ones(4)))
    image_path = (
        bids_dir
        / "sub-000"
        / "ses-M000"
        / "anat"
        / "sub-000_ses-M000_res-1d3x1d2x1d1_T1w.nii.gz"
    )
    mask_path = (
        bids_dir
        / "derivatives"
        / "caps"
        / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
    )

    # first basic test
    data_point = DataPoint(
        image_path,
        participant_id="sub-000",
        session_id="ses-M000",
        mask_1=mask,
        other=0,
    )
    data_point.add_image(image_path, "image_2")
    data_point.add_image(image, "image_3")
    data_point.add_image(image.tensor, "image_4")

    data_point.add_mask(str(mask_path), "mask_2")
    data_point.add_mask(mask, "mask_3")
    data_point.add_mask(image.tensor, "mask_4")

    assert isinstance(data_point.image, tio.ScalarImage)
    assert (
        data_point.image.tensor == torch.from_numpy(nib.load(image_path).get_fdata())
    ).all()

    assert isinstance(data_point["image_2"], tio.ScalarImage)
    assert isinstance(data_point["image_3"], tio.ScalarImage)
    assert isinstance(data_point["image_4"], tio.ScalarImage)
    assert (
        data_point["image_2"].tensor
        == torch.from_numpy(nib.load(image_path).get_fdata())
    ).all()
    assert (data_point["image_3"].tensor == image.tensor).all()
    assert (data_point["image_4"].tensor == image.tensor).all()
    np.testing.assert_allclose(data_point["image_4"].affine, image.affine)

    assert isinstance(data_point["mask_2"], tio.LabelMap)
    assert isinstance(data_point["mask_3"], tio.LabelMap)
    assert isinstance(data_point["mask_4"], tio.LabelMap)
    assert (
        data_point["mask_2"].tensor == torch.from_numpy(nib.load(mask_path).get_fdata())
    ).all()
    assert (data_point["mask_3"].tensor == mask.tensor).all()
    np.testing.assert_allclose(data_point["mask_4"].affine, image.affine)

    assert data_point.participant_id == "sub-000"
    assert data_point.session_id == "ses-M000"

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
    assert set(data_point.get_images_dict().keys()) == {
        "image",
        "image_2",
        "image_3",
        "image_4",
    }
    assert len(data_point.get_images_dict(intensity_only=False)) == 8
    assert len(data_point.get_images_dict(intensity_only=False, include=["image"])) == 1
    assert len(data_point.get_images_dict(intensity_only=False, exclude=["image"])) == 7

    # get masks
    assert set(data_point.get_masks_dict(include=["mask_1", "mask_2"]).keys()) == {
        "mask_1",
        "mask_2",
    }
    assert set(data_point.get_masks_dict(exclude=["mask_1", "mask_2"]).keys()) == {
        "mask_3",
        "mask_4",
    }

    assert data_point.get_image_tensor("image").shape == (1, 3, 3, 3)

    # get other fields
    assert data_point.get_non_images_dict() == {
        "other": 0,
        "participant_id": "sub-000",
        "session_id": "ses-M000",
    }
    assert set(
        data_point.get_non_images_dict(include=["participant_id", "session_id"]).keys()
    ) == {"participant_id", "session_id"}
    assert set(data_point.get_non_images_dict(exclude=["participant_id"]).keys()) == {
        "session_id",
        "other",
    }

    # get_keys
    assert sorted(data_point.get_keys()) == sorted(list(data_point.keys()))
    assert sorted(data_point.get_keys(include=["participant_id", "image"])) == [
        "image",
        "participant_id",
    ]
    assert sorted(data_point.get_keys(exclude=["participant_id", "image"])) == sorted(
        [key for key in data_point.keys() if key not in ["participant_id", "image"]]
    )
    assert sorted(
        data_point.get_keys(
            include=["participant_id", "image"], exclude=["participant_id"]
        )
    ) == ["image"]

    # test copy
    data_point = SubDataPoint(**data_point)
    c = copy(data_point)
    assert isinstance(c, SubDataPoint)
    assert isinstance(c.image, tio.ScalarImage)
    assert c.participant_id == "sub-000"
    assert c.session_id == "ses-M000"
    assert isinstance(c["mask_3"], tio.LabelMap)

    # other attributes
    data_point = DataPoint(
        image=image,
        participant_id="sub-000",
        session_id="ses-M000",
        age=1,
    )
    data_point["x"] = np.array([1])
    assert (data_point.image.tensor == image.tensor).all()
    assert data_point.age == 1
    np.testing.assert_allclose(data_point.x, np.array([1]))
    data_point["x"] = "x"
    assert data_point["x"] == "x"

    # spacing, shape
    data_point = DataPoint(
        image,
        participant_id="sub-000",
        session_id="ses-M00",
    )
    data_point.add_mask(mask_path, "mask")
    assert data_point.spacing == (1.3, 1.2, 1.1)
    assert (data_point.affine == np.diag([1.3, 1.2, 1.1, 1])).all()
    assert data_point.spatial_shape == (3, 3, 3)
    assert data_point.shape == (1, 3, 3, 3)

    # transforms history
    transform = tio.Clamp(out_min=0, out_max=1)
    transformed_datapoint = transform(data_point)
    assert len(transformed_datapoint.get_applied_transforms()) == 1
    assert isinstance(transformed_datapoint.get_applied_transforms()[0], tio.Clamp)

    # test attributes update
    data_point["abc"] = 0
    assert data_point.abc == 0
    data_point["abc"] = 1
    assert data_point.abc == 1
    del data_point["abc"]
    assert not hasattr(data_point, "abc")
