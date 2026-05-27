from pathlib import Path

import numpy as np
import torch
import torchio as tio

from clinicadl.data.structures import (
    CommonMask,
    DataPoint,
    Image,
    IndividualMask,
    Tensor,
    TensorContent,
)
from clinicadl.io import Bids, BidsFileType, T1Linear, TensorType
from clinicadl.utils.json import read_json

CAPS = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "caps"
MASKS = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "masks"
TENSORS = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "tensors"


def test_image():
    caps = Bids(CAPS)
    file_type = T1Linear(use_uncropped_image=True)
    image = Image(caps, file_type)
    assert isinstance(image.get("sub-000", "ses-M000"), tio.ScalarImage)


def test_individual_mask():
    masks = Bids(MASKS)
    file_type = BidsFileType(data_type="anat", suffix="dseg")
    mask = IndividualMask(masks, file_type)
    assert isinstance(mask.get("sub-000", "ses-M000"), tio.LabelMap)


def test_common_mask():
    mask = CommonMask(
        CAPS / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
    )
    assert isinstance(mask.get(), tio.LabelMap)
    assert mask._mask is not None
    assert isinstance(mask.get(), tio.LabelMap)


class TestTensorContent:
    def test_from_datapoint(self):
        d = DataPoint(
            image=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)),
            image_=tio.ScalarImage(
                path=(
                    img_path := CAPS
                    / "subjects"
                    / "sub-000"
                    / "ses-M000"
                    / "t1_linear"
                    / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
                )
            ),
            mask=tio.LabelMap(tensor=torch.zeros(1, 1, 1, 1)),
            mask_=tio.LabelMap(
                path=(
                    mask_path := CAPS
                    / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
                )
            ),
            participant="",
            session="",
            other=0,
            other_=1,
        )

        tensor = TensorContent.from_datapoint(d)
        assert set(tensor.images.keys()) == {"image", "image_"}
        torch.testing.assert_close(
            tensor.images["image"].tensor, torch.zeros(1, 2, 2, 2)
        )
        assert set(tensor.masks.keys()) == {"mask", "mask_"}
        torch.testing.assert_close(tensor.masks["mask"].tensor, torch.zeros(1, 1, 1, 1))
        assert tensor.additional_data == {"other": 0, "other_": 1}
        assert tensor.paths == [img_path, mask_path]

        tensor = TensorContent.from_datapoint(d, include=["image", "mask_", "other"])
        assert set(tensor.images.keys()) == {"image"}
        assert set(tensor.masks.keys()) == {"mask_"}
        assert set(tensor.additional_data.keys()) == {"other"}
        assert tensor.paths == [mask_path]

    def test_save_load(self, tmp_path):
        tensor = TensorContent(
            images={
                "image": tio.ScalarImage(
                    tensor=torch.zeros(1, 2, 2, 2), affine=torch.eye(4) * 2
                ),
                "image_": tio.ScalarImage(
                    tensor=torch.ones(1, 2, 2, 2), affine=torch.eye(4)
                ),
            },
            masks={
                "mask": tio.LabelMap(
                    tensor=torch.zeros(1, 1, 1, 1), affine=torch.eye(4) * 2
                ),
                "mask_": tio.LabelMap(
                    tensor=torch.ones(1, 1, 1, 1), affine=torch.eye(4)
                ),
            },
            additional_data={"other": 0, "other_": 1},
            paths=["dir/path.nii.gz", "dir_/path_.nii.gz"],
        )
        tensor.save(tmp_path / "tensor.pt")
        json = read_json(tmp_path / "tensor.json")
        assert json["Sources"] == ["file://dir/path.nii.gz", "file://dir_/path_.nii.gz"]

        tensor = TensorContent.load(tmp_path / "tensor.pt")
        assert set(tensor.images.keys()) == {"image", "image_"}
        torch.testing.assert_close(
            tensor.images["image"].tensor, torch.zeros(1, 2, 2, 2)
        )
        np.testing.assert_allclose(tensor.images["image"].affine, np.eye(4) * 2)
        assert set(tensor.masks.keys()) == {"mask", "mask_"}
        torch.testing.assert_close(tensor.masks["mask"].tensor, torch.zeros(1, 1, 1, 1))
        np.testing.assert_allclose(tensor.masks["mask"].affine, np.eye(4) * 2)
        assert tensor.additional_data == {"other": 0, "other_": 1}
        assert tensor.paths == ["dir/path.nii.gz", "dir_/path_.nii.gz"]


def test_tensor():
    tensors = Bids(TENSORS)
    tensor_type = TensorType({"conv": "T1Transform"})
    tensor = Tensor(tensors, tensor_type)
    data = tensor.get("sub-000", "ses-M000")
    assert set(data.get_images_dict().keys()) == {"image", "other_image"}
    assert set(data.get_masks_dict().keys()) == {
        "seg",
        "brain",
        "left_hippo",
        "right_hippo",
        "other_mask",
    }
    assert set(data.get_non_images_dict().keys()) == {
        "coefficient",
        "image_path",
        "file_type",
        "participant",
        "session",
    }

    tensor = Tensor(tensors, tensor_type, to_load=["seg", "coefficient"])
    data = tensor.get("sub-000", "ses-M000")
    assert set(data.get_images_dict().keys()) == {"image"}
    assert set(data.get_masks_dict().keys()) == {"seg"}
    assert data["image"].spatial_shape == (2, 2, 2)
    np.testing.assert_allclose(data["image"].affine, np.diag([1.1, 1.1, 1.1, 1]))
    assert data["seg"].spatial_shape == (2, 2, 2)
    np.testing.assert_allclose(data["seg"].affine, np.diag([1.1, 1.1, 1.1, 1]))
    assert data["coefficient"] == 0.5
