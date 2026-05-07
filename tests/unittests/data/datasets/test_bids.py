import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.datasets import BidsDataset
from clinicadl.data.structures import Sample2D
from clinicadl.io import Bids, BidsFileType
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import CannotReadJsonFieldError

BIDS = Path(__file__).parents[2] / "resources" / "bids"
MASKS = BIDS / "derivatives" / "masks"
CAPS = BIDS / "derivatives" / "caps"


def transform(x: dict) -> dict:
    x["new_data"] = 0
    x.add_image(torch.ones(1, 2, 2, 2), "new_image")
    return x


class TestBidsDataset:
    def test__init__(self):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            masks={
                "mask1": BidsFileType(suffix="mask", data_type="anat"),
                "mask2": (Bids(MASKS), BidsFileType(suffix="dseg", data_type="anat")),
                "mask3": CAPS
                / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii",
            },
        )
        assert dataset.transforms == TransformsHandler()
        assert dataset.image.bids.path == BIDS
        assert dataset.image.file_type == BidsFileType(suffix="T1w", data_type="anat")
        assert set(dataset.individual_masks.keys()) == {"mask1", "mask2"}
        assert set(dataset.common_masks.keys()) == {"mask3"}
        assert dataset.individual_masks["mask1"].bids.path == BIDS
        assert dataset.individual_masks["mask2"].bids.path == MASKS
        assert (
            dataset.common_masks["mask3"].file.path
            == CAPS
            / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
        )
        pd.testing.assert_frame_equal(
            dataset.df,
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-000", "sub-010", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003", "ses-M003", "ses-M012"],
                }
            ),
        )

        dataset = BidsDataset(
            bids=Bids(BIDS),
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            masks={
                "mask3": str(
                    CAPS
                    / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
                ),
            },
        )
        assert dataset.image.bids.path == BIDS
        assert (
            dataset.common_masks["mask3"].file.path
            == CAPS
            / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
        )

    def test__len__(self):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            transforms=TransformsHandler(extraction=Slice()),
        )
        assert len(dataset) == 10

    def test__getitem__(self):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            transforms=TransformsHandler(
                extraction=Slice(), sample_transforms=[tio.Crop((0, 0, 0, 1, 0, 0))]
            ),
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [1, 2],
                }
            ),
            columns={"age": lambda x: x * 10},
            masks={
                "mask1": BidsFileType(suffix="mask", data_type="anat"),
                "mask2": (Bids(MASKS), BidsFileType(suffix="dseg", data_type="anat")),
                "mask3": (
                    mask_3_path := CAPS
                    / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
                ),
            },
        )

        sample = dataset[4]
        assert isinstance(sample, Sample2D)
        assert sample.participant == "sub-010"
        assert sample.session == "ses-M003"
        assert sample.sample_position == 1
        assert sample.file_type[0] == BidsFileType(suffix="T1w", data_type="anat")
        assert sample["age"] == 20.0

        image_path = (
            BIDS
            / "sub-010"
            / "ses-M003"
            / "anat"
            / "sub-010_ses-M003_res-1d3x1d2x1d1_T1w.nii.gz"
        )
        torch.testing.assert_close(
            tio.ScalarImage(path=image_path).tensor[:, 1:2, :2, :],
            sample["image"].tensor,
        )
        mask_1_path = (
            BIDS
            / "sub-010"
            / "ses-M003"
            / "anat"
            / "sub-010_ses-M003_res-1d3x1d2x1d1_label-brain_mask.nii.gz"
        )
        torch.testing.assert_close(
            tio.LabelMap(path=mask_1_path).tensor[:, 1:2, :2, :],
            sample["mask1"].tensor,
        )
        mask_2_path = (
            BIDS
            / "derivatives"
            / "masks"
            / "sub-010"
            / "ses-M003"
            / "anat"
            / "sub-010_ses-M003_res-1d3x1d2x1d1_seg-FreeSurfer_dseg.nii.gz"
        )
        torch.testing.assert_close(
            tio.LabelMap(path=mask_2_path).tensor[:, 1:2, :2, :],
            sample["mask2"].tensor,
        )
        torch.testing.assert_close(
            tio.LabelMap(path=mask_3_path).tensor[:, 1:2, :2, :],
            sample["mask3"].tensor,
        )

    def test_describe(self):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
        )
        assert dataset.describe()["file_type"]["suffix"] == "T1w"

    def test_sanity_check(self):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="pet", data_type="pet"),
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-999"],
                    "session_id": ["ses-M000", "ses-M999"],
                }
            ),
        )
        with pytest.raises(
            RuntimeError, match="Different voxel spacing found in the dataset"
        ):
            dataset.sanity_check(spatial_checks=["global_spacing"])

    def test_subset(self):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
        ).subset([("sub-000", "ses-M000"), ("sub-010", "ses-M003")])
        assert dataset.get_participant_session_couples() == {
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        }

    def test_get_sample_info(self):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            transforms=TransformsHandler(extraction=Slice()),
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [1, 2],
                }
            ),
            columns=["age"],
        )
        dataset.get_sample_info(5, "age") == 2

    def test_train_eval(self):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            transforms=TransformsHandler(
                augmentations=[tio.Crop(cropping=(0, 1, 0, 1, 0, 1))]
            ),
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                }
            ),
        )
        assert dataset[0].image.spatial_shape == (2, 2, 2)
        dataset.eval()
        assert dataset[0].image.spatial_shape == (3, 3, 3)
        dataset.train()
        assert dataset[0].image.spatial_shape == (2, 2, 2)

    def test_from_to_json(self, tmp_path):
        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            masks={
                "mask1": BidsFileType(suffix="mask", data_type="anat"),
                "mask2": (Bids(MASKS), BidsFileType(suffix="dseg", data_type="anat")),
                "mask3": CAPS
                / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii",
            },
            transforms=TransformsHandler(extraction=Slice(slice_direction=1)),
        )
        dataset.to_json(tmp_path / "dataset.json")
        new_dataset = dataset.from_json(tmp_path / "dataset.json")
        assert dataset.config == new_dataset.config

        dataset = BidsDataset(
            bids=BIDS,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            transforms=TransformsHandler(image_transforms=[tio.Crop(1)]),
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [1, 2],
                }
            ),
            columns={"age": lambda x: x * 10},
        )
        len(dataset)
        dataset.to_json(tmp_path / "dataset.json", overwrite=True)
        with pytest.raises(
            CannotReadJsonFieldError,
            match=re.escape("BidsDataset cannot read the field(s) ['transforms']"),
        ):
            dataset.from_json(tmp_path / "dataset.json")
        new_dataset = dataset.from_json(
            tmp_path / "dataset.json",
            transforms=TransformsHandler(image_transforms=[tio.Crop(1)]),
        )
        pd.testing.assert_frame_equal(dataset.config.data, new_dataset.config.data)

    def test_to_tensors(self, tmp_path):
        shutil.copytree(BIDS, tmp_path, dirs_exist_ok=True)

        dataset = BidsDataset(
            bids=tmp_path,
            file_type=BidsFileType(suffix="T1w", data_type="anat"),
            transforms=TransformsHandler(
                image_transforms=[tio.Crop((0, 1, 0, 1, 0, 1)), transform],
                extraction=Slice(),
            ),
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [1, 2],
                }
            ),
            columns={"age": lambda x: x * 10},
            masks={
                "mask1": BidsFileType(suffix="mask", data_type="anat"),
                "mask2": (
                    CAPS
                    / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
                ),
            },
        )
        tensor_dataset = dataset.to_tensors(conversion_name="x", save_transforms=True)
        assert (
            tmp_path / "derivatives" / "tensors" / "src-T1w_conv-x_description.json"
        ).exists()
        sample = tensor_dataset[0]
        assert sample["age"] == 10
        assert sample.spatial_shape == (1, 2, 2)
        assert "new_data" in sample
        assert set(sample.get_images_dict(intensity_only=False).keys()) == {
            "image",
            "new_image",
            "mask1",
            "mask2",
        }

        tensor_dataset = dataset.to_tensors(conversion_name="y", save_transforms=False)
        sample = tensor_dataset[0]
        assert sample.spatial_shape == (1, 2, 2)
        assert "new_data" in sample
