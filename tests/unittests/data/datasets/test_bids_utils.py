import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.datasets.bids_utils import BidsNiftiDataset, BidsTensorDataset
from clinicadl.data.structures import (
    CommonMask,
    DataPoint,
    Image,
    IndividualMask,
    Tensor,
)
from clinicadl.io import Bids, BidsFileType, TensorType
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.extraction import Slice

BIDS = Path(__file__).parents[2] / "resources" / "bids"
TENSORS = BIDS / "derivatives" / "tensors"


class TestBidsNiftiDataset:
    def test_getitem(self):
        dataset = BidsNiftiDataset(
            image=Image(
                Bids(BIDS),
                BidsFileType(data_type="anat", suffix="T1w"),
            ),
            transforms=TransformsHandler(
                extraction=Slice(),
                image_transforms=[tio.Crop(cropping=(0, 0, 0, 1, 0, 0))],
            ),
            columns=["age"],
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [0, 1],
                }
            ),
            masks={
                "mask1": IndividualMask(
                    Bids(BIDS / "derivatives" / "masks"),
                    BidsFileType(data_type="anat", suffix="dseg"),
                ),
                "mask2": CommonMask(
                    mask_2_path := BIDS
                    / "derivatives"
                    / "caps"
                    / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
                ),
            },
        )
        assert list(dataset.individual_masks.keys()) == ["mask1"]
        assert list(dataset.common_masks.keys()) == ["mask2"]
        sample = dataset[3]
        assert sample.sample_position == 0
        assert sample.participant == "sub-010"
        assert sample.spatial_shape == (1, 2, 3)
        assert sample["age"] == 1.0
        assert set(sample.get_masks_dict().keys()) == {"mask1", "mask2"}

        image_path = (
            BIDS
            / "sub-010"
            / "ses-M003"
            / "anat"
            / "sub-010_ses-M003_res-1d3x1d2x1d1_T1w.nii.gz"
        )
        torch.testing.assert_close(
            tio.ScalarImage(path=image_path).tensor[:, 0:1, :2, :],
            sample["image"].tensor,
        )
        mask_1_path = (
            BIDS
            / "derivatives"
            / "masks"
            / "sub-010"
            / "ses-M003"
            / "anat"
            / "sub-010_ses-M003_res-1d3x1d2x1d1_seg-FreeSurfer_dseg.nii.gz"
        )
        torch.testing.assert_close(
            tio.LabelMap(path=mask_1_path).tensor[:, 0:1, :2, :],
            sample["mask1"].tensor,
        )
        torch.testing.assert_close(
            tio.LabelMap(path=mask_2_path).tensor[:, 0:1, :2, :],
            sample["mask2"].tensor,
        )

    def test_init(self):
        dataset = BidsNiftiDataset(
            image=Image(
                Bids(BIDS),
                BidsFileType(
                    data_type="anat", suffix="T1w", with_entities={"res": "1d3x1d2x1d1"}
                ),
            ),
            transforms=TransformsHandler(),
            columns=None,
            data=None,
            masks=None,
        )
        pd.testing.assert_frame_equal(
            dataset.df,
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003", "ses-M003"],
                }
            ),
        )

    def test_init_error(self):
        with pytest.raises(
            RuntimeError, match=r"For \(sub-100 \| ses-M000\).*more than 1 file found"
        ):
            BidsNiftiDataset(
                image=Image(Bids(BIDS), BidsFileType(data_type="pet", suffix="pet")),
                transforms=TransformsHandler(),
                columns=None,
                data=None,
                masks=None,
            )
        with pytest.raises(RuntimeError, match=f"No image found in {BIDS} for .*"):
            BidsNiftiDataset(
                image=Image(Bids(BIDS), BidsFileType(data_type="pet", suffix="T1w")),
                transforms=TransformsHandler(),
                columns=None,
                data=None,
                masks=None,
            )
        with pytest.raises(
            AssertionError,
            match=r"For \(sub-999, ses-M999\), no image associated with suffix=re.compile\('T1w'\).*",
        ):
            BidsNiftiDataset(
                image=Image(Bids(BIDS), BidsFileType(data_type="anat", suffix="T1w")),
                transforms=TransformsHandler(),
                columns=None,
                data=pd.DataFrame(
                    {
                        "participant_id": ["sub-000", "sub-999"],
                        "session_id": ["ses-M000", "ses-M999"],
                    }
                ),
                masks=None,
            )
        with pytest.raises(
            AssertionError,
            match=r"For \(sub-999, ses-M999\), no mask associated with suffix=re.compile\('dseg'\).*",
        ):
            BidsNiftiDataset(
                image=Image(
                    Bids(BIDS),
                    BidsFileType(
                        data_type="pet", suffix="pet", without_entities={"desc": "Crop"}
                    ),
                ),
                transforms=TransformsHandler(),
                columns=None,
                data=pd.DataFrame(
                    {
                        "participant_id": ["sub-000", "sub-999"],
                        "session_id": ["ses-M000", "ses-M999"],
                    }
                ),
                masks={
                    "mask": IndividualMask(
                        Bids(BIDS / "derivatives" / "masks"),
                        BidsFileType(data_type="anat", suffix="dseg"),
                    )
                },
            )
        with pytest.raises(
            AssertionError,
            match=f"Cannot find the mask in {BIDS / 'mask.nii.gz'}",
        ):
            BidsNiftiDataset(
                image=Image(
                    Bids(BIDS),
                    BidsFileType(
                        data_type="pet", suffix="pet", without_entities={"desc": "Crop"}
                    ),
                ),
                masks={"mask": CommonMask(BIDS / "mask.nii.gz")},
                columns=None,
                data=None,
                transforms=TransformsHandler(),
            )
        with pytest.raises(
            KeyError,
            match="No column named 'age'",
        ):
            BidsNiftiDataset(
                image=Image(
                    Bids(BIDS),
                    BidsFileType(
                        data_type="pet", suffix="pet", without_entities={"desc": "Crop"}
                    ),
                ),
                masks=None,
                columns=["age"],
                data=None,
                transforms=TransformsHandler(),
            )
        with pytest.raises(
            ValueError,
            match=r"A mask cannot be named 'sample_type'",
        ):
            BidsNiftiDataset(
                image=Image(
                    Bids(BIDS),
                    BidsFileType(
                        data_type="pet", suffix="pet", without_entities={"desc": "Crop"}
                    ),
                ),
                masks={"sample_type": None},
                columns=None,
                data=None,
                transforms=TransformsHandler(),
            )
        with pytest.raises(
            ValueError,
            match=r"'age' is in columns and masks!",
        ):
            BidsNiftiDataset(
                image=Image(
                    Bids(BIDS),
                    BidsFileType(
                        data_type="pet", suffix="pet", without_entities={"desc": "Crop"}
                    ),
                ),
                masks={"age": None},
                columns=["age"],
                data=pd.DataFrame(
                    {
                        "participant_id": ["sub-000", "sub-010"],
                        "session_id": ["ses-M000", "ses-M003"],
                        "age": [0, 1],
                    }
                ),
                transforms=TransformsHandler(),
            )


class TestBidsTensorDataset:
    def test_getitem(self):
        dataset = BidsTensorDataset(
            tensor=Tensor(
                Bids(TENSORS),
                TensorType(entities={"conv": "T1Transform"}),
                to_load=["brain", "left_hippo", "other_image"],
            ),
            transforms=TransformsHandler(
                extraction=Slice(),
                image_transforms=[tio.Crop(cropping=(0, 0, 0, 1, 0, 0))],
            ),
            columns=["age"],
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [0, 1],
                }
            ),
        )
        sample = dataset[3]
        np.testing.assert_allclose(sample.spacing, (1.1, 1.1, 1.1))
        assert sample.sample_position == 1
        assert sample.participant == "sub-010"
        assert sample.spatial_shape == (1, 1, 2)
        assert sample["age"] == 1.0
        assert set(sample.get_images_dict().keys()) == {"image", "other_image"}
        assert set(sample.get_masks_dict().keys()) == {"brain", "left_hippo"}

        pt_path = (
            TENSORS
            / "sub-010"
            / "ses-M003"
            / "tensors"
            / "sub-010_ses-M003_src-T1w_conv-T1Transform_tensors.pt"
        )
        tensors = torch.load(pt_path, weights_only=False)
        torch.testing.assert_close(
            tensors["images"]["image"][0][:, 1:2, 0:1, :],
            sample["image"].tensor,
        )
        torch.testing.assert_close(
            tensors["images"]["other_image"][0][:, 1:2, 0:1, :],
            sample["other_image"].tensor,
        )
        torch.testing.assert_close(
            tensors["masks"]["brain"][0][:, 1:2, 0:1, :],
            sample["brain"].tensor,
        )
        torch.testing.assert_close(
            tensors["masks"]["left_hippo"][0][:, 1:2, 0:1, :],
            sample["left_hippo"].tensor,
        )

    def test_init(self):
        dataset = BidsTensorDataset(
            tensor=Tensor(
                Bids(TENSORS),
                TensorType(entities={"conv": "T1Masks"}),
            ),
            transforms=TransformsHandler(),
            columns=None,
            data=None,
        )
        pd.testing.assert_frame_equal(
            dataset.df,
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                }
            ),
        )

    def test_init_error(self):
        with pytest.raises(
            AssertionError,
            match=r"For \(sub-999, ses-M999\), no image associated with suffix=re.compile\('tensors'\).*",
        ):
            BidsTensorDataset(
                tensor=Tensor(
                    Bids(TENSORS),
                    TensorType(entities={"conv": "T1Masks"}),
                ),
                transforms=TransformsHandler(),
                columns=None,
                data=pd.DataFrame(
                    {
                        "participant_id": ["sub-000", "sub-999"],
                        "session_id": ["ses-M000", "ses-M999"],
                    }
                ),
            )
