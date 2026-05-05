import re
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.datasets import TensorDataset
from clinicadl.data.structures import Sample2D
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import CannotReadJsonFieldError

TENSORS = Path(__file__).parents[2] / "resources" / "bids" / "derivatives" / "tensors"


@pytest.fixture
def change_test_dir(monkeypatch):
    monkeypatch.chdir(TENSORS)


@pytest.mark.usefixtures("change_test_dir")
class TestTensorDataset:
    def test__init__(self):
        dataset = TensorDataset(
            TENSORS / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json",
            to_load=["brain"],
        )
        assert dataset.transforms == TransformsHandler()
        assert dataset.image.bids.path == TENSORS
        assert dataset.image.file_type.with_entities == {
            "res": re.compile("1d3x1d2x1d1"),
            "src": re.compile("T1w"),
            "conv": re.compile("T1Masks"),
        }
        assert dataset.image.to_load == ["brain"]
        pd.testing.assert_frame_equal(
            dataset.df,
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                }
            ),
        )

    def test__len__(self):
        dataset = TensorDataset(
            TENSORS / "src-T1w_conv-T1Transform_description.json",
            transforms=TransformsHandler(extraction=Slice()),
        )
        assert len(dataset) == 4

    def test__getitem__(self):
        dataset = TensorDataset(
            TENSORS / "src-T1w_conv-T1Transform_description.json",
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
            to_load=["seg", "right_hippo", "other_image"],
        )

        sample = dataset[3]
        assert isinstance(sample, Sample2D)
        assert sample.participant == "sub-010"
        assert sample.session == "ses-M003"
        assert sample.sample_position == 1
        assert sample.file_type[0].with_entities == {
            "src": re.compile("T1w"),
            "conv": re.compile("T1Transform"),
        }
        assert sample.file_type[0].data_type == re.compile("tensors")
        assert sample["age"] == 20.0

        image_path = (
            TENSORS
            / "sub-010"
            / "ses-M003"
            / "tensors"
            / "sub-010_ses-M003_src-T1w_conv-T1Transform_tensors.pt"
        )
        image_file = torch.load(image_path)
        torch.testing.assert_close(
            image_file["images"]["image"][0][:, 1:2, :1, :],
            sample["image"].tensor,
        )
        torch.testing.assert_close(
            image_file["images"]["other_image"][0][:, 1:2, :1, :],
            sample["other_image"].tensor,
        )
        torch.testing.assert_close(
            image_file["masks"]["seg"][0][:, 1:2, :1, :],
            sample["seg"].tensor,
        )
        torch.testing.assert_close(
            image_file["masks"]["right_hippo"][0][:, 1:2, :1, :],
            sample["right_hippo"].tensor,
        )

    def test_describe(self):
        dataset = TensorDataset(
            TENSORS
            / "trc-18FAV45_res-0d8x0d8x0d8_src-pet_conv-PetSpacing0d8_description.json",
        )
        assert dataset.describe()["file_type"]["suffix"] == "tensors"

    def test_sanity_check(self):
        dataset = TensorDataset(
            TENSORS / "trc-18FAV45_src-pet_conv-raw_description.json",
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
        dataset = TensorDataset(
            TENSORS / "trc-18FAV45_src-pet_conv-raw_description.json",
        ).subset([("sub-000", "ses-M000"), ("sub-010", "ses-M003")])
        assert dataset.get_participant_session_couples() == {
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        }

    def test_get_sample_info(self):
        dataset = TensorDataset(
            TENSORS / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json",
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
        dataset = TensorDataset(
            TENSORS / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json",
            transforms=TransformsHandler(
                augmentations=[tio.Crop(cropping=(0, 1, 0, 1, 0, 1))]
            ),
        )
        assert dataset[0].image.spatial_shape == (2, 2, 2)
        dataset.eval()
        assert dataset[0].image.spatial_shape == (3, 3, 3)
        dataset.train()
        assert dataset[0].image.spatial_shape == (2, 2, 2)

    def test_from_to_json(self, tmp_path):
        dataset = TensorDataset(
            TENSORS / "src-T1w_conv-T1Transform_description.json",
            transforms=TransformsHandler(extraction=Slice()),
            data=pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [1, 2],
                }
            ),
            columns=["age"],
            to_load=["seg", "right_hippo", "other_image"],
        )
        dataset.to_json(tmp_path / "dataset.json")
        new_dataset = dataset.from_json(tmp_path / "dataset.json")
        assert dataset.config == new_dataset.config

        dataset = TensorDataset(
            TENSORS / "src-T1w_conv-T1Transform_description.json",
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
            match=re.escape("TensorDataset cannot read the field(s) ['transforms']"),
        ):
            dataset.from_json(tmp_path / "dataset.json")
        new_dataset = dataset.from_json(
            tmp_path / "dataset.json",
            transforms=TransformsHandler(image_transforms=[tio.Crop(1)]),
        )
        pd.testing.assert_frame_equal(dataset.config.data, new_dataset.config.data)
