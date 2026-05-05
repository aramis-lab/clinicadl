import logging
from pathlib import Path

import pandas as pd
import pytest
import torchio as tio

from clinicadl.data.structures import CommonMask, Image, IndividualMask
from clinicadl.data.tensors import TensorDescription
from clinicadl.io import Bids, BidsFileType
from clinicadl.transforms.config import CropConfig, PadConfig
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.tsvtools import read_data

BIDS = Path(__file__).parents[2] / "resources" / "bids"


def _tensor_description(custom_transforms: bool):
    return TensorDescription(
        tensor_type=BidsFileType(
            suffix="tensors",
            extension=".pt",
            data_type="tensors",
            with_entities={"src": "pet", "conv": "raw"},
        ),
        image=Image(Bids(BIDS), BidsFileType(suffix="T1w", data_type="anat")),
        masks={
            "common_mask": CommonMask(BIDS / "mask.nii.gz"),
            "individual_mask": IndividualMask(
                Bids(BIDS), BidsFileType(suffix="mask", data_type="anat")
            ),
        },
        additional_data=["coeff"],
        transforms=[
            tio.Crop(1) if custom_transforms else CropConfig(cropping=1),
            PadConfig(padding=1),
        ],
        spacing=(1, 1, 1),
        spatial_shape=(3, 3, 3),
        interrupted=True,
        participants_sessions=pd.DataFrame(
            {"participant_id": ["sub-000"], "session_id": ["ses-M000"]}
        ),
    )


@pytest.fixture
def tensor_description():
    return _tensor_description(custom_transforms=False)


@pytest.fixture
def tensor_description_with_trasnforms():
    return _tensor_description(custom_transforms=True)


class TestTensorDescription:
    def test_get_json_filename(self, tensor_description: TensorDescription):
        assert str(tensor_description.get_json_filename(BIDS)) == str(
            BIDS / "src-pet_conv-raw_description.json"
        )

    def test_get_df_filename(self, tensor_description: TensorDescription):
        assert str(tensor_description.get_df_filename(BIDS)) == str(
            BIDS / "src-pet_conv-raw_participantsXsessions.tsv"
        )

    def test_read_write(self, tensor_description: TensorDescription, tmp_path, caplog):
        write_json(
            tmp_path / "dataset_description.json",
            {"BIDSVersion": "1.10.0", "DatasetType": "derivative", "Name": "abc"},
        )
        with caplog.at_level(logging.INFO):
            tensor_description.write(tmp_path)
        assert f"Tensor conversion description saved in {tmp_path / 'src-pet_conv-raw_description.json'}"
        assert f"(participant, session) pairs converted saved in {tmp_path / 'src-pet_conv-raw_participantsXsessions.tsv'}"
        pd.testing.assert_frame_equal(
            read_data(tmp_path / "src-pet_conv-raw_participantsXsessions.tsv"),
            tensor_description.participants_sessions,
        )
        json = read_json(tmp_path / "src-pet_conv-raw_description.json")
        assert json["TensorType"]["with_entities"] == {"src": "pet", "conv": "raw"}
        assert json["Transforms"][0]["name_"] == "Crop"
        assert json["Transforms"][1]["name_"] == "Pad"
        assert json["Image"][0] == str(BIDS)
        assert json["Image"][1]["suffix"] == "T1w"
        assert set(json["Masks"].keys()) == {"common_mask", "individual_mask"}
        assert json["Masks"]["individual_mask"][0] == str(BIDS)
        assert json["Masks"]["individual_mask"][1]["suffix"] == "mask"
        assert json["Masks"]["common_mask"] == str(BIDS / "mask.nii.gz")
        assert json["AdditionalData"] == ["coeff"]
        assert json["Spacing"] == [1.0, 1.0, 1.0]
        assert json["SpatialShape"] == [3, 3, 3]
        assert json["Interrupted"]
        assert "participants_sessions" not in json

        new_tensor_description = TensorDescription.read(
            tmp_path / "src-pet_conv-raw_description.json"
        )
        assert tensor_description.tensor_type == new_tensor_description.tensor_type
        assert (
            tensor_description.image.file_type.suffix
            == new_tensor_description.image.file_type.suffix
        )
        assert (
            tensor_description.masks["individual_mask"].file_type.suffix
            == new_tensor_description.masks["individual_mask"].file_type.suffix
        )
        assert (
            tensor_description.masks["common_mask"]
            == tensor_description.masks["common_mask"]
        )
        assert (
            tensor_description.additional_data == new_tensor_description.additional_data
        )
        assert tensor_description.transforms == new_tensor_description.transforms
        assert tensor_description.spacing == new_tensor_description.spacing
        assert tensor_description.spatial_shape == new_tensor_description.spatial_shape
        assert tensor_description.interrupted == new_tensor_description.interrupted
        pd.testing.assert_frame_equal(
            tensor_description.participants_sessions,
            new_tensor_description.participants_sessions,
        )

    def test_read_write_with_custom_transform(
        self, tensor_description_with_trasnforms: TensorDescription, tmp_path
    ):
        write_json(
            tmp_path / "dataset_description.json",
            {"BIDSVersion": "1.10.0", "DatasetType": "derivative", "Name": "abc"},
        )
        tensor_description_with_trasnforms.write(tmp_path)
        json = read_json(tmp_path / "src-pet_conv-raw_description.json")
        assert json["Transforms"][0] == "Crop(cropping=1)"
        assert json["Transforms"][1]["name_"] == "Pad"

        new_tensor_description = TensorDescription.read(
            tmp_path / "src-pet_conv-raw_description.json"
        )
        assert new_tensor_description.transforms[0] == "Crop(cropping=1)"
        assert new_tensor_description.transforms[1].padding == 1
