from pathlib import Path

import pandas as pd
import pytest
import torchio as tio

from clinicadl.data.datasets import TensorDataset
from clinicadl.data.datasets.tensor import TensorDescription
from clinicadl.io import BidsFileType
from clinicadl.transforms.config import CropConfig
from clinicadl.utils.json import read_json, write_json
from clinicadl.utils.tsvtools import read_data

BIDS = Path(__file__).parents[2] / "resources" / "bids"


@pytest.fixture
def tensor_description():
    return TensorDescription(
        tensor_type=BidsFileType(
            suffix="tensors",
            extension=".pt",
            data_type="tensors",
            with_entities={"src": "pet", "conv": "raw"},
        ),
        images=["image"],
        masks=["mask"],
        additional_data=["coeff"],
        transforms=[tio.Crop(1), CropConfig(cropping=1)],
        spacing=(1, 1, 1),
        spatial_shape=(3, 3, 3),
        interrupted=True,
        participants_sessions=pd.DataFrame(
            {"participant_id": ["sub-000"], "session_id": ["ses-M000"]}
        ),
    )


class TestTensorDescription:
    def test_get_json_filename(self, tensor_description: TensorDescription):
        assert str(tensor_description.get_json_filename(BIDS)) == str(
            BIDS / "src-pet_conv-raw_description.json"
        )

    def test_get_df_filename(self, tensor_description: TensorDescription):
        assert str(tensor_description.get_df_filename(BIDS)) == str(
            BIDS / "src-pet_conv-raw_participantsXsessions.tsv"
        )

    def test_read_write(self, tensor_description: TensorDescription, tmp_path):
        write_json(
            tmp_path / "dataset_description.json",
            {"BIDSVersion": "1", "DatasetType": "derivative"},
        )
        tensor_description.write(tmp_path)
        pd.testing.assert_frame_equal(
            read_data(tmp_path / "src-pet_conv-raw_participantsXsessions.tsv"),
            tensor_description.participants_sessions,
        )
        json = read_json(tmp_path / "src-pet_conv-raw_description.json")
        assert json["tensor_type"]["with_entities"] == {"src": "pet", "conv": "raw"}
        assert json["transforms"][0] == "Crop(cropping=1)"
        assert json["transforms"][1]["name_"] == "Crop"
        assert json["images"] == ["image"]
        assert json["masks"] == ["mask"]
        assert json["additional_data"] == ["coeff"]
        assert json["spacing"] == [1.0, 1.0, 1.0]
        assert json["spatial_shape"] == [3, 3, 3]
        assert json["interrupted"]
        assert "participants_sessions" not in json

        new_tensor_description = TensorDescription.read(
            tmp_path / "src-pet_conv-raw_description.json"
        )
        assert tensor_description.tensor_type == new_tensor_description.tensor_type
        assert new_tensor_description.transforms == []
        assert tensor_description.images == new_tensor_description.images
        assert tensor_description.masks == new_tensor_description.masks
        assert (
            tensor_description.additional_data == new_tensor_description.additional_data
        )
        assert tensor_description.spacing == new_tensor_description.spacing
        assert tensor_description.spatial_shape == new_tensor_description.spatial_shape
        assert tensor_description.interrupted == new_tensor_description.interrupted
        pd.testing.assert_frame_equal(
            tensor_description.participants_sessions,
            new_tensor_description.participants_sessions,
        )
