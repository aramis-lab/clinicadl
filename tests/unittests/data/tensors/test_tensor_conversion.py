import json
import re
import shutil
import warnings
from copy import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint, TensorContent
from clinicadl.data.structures.images import CommonMask, Image, IndividualMask
from clinicadl.data.tensors import (
    TensorConversion,
    TensorDescription,
    tensor_conversion,
)
from clinicadl.io import Bids, BidsFileType
from clinicadl.io.maps.training.splits.tmp import TmpDir
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import (
    ClampConfig,
    CropConfig,
    PadConfig,
    RescaleIntensityConfig,
    ToCanonicalConfig,
)
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import TensorConversionError
from clinicadl.utils.json import read_json, write_json

BIDS = Path(__file__).parents[2] / "resources" / "bids"
T1_FILE_TYPE = BidsFileType(data_type="anat", suffix="T1w")
BRAIN_MASK = IndividualMask(
    Bids(BIDS), BRAIN_FILE_TYPE := BidsFileType(data_type="anat", suffix="mask")
)
SEG_MASK = IndividualMask(
    Bids(BIDS / "derivatives" / "masks"),
    SEG_FILE_TYPE := BidsFileType(data_type="anat", suffix="dseg"),
)
LEFT_HIPPO = CommonMask(
    LEFT_HIPPO_PATH := BIDS
    / "derivatives"
    / "caps"
    / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
)
RIGHT_HIPPO = CommonMask(
    RIGHT_HIPPO_PATH := BIDS
    / "derivatives"
    / "caps"
    / "space-MNI152NLin2009cSym_res-1x1x1_label-rightHippocampus_mask.nii.gz"
)


# @pytest.fixture(scope="class")
@pytest.fixture
def bids_dir(tmp_path):
    # bids_dir = tmp_path_factory.mktemp("bids")
    shutil.copytree(BIDS, tmp_path, dirs_exist_ok=True)
    return tmp_path


class CustomTransform(tio.Transform):
    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        datapoint.add_image(datapoint.image, "other_image")
        datapoint.add_mask(
            tio.LabelMap(
                tensor=torch.ones_like(datapoint.image.tensor),
                affine=datapoint.image.affine,
            ),
            "other_mask",
        )
        datapoint["coefficient"] = 0.5

        return datapoint


class CustomTransformBis(tio.Transform):
    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        datapoint.add_image(datapoint.image, "other_image")
        datapoint["other_mask"] = 1.0
        datapoint["coefficient"] = 0.5

        return datapoint


class BidsDataset:
    def __init__(
        self,
        bids_path: Path,
        sub_ses: set[tuple[str, str]],
        transforms: Optional[TransformsHandler] = None,
        individual_masks: Optional[dict[str, IndividualMask]] = None,
        common_masks: Optional[dict[str, CommonMask]] = None,
    ):
        self.image = Image(Bids(bids_path), T1_FILE_TYPE)
        self.sub_ses = sub_ses
        self.transforms = transforms or TransformsHandler()

        self.individual_masks = individual_masks or {}
        self.common_masks = common_masks or {}

    def get_participant_session_couples(self) -> set[tuple[str, str]]:
        return self.sub_ses

    def _get_images(self, participant: str, session: str) -> DataPoint:
        if participant == "sub-000":
            datapoint = DataPoint(
                participant=participant,
                session=session,
                image=tio.ScalarImage(
                    tensor=torch.ones(1, 3, 3, 3),
                    affine=np.diag([1.5, 1.5, 1.5, 1]),
                ),
            )
        elif participant == "sub-001":
            datapoint = DataPoint(
                participant=participant,
                session=session,
                image=tio.ScalarImage(
                    tensor=torch.zeros(1, 3, 3, 3),
                    affine=np.diag([1.5, 1.5, 1.5, 1]),
                ),
            )
        elif participant == "sub-002" or participant == "sub-003":
            datapoint = DataPoint(
                participant=participant,
                session=session,
                image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
            )
        elif participant == "sub-004":
            datapoint = DataPoint(
                participant=participant,
                session=session,
                image=tio.ScalarImage(
                    tensor=torch.randn(1, 3, 3, 3), affine=np.diag([1.5, 1.5, 1.5, 1])
                ),
            )
        elif participant == "sub-005":
            raise ValueError()
        elif participant == "sub-006":
            datapoint = DataPoint(
                participant=participant,
                session=session,
                image=tio.ScalarImage(
                    tensor=torch.randn(1, 2, 2, 2), affine=np.diag([1.5, 1.5, 1.5, 1])
                ),
            )

        datapoint.image.path = participant + ".nii.gz"

        for i, mask in enumerate(self.individual_masks | self.common_masks):
            if participant == "sub-004":
                datapoint.add_mask(
                    tio.LabelMap(
                        tensor=torch.zeros(1, 3, 3, 3) + i, affine=datapoint.affine + 1
                    ),
                    mask,
                )
            else:
                datapoint.add_mask(
                    tio.LabelMap(
                        tensor=torch.zeros(1, 3, 3, 3) + i, affine=datapoint.affine
                    ),
                    mask,
                )
            datapoint[mask].path = participant + "_" + f"mask{i}" + ".nii.gz"

        return datapoint


# PET_FILE_TYPE = BidsFileType(
#     data_type="pet",
#     suffix="pet",
#     with_entities={"trc": "18FAV45"},
#     without_entities={"desc": "Crop"},
# )


class TestToTensors:
    def test_control(self, bids_dir, caplog):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
                ("sub-001", "ses-M001"),
            },
            individual_masks={"brain": BRAIN_MASK, "seg": SEG_MASK},
            common_masks={"left_hippo": LEFT_HIPPO},
            transforms=TransformsHandler(
                image_transforms=[
                    CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
                    tio.Clamp(out_max=10),
                ]
            ),
        )

        converter = TensorConversion(dataset)
        with caplog.at_level("DEBUG"):
            tensor_description = converter.to_tensors(
                conversion_name="t1",
                n_proc=2,
                save_transforms=True,
                spatial_checks=["affine", "shape", "global_spacing", "global_shape"],
                description="a conversion",
            )

        assert len(caplog.records) == 4
        assert "Conversion of (sub-000, ses-M000)." in (
            caplog.records[0].message,
            caplog.records[1].message,
        )
        assert "Conversion of (sub-001, ses-M001)." in (
            caplog.records[0].message,
            caplog.records[1].message,
        )
        assert caplog.records[0].levelname == "DEBUG"
        assert caplog.records[1].levelname == "DEBUG"
        assert (
            caplog.records[2].message
            == f"Tensor conversion description saved in {bids_dir / 'derivatives' / 'tensors' / 'src-T1w_conv-t1_description.json'}"
        )
        assert caplog.records[2].levelname == "INFO"
        assert (
            caplog.records[3].message
            == f"(participant, session) pairs converted saved in {bids_dir / 'derivatives' / 'tensors' / 'src-T1w_conv-t1_participantsXsessions.tsv'}"
        )
        assert caplog.records[3].levelname == "INFO"

        json_tensor_description = TensorDescription.read(
            bids_dir / "derivatives" / "tensors" / "src-T1w_conv-t1_description.json"
        )
        assert (
            BidsFileType(**tensor_description.tensor_type.to_dict())
            == json_tensor_description.tensor_type
        )
        assert tensor_description.tensor_type.with_entities == {
            "src": re.compile("T1w"),
            "conv": re.compile("t1"),
        }
        assert tensor_description.image == json_tensor_description.image
        assert tensor_description.image == (bids_dir, T1_FILE_TYPE)
        assert tensor_description.masks == json_tensor_description.masks
        assert tensor_description.masks == {
            "brain": (BIDS, BRAIN_FILE_TYPE),
            "seg": (BIDS / "derivatives" / "masks", SEG_FILE_TYPE),
            "left_hippo": LEFT_HIPPO_PATH,
        }
        assert tensor_description.transforms[0] == CropConfig(
            cropping=(0, 1, 0, 1, 0, 1)
        )
        assert isinstance(tensor_description.transforms[1], tio.Clamp)
        assert json_tensor_description.transforms == [
            CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
            "Clamp(out_min=None, out_max=10)",
        ]
        assert tensor_description.spacing == json_tensor_description.spacing
        assert tensor_description.spacing == (1.5, 1.5, 1.5)
        assert tensor_description.spatial_shape == json_tensor_description.spatial_shape
        assert tensor_description.spatial_shape == (2, 2, 2)
        assert tensor_description.interrupted == json_tensor_description.interrupted
        assert not tensor_description.interrupted
        assert tensor_description.description == json_tensor_description.description
        assert tensor_description.description == "a conversion"

        pd.testing.assert_frame_equal(
            tensor_description.participants_sessions,
            json_tensor_description.participants_sessions,
        )
        pd.testing.assert_frame_equal(
            tensor_description.participants_sessions,
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-001"],
                    "session_id": ["ses-M000", "ses-M001"],
                }
            ),
        )

        tensor = TensorContent.load(
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-000"
            / "ses-M000"
            / "tensors"
            / "sub-000_ses-M000_src-T1w_conv-t1_tensors.pt"
        )
        assert set(tensor.images.keys()) == {"image"}
        torch.testing.assert_close(
            tensor.images["image"].tensor, torch.ones(1, 2, 2, 2)
        )
        np.testing.assert_allclose(
            tensor.images["image"].affine, torch.diag(torch.tensor([1.5, 1.5, 1.5, 1]))
        )

        assert set(tensor.masks.keys()) == {"brain", "seg", "left_hippo"}
        torch.testing.assert_close(
            tensor.masks["brain"].tensor, torch.zeros(1, 2, 2, 2)
        )
        np.testing.assert_allclose(
            tensor.masks["brain"].affine, torch.diag(torch.tensor([1.5, 1.5, 1.5, 1]))
        )
        torch.testing.assert_close(
            tensor.masks["left_hippo"].tensor, torch.zeros(1, 2, 2, 2) + 2
        )
        np.testing.assert_allclose(
            tensor.masks["left_hippo"].affine,
            torch.diag(torch.tensor([1.5, 1.5, 1.5, 1])),
        )
        json = read_json(
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-000"
            / "ses-M000"
            / "tensors"
            / "sub-000_ses-M000_src-T1w_conv-t1_tensors.json"
        )
        assert json["Sources"] == [
            "file://sub-000.nii.gz",
            "file://sub-000_mask0.nii.gz",
            "file://sub-000_mask1.nii.gz",
            "file://sub-000_mask2.nii.gz",
        ]

        tensor = TensorContent.load(
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-001"
            / "ses-M001"
            / "tensors"
            / "sub-001_ses-M001_src-T1w_conv-t1_tensors.pt"
        )
        assert set(tensor.images.keys()) == {"image"}
        torch.testing.assert_close(
            tensor.images["image"].tensor, torch.zeros(1, 2, 2, 2)
        )
        assert set(tensor.masks.keys()) == {"brain", "seg", "left_hippo"}
        json = read_json(
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-001"
            / "ses-M001"
            / "tensors"
            / "sub-001_ses-M001_src-T1w_conv-t1_tensors.json"
        )
        assert json["Sources"] == [
            "file://sub-001.nii.gz",
            "file://sub-001_mask0.nii.gz",
            "file://sub-001_mask1.nii.gz",
            "file://sub-001_mask2.nii.gz",
        ]

    def test_without_transforms(self, bids_dir, caplog):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
            transforms=TransformsHandler(
                image_transforms=[
                    CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
                ]
            ),
        )

        converter = TensorConversion(dataset)
        with pytest.raises(
            ValueError,
            match="You cannot pass a description if you don't pass a conversion_name.",
        ):
            converter.to_tensors(
                conversion_name=None,
                save_transforms=False,
                spatial_checks=None,
                description="abc",
            )

        tensor_description = converter.to_tensors(
            conversion_name=None,
            save_transforms=False,
            spatial_checks=None,
        )

        assert tensor_description.transforms == []
        assert (
            tensor_description.description
            == "Raw images are converted without any transformation."
        )
        tensors = TensorContent.load(
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-000"
            / "ses-M000"
            / "tensors"
            / "sub-000_ses-M000_src-T1w_conv-raw_tensors.pt",
        )
        assert tensors.images["image"].spatial_shape == (3, 3, 3)  # not cropped

        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
        )
        with caplog.at_level("INFO"):
            TensorConversion(dataset).to_tensors(
                conversion_name="name",
                save_transforms=True,
                spatial_checks=None,
            )
        assert (
            caplog.records[0].message
            == "save_transforms is True, but there are no image transform."
        )

    def test_custom_transform(self, bids_dir):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
            transforms=TransformsHandler(
                image_transforms=[
                    CustomTransform(),
                ]
            ),
            individual_masks={"brain": BRAIN_MASK},
        )
        converter = TensorConversion(dataset)
        with pytest.raises(
            ValueError,
            match="Please pass a conversion_name if save_transforms is True.",
        ):
            converter.to_tensors(
                conversion_name=None,
                save_transforms=True,
                spatial_checks=None,
            )

        conversion_info = converter.to_tensors(
            conversion_name="custom",
            save_transforms=True,
            spatial_checks=None,
        )

        assert set(conversion_info.additional_data) == {
            "other_image",
            "other_mask",
            "coefficient",
        }
        tensors = TensorContent.load(
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-000"
            / "ses-M000"
            / "tensors"
            / "sub-000_ses-M000_src-T1w_conv-custom_tensors.pt",
        )
        assert set(tensors.images.keys()) == {"image", "other_image"}
        assert set(tensors.masks.keys()) == {"brain", "other_mask"}
        assert set(tensors.additional_data.keys()) == {"coefficient"}

    def test_spatial_checks(self, bids_dir, caplog):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
                ("sub-002", "ses-M000"),
                ("sub-003", "ses-M000"),
            },
        )
        with caplog.at_level("WARNING"):
            conversion_info = TensorConversion(dataset).to_tensors(
                conversion_name="global",
                spatial_checks=["global_spacing", "global_shape"],
                save_transforms=False,
            )
        assert len(caplog.records) == 2
        assert "Different voxel spacing" in caplog.records[0].message
        assert "Different spatial shape" in caplog.records[1].message
        assert conversion_info.spatial_shape is None
        assert conversion_info.spacing is None

        caplog.clear()
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-004", "ses-M000"),
            },
            individual_masks={"brain": BRAIN_MASK},
        )
        with caplog.at_level("WARNING"):
            conversion_info = TensorConversion(dataset).to_tensors(
                conversion_name="individual",
                spatial_checks=["affine", "shape"],
                save_transforms=False,
            )
        assert len(caplog.records) == 1
        assert 'More than one value for "affine"' in caplog.records[0].message
        assert conversion_info.spatial_shape == (3, 3, 3)
        assert conversion_info.spacing is None

    def test_error(self, bids_dir):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-005", "ses-M000"),
            },
        )
        with pytest.raises(
            TensorConversionError, match="An error occurred during conversion."
        ):
            TensorConversion(dataset).to_tensors(
                conversion_name=None,
                save_transforms=False,
                spatial_checks=None,
            )

    def test__init__(self, tmp_path):
        write_json(
            tmp_path / "dataset_description.json",
            {"Name": "", "BIDSVersion": "0.0.0", "DatasetType": "raw"},
        )
        dataset = BidsDataset(
            tmp_path,
            {
                ("sub-000", "ses-M000"),
            },
        )
        converter = TensorConversion(dataset)
        json = read_json(
            tmp_path / "derivatives" / "tensors" / "dataset_description.json"
        )
        assert "BIDSVersion" in json
        assert json["DatasetType"] == "derivative"
        assert json["Name"] == "Conversions to tensors"
        with pytest.raises(
            ValueError, match="If you pass a conversion name, it cannot be 'raw'."
        ):
            converter.to_tensors(
                conversion_name="raw",
                save_transforms=False,
                spatial_checks=None,
            )

    def test_overwrite(self, bids_dir):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
        )
        TensorConversion(dataset).to_tensors(
            conversion_name="abc",
            save_transforms=False,
            spatial_checks=None,
            description="x",
        )

        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-001", "ses-M001"),
            },
        )
        TensorConversion(dataset).to_tensors(
            conversion_name="abc",
            save_transforms=False,
            spatial_checks=None,
            description="y",
            overwrite=True,
        )

        df = pd.read_csv(
            bids_dir / "derivatives" / "tensors" / "conversions.tsv", sep="\t"
        )
        assert df.iloc[-1]["description"] == "y"
        assert (
            read_json(
                bids_dir
                / "derivatives"
                / "tensors"
                / "src-T1w_conv-abc_description.json"
            )["Description"]
            == "y"
        )
        assert not (
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-000"
            / "ses-M000"
            / "tensors"
            / "sub-000_ses-M000_src-T1w_conv-abc_tensors.pt"
        ).exists()

        assert (
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-001"
            / "ses-M001"
            / "tensors"
            / "sub-001_ses-M001_src-T1w_conv-abc_tensors.pt"
        ).exists()

    def test_merge_conversions_control(self, bids_dir):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
        )
        converter = TensorConversion(dataset)
        converter.to_tensors(
            conversion_name=None, spatial_checks=None, save_transforms=False
        )

        dataset.sub_ses = {
            ("sub-001", "ses-M001"),
        }
        tensor_description = converter.to_tensors(
            conversion_name=None, spatial_checks=None, save_transforms=False
        )
        assert tensor_description.spatial_shape == (3, 3, 3)
        assert tensor_description.spacing == (1.5, 1.5, 1.5)

        pd.testing.assert_frame_equal(
            pd.read_csv(
                bids_dir
                / "derivatives"
                / "tensors"
                / "src-T1w_conv-raw_participantsXsessions.tsv",
                sep="\t",
            ),
            pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-001"],
                    "session_id": ["ses-M000", "ses-M001"],
                }
            ),
        )
        assert (
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-000"
            / "ses-M000"
            / "tensors"
            / "sub-000_ses-M000_src-T1w_conv-raw_tensors.pt"
        ).exists()

        assert (
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-001"
            / "ses-M001"
            / "tensors"
            / "sub-001_ses-M001_src-T1w_conv-raw_tensors.pt"
        ).exists()

    def test_merge_conversions_with_transform(self, bids_dir):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
            individual_masks={"brain": BRAIN_MASK},
            common_masks={"left_hippo": LEFT_HIPPO},
            transforms=TransformsHandler(
                image_transforms=[
                    CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
                ]
            ),
        )
        converter = TensorConversion(dataset)
        converter.to_tensors(
            conversion_name="transform", spatial_checks=None, save_transforms=True
        )

        dataset.sub_ses = {("sub-001", "ses-M001")}
        converter.to_tensors(
            conversion_name="transform", spatial_checks=None, save_transforms=True
        )

        assert (
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-000"
            / "ses-M000"
            / "tensors"
            / "sub-000_ses-M000_src-T1w_conv-transform_tensors.pt"
        ).exists()

        tensor = TensorContent.load(
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-001"
            / "ses-M001"
            / "tensors"
            / "sub-001_ses-M001_src-T1w_conv-transform_tensors.pt"
        )
        assert tensor.images["image"].spatial_shape == (2, 2, 2)
        assert set(tensor.masks.keys()) == {"brain", "left_hippo"}

    def test_merge_conversions_spatial_checks(self, bids_dir, caplog):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
        )
        converter = TensorConversion(dataset)
        converter.to_tensors(
            conversion_name=None, spatial_checks=None, save_transforms=False
        )
        dataset.sub_ses = {("sub-006", "ses-M000")}
        with caplog.at_level("WARNING"):
            tensor_conversion = converter.to_tensors(
                conversion_name=None,
                spatial_checks=["global_shape", "global_spacing"],
                save_transforms=True,
            )
        assert len(caplog.records) == 1
        assert "Different spatial shape" in caplog.records[0].message
        assert tensor_conversion.spatial_shape is None
        assert tensor_conversion.spacing == (1.5, 1.5, 1.5)

        caplog.clear()
        dataset.sub_ses = {("sub-002", "ses-M000")}
        with caplog.at_level("WARNING"):
            tensor_conversion = converter.to_tensors(
                conversion_name=None,
                spatial_checks=["global_shape", "global_spacing"],
                save_transforms=True,
            )
        assert "Different voxel spacing" in caplog.records[0].message
        assert tensor_conversion.spacing is None

    def test_merge_conversion_errors(self, bids_dir):
        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
            individual_masks={"brain": BRAIN_MASK, "seg": SEG_MASK},
            common_masks={"left_hippo": LEFT_HIPPO},
            transforms=TransformsHandler(
                image_transforms=[
                    CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
                    tio.Clamp(out_max=10),
                ]
            ),
        )
        converter = TensorConversion(dataset)
        converter.to_tensors(
            conversion_name="t1",
            save_transforms=True,
            spatial_checks=None,
        )

        dataset.sub_ses = {
            ("sub-001", "ses-M001"),
        }
        file_type = copy(T1_FILE_TYPE)
        file_type.data_type = "pet"
        dataset.image = Image(Bids(bids_dir), file_type)
        with pytest.raises(
            TensorConversionError,
        ) as excinfo:
            converter.to_tensors(
                conversion_name="t1",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert re.compile(
            "The file type of the previous dataset does not match the current file type."
        ).match(str(inner))

        dataset.image = Image(Bids(bids_dir / "derivatives" / "caps"), T1_FILE_TYPE)
        with pytest.raises(
            TensorConversionError,
        ) as excinfo:
            converter.to_tensors(
                conversion_name="t1",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert re.compile(
            "The path to the BIDS of the previous dataset does not match the current path."
        ).match(str(inner))

        dataset.image = Image(Bids(bids_dir), T1_FILE_TYPE)
        dataset.individual_masks = {"brain": BRAIN_MASK, "seg": BRAIN_MASK}
        with pytest.raises(
            TensorConversionError,
        ) as excinfo:
            converter.to_tensors(
                conversion_name="t1",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert re.compile(
            "The file type of the mask 'seg' in the previous dataset does not match the current file type."
        ).match(str(inner))

        dataset.individual_masks = {"brain": BRAIN_MASK}
        with pytest.raises(
            TensorConversionError,
            match=f"{bids_dir / 'derivatives' / 'tensors' / 'src-T1w_conv-t1_description.json'} already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.",
        ) as excinfo:
            converter.to_tensors(
                conversion_name="t1",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert re.compile(
            "The masks in the previous dataset were: {.*}. The masks in the current one are: {.*}."
        ).match(str(inner))

        dataset.common_masks = {"left_hippo": LEFT_HIPPO, "seg": LEFT_HIPPO}
        with pytest.raises(TensorConversionError) as excinfo:
            converter.to_tensors(
                conversion_name="t1",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert (
            str(inner)
            == "Previously, mask 'seg' was a subject-specific mask, currently it is a common mask."
        )

        dataset.individual_masks = {"brain": BRAIN_MASK, "seg": SEG_MASK}
        dataset.common_masks = {"left_hippo": RIGHT_HIPPO}
        with pytest.raises(TensorConversionError) as excinfo:
            converter.to_tensors(
                conversion_name="t1",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert (
            str(inner)
            == f"Previously, mask 'left_hippo' was in {LEFT_HIPPO_PATH}, currently it is in {RIGHT_HIPPO_PATH}."
        )

        dataset.common_masks = {"left_hippo": LEFT_HIPPO}
        with pytest.raises(TensorConversionError) as excinfo:
            converter.to_tensors(
                conversion_name="t1",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert (
            "Custom transforms have been used in the previous dataset, e.g.: 'Clamp(out_min=None, out_max=10)'."
            in str(inner)
        )

        dataset = BidsDataset(
            bids_dir,
            {
                ("sub-000", "ses-M000"),
            },
            transforms=TransformsHandler(
                image_transforms=[
                    CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
                ]
            ),
        )
        converter = TensorConversion(dataset)
        converter.to_tensors(
            conversion_name="t1bis",
            save_transforms=True,
            spatial_checks=None,
        )

        dataset.sub_ses = {
            ("sub-001", "ses-M001"),
        }
        dataset.transforms = TransformsHandler(
            image_transforms=[
                CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
                tio.Clamp(out_max=10),
            ]
        )
        with pytest.raises(TensorConversionError) as excinfo:
            converter.to_tensors(
                conversion_name="t1bis",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert (
            "Custom transforms are used in the current dataset, e.g.: 'Clamp(out_min=None, out_max=10)'."
            in str(inner)
        )

        dataset.transforms = TransformsHandler(
            image_transforms=[
                CropConfig(cropping=(0, 1, 0, 1, 0, 0)),
            ]
        )
        with pytest.raises(TensorConversionError) as excinfo:
            converter.to_tensors(
                conversion_name="t1bis",
                save_transforms=True,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert (
            "The image transforms previously applied don't match with those passed to the current dataset."
            in str(inner)
        )

        with pytest.raises(TensorConversionError) as excinfo:
            converter.to_tensors(
                conversion_name="t1bis",
                save_transforms=False,
                spatial_checks=None,
            )
        inner = excinfo.value.__cause__
        assert (
            str(inner)
            == "'save_transforms' is set to False, but some transforms have already been saved in the previous tensor files."
        )

        converter.to_tensors(
            conversion_name="t1bis",
            save_transforms=True,
            spatial_checks=None,
            check_transforms=False,
        )

        assert (
            bids_dir
            / "derivatives"
            / "tensors"
            / "sub-001"
            / "ses-M001"
            / "tensors"
            / "sub-001_ses-M001_src-T1w_conv-t1bis_tensors.pt"
        ).exists()
