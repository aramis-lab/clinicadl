import json
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.data.structures import DataPoint
from clinicadl.data.tensor_conversion import TensorConversion
from clinicadl.transforms import Transforms
from clinicadl.transforms.config import get_transform_config
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLTensorConversionError,
)

caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
full_data = pd.read_csv(caps_dir / "tsv" / "labels.tsv", sep="\t")
tmp_dir = Path(__file__).parents[1] / "resources" / "caps_tmp"


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


def sub_data(participants_sessions: list[tuple[str, str]]) -> pd.DataFrame:
    data = full_data.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


def delete_files(suffix):
    for root, _, files in os.walk(tmp_dir):
        for file in files:
            if file.endswith(suffix):
                os.remove(os.path.join(root, file))


def remove_empty_dirs():
    for path, _, _ in list(os.walk(tmp_dir))[::-1]:
        if len(os.listdir(path)) == 0:
            shutil.rmtree(path)


def copy_caps():
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    shutil.copytree(caps_dir, tmp_dir)


def copy_caps_without_tensors():
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    shutil.copytree(caps_dir, tmp_dir)
    delete_files(".pt")
    delete_files(".json")
    remove_empty_dirs()


def test_convert_and_read():
    copy_caps()

    sub_ses = [
        ("sub-100", "ses-M000"),
    ]
    data = sub_data(sub_ses)
    preprocessing = PETLinear(
        tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=False
    )
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors(conversion_name="pet_tmp")
    converter.read_conversion(conversion_name="pet_tmp")

    converter.convert_to_tensors()
    converter.read_conversion()

    shutil.rmtree(tmp_dir)


def test_read_conversion():
    sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-000", "ses-M003"),
        ("sub-010", "ses-M003"),
        ("sub-010", "ses-M012"),
    ]
    data = sub_data(sub_ses)
    preprocessing = PETLinear(
        tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
    )

    # control
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion()
    assert str(converter.json) == str(
        caps_dir / "tensor_conversion" / "default_pet-linear_18FAV45_pons2.json"
    )
    assert converter.tensor_folder_name == "default"
    info = converter.get_info()
    assert info.preprocessing == preprocessing
    assert info.individual_masks == []
    assert info.common_masks == []
    assert info.transforms == []
    assert info.spacing is None
    assert info.shape == (1, 1, 1)
    assert sorted(info.participants_sessions) == sorted(sub_ses)

    # test tensor path and json name
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion("default_pet-linear_18FAV45_pons2")
    assert str(converter.json) == str(
        caps_dir / "tensor_conversion" / "default_pet-linear_18FAV45_pons2.json"
    )
    assert converter.tensor_folder_name == "default"
    converter.read_conversion("pet_small", check_pt_files=False)
    assert str(converter.json) == str(caps_dir / "tensor_conversion" / "pet_small.json")
    assert converter.tensor_folder_name == "pet_small"

    # check pt files
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=sub_data([("sub-010", "ses-M012")]),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        FileNotFoundError,
        match="Tensor conversion was performed, as suggested by the presence of*",
    ):
        converter.read_conversion(conversion_name="t1_missing_session")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=sub_data([("sub-000", "ses-M000")]),
        masks=["leftHemisphere.nii"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        FileNotFoundError,
        match="Tensor conversion was performed, as suggested by the presence of*",
    ):
        converter.read_conversion(conversion_name="t1_missing_mask")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=sub_data([("sub-000", "ses-M000")]),
        masks=["leftHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(conversion_name="t1_masks")

    # check json
    with pytest.raises(FileNotFoundError):
        converter.read_conversion(conversion_name="abc")
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".* is not a valid tensor conversion file. Value for 'transforms'*",
    ):
        converter.read_conversion(conversion_name="pet_ref_corrupted")
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".* is not a valid tensor conversion file. Some values have been corrupted*",
    ):
        converter.read_conversion(conversion_name="pet_ref_corrupted_bis")
    with pytest.raises(
        ClinicaDLArgumentError,
        match=r".* is not a valid json file for TensorConversionInfo. A valid file should contain the keys*",
    ):
        converter.read_conversion(conversion_name="pet_ref_missing_field")

    # check preprocessing
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match="The preprocessing of the old conversion does not match the current preprocessing.*",
    ):
        converter.read_conversion("default_pet-linear_18FAV45_pons2")

    # check label
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        label="brain",
        masks=["brain"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match="Some image-specific masks have not been converted*",
    ):
        converter.read_conversion()
    converter.read_conversion(conversion_name="pet_masks", check_pt_files=False)

    # check masks
    caps_dataset = CapsDataset(
        caps_dir, preprocessing=preprocessing, data=data, masks=["brain", "seg"]
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(conversion_name="pet_masks", check_pt_files=False)
    assert set(converter.get_info().individual_masks) == set(["brain", "seg"])

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["leftHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(conversion_name="pet_masks", check_pt_files=False)
    assert converter.get_info().common_masks == ["leftHippocampus.nii.gz"]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["brain", "seg", "leftHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(conversion_name="pet_masks", check_pt_files=False)
    assert set(converter.get_info().individual_masks) == set(["brain", "seg"])
    assert converter.get_info().common_masks == ["leftHippocampus.nii.gz"]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["brain", "seg", "hippocampus"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match="Some image-specific masks have not been converted*",
    ):
        converter.read_conversion(conversion_name="pet_masks")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["leftHippocampus.nii.gz", "rightHemisphere.nii"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError, match="Some masks have not been converted*"
    ):
        converter.read_conversion(conversion_name="pet_masks")

    # check also
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(
            image_transforms=[CustomTransform()],
        ),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match="You asked 'abc' in 'load_also', but no such information was stored during conversion*",
    ):
        converter.read_conversion(
            conversion_name="pet_custom_transform",
            load_also=["other_image", "abc"],
            check_transforms=False,
        )
    converter.read_conversion(
        conversion_name="pet_custom_transform",
        load_also=["other_image", "coefficient"],
        check_transforms=False,
        check_pt_files=False,
    )

    # check transforms
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(
            extraction=Slice(slices=[0]),
            image_transforms=[
                get_transform_config("RescaleIntensity"),
                get_transform_config("Clamp", out_min=-10, out_max=10),
            ],
            sample_transforms=[get_transform_config("Crop", cropping=1)],
            augmentation=[get_transform_config("Pad", padding=1)],
        ),
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion("pet_transform", check_pt_files=False)
    assert len(converter.get_info().transforms) == 2
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match="Custom transforms have been used during the old conversion*",
    ):
        converter.read_conversion(conversion_name="pet_custom_transform")
    converter.read_conversion(
        "pet_custom_transform", check_transforms=False, check_pt_files=False
    )
    assert len(converter.get_info().transforms) == 2

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(
            extraction=Slice(slices=[0]),
            image_transforms=[
                get_transform_config("RescaleIntensity"),
                get_transform_config("ToCanonical"),
            ],
            sample_transforms=[get_transform_config("Crop", cropping=1)],
            augmentation=[get_transform_config("Pad", padding=1)],
        ),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match="The image transforms applied during the old conversion*",
    ):
        converter.read_conversion(
            "pet_transform",
        )

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(
            extraction=Slice(slices=[0]),
            image_transforms=[
                get_transform_config("RescaleIntensity"),
                tio.Clamp(out_min=-10, out_max=10),
            ],
            sample_transforms=[get_transform_config("Crop", cropping=1)],
            augmentation=[get_transform_config("Pad", padding=1)],
        ),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match="Custom transforms have been passed to CapsDataset*",
    ):
        converter.read_conversion(conversion_name="pet_transform")
    converter.read_conversion(conversion_name="default_pet-linear_18FAV45_pons2")
    assert converter.get_info().transforms == []

    # check subject session
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=sub_data(
            [
                ("sub-000", "ses-M000"),
                ("sub-000", "ses-M003"),
                ("sub-010", "ses-M003"),
            ]
        ),
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion()

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=sub_data(
            [
                ("sub-000", "ses-M000"),
                ("sub-000", "ses-M003"),
                ("sub-999", "ses-M999"),
            ]
        ),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r"Some \(participant, session\) have not been converted*",
    ):
        converter.read_conversion("pet_small")


def test_convert_to_tensors():
    copy_caps_without_tensors()

    sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M003"),
    ]
    data = sub_data(sub_ses)
    preprocessing = T1Linear(use_uncropped_image=True)

    # control
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(
            image_transforms=[
                get_transform_config("Crop", cropping=(0, 1, 0, 1, 0, 1)),
                tio.Clamp(out_max=10),
            ]
        ),
        label="seg",
        masks=["brain", "leftHippocampus.nii.gz", "seg"],
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors(
        conversion_name="new_conversion", n_proc=2, save_transforms=True
    )
    with open(tmp_dir / "tensor_conversion" / "new_conversion.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["preprocessing"] == {
        "name": "t1-linear",
        "use_uncropped_image": True,
        "modality": "T1w",
        "file_type": {
            "pattern": "t1_linear/sub-*_ses-*_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii*",
            "description": "T1w images registered to MNI152NLin2009cSym space using Clinica's 't1-linear' pipeline",
            "needed_pipeline": "t1-linear",
        },
    }
    assert set(conversion_info["individual_masks"]) == {"brain", "seg"}
    assert conversion_info["common_masks"] == ["leftHippocampus.nii.gz"]
    assert conversion_info["transforms"] == [
        {
            "name": "Crop",
            "exclude": None,
            "include": None,
            "cropping": [0, 1, 0, 1, 0, 1],
        },
        "Custom transform passed by the user: 'Clamp'",
    ]
    assert np.isclose(conversion_info["spacing"], [1.3, 1.2, 1.1]).all()
    assert conversion_info["shape"] == [2, 2, 2]
    assert not conversion_info["interrupted"]
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
    )

    #       check files
    assert (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "new_conversion"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    tensors: dict = torch.load(
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "new_conversion"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    true_tensors: dict = torch.load(
        caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "t1_transform"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert sorted(list(tensors.keys())) == sorted(["image", "seg", "brain", "affine"])
    for name in tensors.keys():
        assert (tensors[name] == true_tensors[name]).all()

    # test without saving transforms
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        transforms=Transforms(
            image_transforms=[get_transform_config("Crop", cropping=(0, 1, 0, 1, 0, 1))]
        ),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors()
    with open(tmp_dir / "tensor_conversion" / "default_t1-linear.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["transforms"] == []
    tensors = torch.load(
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "default"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert tensors["image"].shape == (1, 3, 3, 3)  # not cropped

    # spacing checked
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M012"),
        ]
    )
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match="An error occurred during conversion.*",
    ):
        converter.convert_to_tensors(conversion_name="check_spacing")
    with open(tmp_dir / "tensor_conversion" / "check_spacing.json", "r") as f:
        conversion_info = json.load(f)
    assert len(conversion_info["participants_sessions"]) == 1
    assert conversion_info["spacing"] is not None
    assert (
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "check_spacing"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file() != (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M012"
        / "t1_linear"
        / "tensors"
        / "check_spacing"
        / "sub-010_ses-M012_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    converter.convert_to_tensors(
        conversion_name="no_check_spacing", ignore_spacing=True
    )
    with open(tmp_dir / "tensor_conversion" / "no_check_spacing.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["spacing"] is None
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M012"],
        ]
    )
    assert (
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "no_check_spacing"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M012"
        / "t1_linear"
        / "tensors"
        / "no_check_spacing"
        / "sub-010_ses-M012_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    # shape warning
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M012"),
        ]
    )
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.warns(match="Different image shapes found in the CAPS dataset:*"):
        converter.convert_to_tensors(conversion_name="check_shape", ignore_spacing=True)
    with open(tmp_dir / "tensor_conversion" / "check_shape.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["shape"] is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.convert_to_tensors(
            conversion_name="not_check_shape", ignore_spacing=True, shape_warning=False
        )

    # check consistency in subjects
    data = sub_data(
        [
            ("sub-000", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        label="seg",
        masks=["seg"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors(conversion_name="subject_consistency")

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        masks=["brain"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors(conversion_name="subject_consistency")

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        masks=["rightHemisphere.nii"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors(conversion_name="subject_consistency")

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        masks=["rightHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors(conversion_name="subject_consistency")
    converter.convert_to_tensors(
        conversion_name="subject_consistency", ignore_spacing=True
    )

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors(conversion_name="subject_consistency_control")

    # custom transform
    sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M003"),
    ]
    data = sub_data(sub_ses)
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(
            image_transforms=[CustomTransform()],
        ),
        label="seg",
        masks=["brain", "seg"],
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors(
        conversion_name="custom_transform", save_transforms=True
    )
    with open(tmp_dir / "tensor_conversion" / "custom_transform.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["individual_masks"]) == sorted(["brain", "seg"])
    assert conversion_info["also"] == {
        "other_image": "image",
        "other_mask": "mask",
        "coefficient": "other",
    }
    tensors: dict = torch.load(
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "custom_transform"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert sorted(tensors.keys()) == sorted(
        [
            "image",
            "seg",
            "brain",
            "coefficient",
            "other_image",
            "other_mask",
            "affine",
        ]
    )
    assert tensors["coefficient"] == 0.5
    assert tensors["other_image"].dtype == torch.float32
    assert tensors["other_mask"].dtype == torch.int32

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(
            image_transforms=[CustomTransform()],
        ),
        label="seg",
        masks=["brain", "seg"],
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors(
        conversion_name="custom_transform_not_saved", save_transforms=False
    )
    with open(
        tmp_dir / "tensor_conversion" / "custom_transform_not_saved.json", "r"
    ) as f:
        conversion_info = json.load(f)
    assert conversion_info["also"] == {}
    assert sorted(
        torch.load(
            tmp_dir
            / "subjects"
            / "sub-000"
            / "ses-M000"
            / "t1_linear"
            / "tensors"
            / "custom_transform_not_saved"
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
            weights_only=True,
        ).keys()
    ) == sorted(
        [
            "image",
            "seg",
            "brain",
            "affine",
        ]
    )

    # check conversion name
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLArgumentError, match="'conversion_name' can't start with default"
    ):
        converter.convert_to_tensors(conversion_name="default_conversion")
    with pytest.raises(
        ClinicaDLArgumentError,
        match="If 'save_transforms' is True, 'conversion_name' cannot be None.",
    ):
        converter.convert_to_tensors(save_transforms=True)

    shutil.rmtree(tmp_dir)


def test_overwrite():
    copy_caps()
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
        ]
    )
    preprocessing = T1Linear(use_uncropped_image=True)
    caps_dataset = CapsDataset(
        tmp_dir, preprocessing=preprocessing, data=data, masks=["brain"]
    )
    converter = TensorConversion(caps_dataset)

    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge*",
    ):
        converter.convert_to_tensors()

    converter.convert_to_tensors(overwrite=True)

    with open(tmp_dir / "tensor_conversion" / "default_t1-linear.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["individual_masks"] == ["brain"]
    tensors = torch.load(
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "default"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert "brain" in tensors

    converter.convert_to_tensors(conversion_name="t1_masks", overwrite=True)

    with open(tmp_dir / "tensor_conversion" / "t1_masks.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["individual_masks"] == ["brain"]
    tensors = torch.load(
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "t1_masks"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert "seg" not in tensors

    shutil.rmtree(tmp_dir)


def test_merge_conversions():
    copy_caps()

    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    preprocessing = T1Linear(use_uncropped_image=True)

    # control
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.convert_to_tensors(conversion_name="t1_ref_interrupted")
    with open(tmp_dir / "tensor_conversion" / "t1_ref_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
    )
    assert np.isclose(conversion_info["spacing"], (1.3, 1.2, 1.1)).all()
    assert conversion_info["shape"] == [3, 3, 3]
    assert (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "t1_ref_interrupted"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    # check preprocessing
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.convert_to_tensors(conversion_name="t1_ref_interrupted")

    # check label
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
        label="brain",
        masks=["seg", "brain"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.convert_to_tensors(conversion_name="t1_ref_interrupted")
    converter.convert_to_tensors(conversion_name="t1_masks_interrupted")

    # check masks
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
        masks=["brain", "seg", "leftHippocampus.nii.gz", "leftHemisphere.nii"],
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors(conversion_name="t1_masks_interrupted")
    with open(tmp_dir / "tensor_conversion" / "t1_masks_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["common_masks"]) == sorted(
        ["leftHippocampus.nii.gz", "rightHippocampus.nii.gz", "leftHemisphere.nii"]
    )
    assert (
        tmp_dir / "masks" / "tensors" / "t1_masks_interrupted" / "leftHemisphere.pt"
    ).is_file()

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
        masks=["seg"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.convert_to_tensors(conversion_name="t1_masks_interrupted")

    # check also
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.convert_to_tensors(
            conversion_name="t1_custom_interrupted",
            save_transforms=True,
            check_transforms=False,
        )

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[CustomTransformBis()]),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.convert_to_tensors(
            conversion_name="t1_custom_interrupted",
            save_transforms=True,
            check_transforms=False,
        )

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[CustomTransform()]),
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors(
        conversion_name="t1_custom_interrupted",
        save_transforms=True,
        check_transforms=False,
    )

    # check transforms
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(
            extraction=Slice(slices=[0]),
            image_transforms=[
                get_transform_config("RescaleIntensity"),
                get_transform_config("Clamp", out_min=-10, out_max=10),
            ],
            sample_transforms=[get_transform_config("Crop", cropping=1)],
            augmentation=[get_transform_config("Pad", padding=1)],
        ),
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors(
        conversion_name="t1_transform_interrupted",
        save_transforms=True,
    )

    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.convert_to_tensors(
            conversion_name="t1_ref_interrupted", save_transforms=True
        )
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.convert_to_tensors(
            conversion_name="t1_transform_interrupted",
            save_transforms=False,
        )

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError,
        match=r".*already exists, so ClinicaDL tried to merge the current tensor conversion with the old one.*",
    ):
        converter.convert_to_tensors(
            conversion_name="t1_transform_interrupted", save_transforms=True
        )

    # spacing and shape
    data = sub_data(
        [
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
        ]
    )
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(
        ClinicaDLTensorConversionError, match="An error occurred during conversion.*"
    ):
        converter.convert_to_tensors(conversion_name="t1_ref_interrupted")
    with pytest.warns(match="Different image shapes found in the CAPS dataset:*"):
        converter.convert_to_tensors(
            conversion_name="t1_ref_interrupted", ignore_spacing=True
        )
    with open(tmp_dir / "tensor_conversion" / "t1_ref_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
            ["sub-010", "ses-M012"],
        ]
    )
    assert conversion_info["spacing"] is None
    assert conversion_info["shape"] is None

    # without shape warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.convert_to_tensors(
            conversion_name="t1_ref_interrupted",
            ignore_spacing=True,
            shape_warning=False,
        )

    shutil.rmtree(tmp_dir)
