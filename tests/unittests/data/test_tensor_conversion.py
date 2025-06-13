import json
import os
import shutil
import warnings
from copy import deepcopy
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


class CustomTransform:
    def __call__(self, datapoint: DataPoint) -> DataPoint:
        transformed = deepcopy(datapoint)
        transformed.add_image(datapoint.image, "other_image")
        transformed.add_mask(
            tio.LabelMap(
                tensor=torch.ones_like(datapoint.image.tensor),
                affine=datapoint.image.affine,
            ),
            "other_mask",
        )
        transformed["age"] = 55

        return transformed


class CustomTransformBis:
    def __call__(self, datapoint: DataPoint) -> DataPoint:
        transformed = deepcopy(datapoint)
        transformed.add_image(datapoint.image, "other_image")
        transformed.add_mask(
            tio.ScalarImage(
                tensor=torch.ones_like(datapoint.image.tensor),
                affine=datapoint.image.affine,
            ),
            "other_mask",
        )
        transformed["age"] = 55

        return transformed


def sub_data(participants_sessions: list[tuple[str, str]]) -> pd.DataFrame:
    data = full_data.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


def delete_pt_files(dir_path: Path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".pt"):
                os.remove(os.path.join(root, file))


def copy_json_dir(tmp_dir: Path):
    if (tmp_dir / "tensor_conversion").is_dir():
        shutil.rmtree(tmp_dir / "tensor_conversion")
    shutil.copytree(caps_dir / "tensor_conversion", tmp_dir / "tensor_conversion")
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted_bis.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_missing_field.json").unlink()


def copy_dir(tmp_dir: Path):
    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    shutil.copytree(caps_dir, tmp_dir)
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted_bis.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_missing_field.json").unlink()


def test_convert_and_read():
    copy_dir(tmp_dir)

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
    converter.convert_to_tensors("pet_tmp")
    converter.read_conversion("pet_tmp")

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
    converter.read_conversion("pet_ref")
    info = converter.get_info()
    assert info.preprocessing == preprocessing
    assert info.individual_masks == []
    assert info.common_masks == []
    assert info.transforms == []
    assert info.spacing == (1.3, 1.2, 1.1)
    assert info.shape == (1, 1, 1)
    assert sorted(info.participants_sessions) == sorted(sub_ses)

    # check json
    with pytest.raises(FileNotFoundError):
        converter.read_conversion("abc")
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_ref_corrupted")
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_ref_corrupted_bis")
    with pytest.raises(ClinicaDLArgumentError):
        converter.read_conversion("pet_ref_missing_field")

    # check preprocessing
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_ref")

    # check label
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        label="brain",
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_ref")
    converter.read_conversion("pet_masks")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        label="brain",
        masks=["brain"],
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion("pet_masks")

    # check masks
    caps_dataset = CapsDataset(
        caps_dir, preprocessing=preprocessing, data=data, masks=["brain", "seg"]
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion("pet_masks")
    assert set(converter.get_info().individual_masks) == set(["brain", "seg"])

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["leftHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion("pet_masks")
    assert converter.get_info().common_masks == ["leftHippocampus.nii.gz"]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["brain", "seg", "leftHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion("pet_masks")
    assert set(converter.get_info().individual_masks) == set(["brain", "seg"])
    assert converter.get_info().common_masks == ["leftHippocampus.nii.gz"]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["brain", "seg", "hippocampus"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_masks")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["leftHippocampus.nii.gz", "rightHemisphere.nii"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_masks")

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
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(
            "pet_custom_transform",
            load_also=["other_image", "sex"],
            check_transforms=False,
        )
    converter.read_conversion(
        "pet_custom_transform", load_also=["other_image", "age"], check_transforms=False
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
    converter.read_conversion(
        "pet_transform",
    )
    assert len(converter.get_info().transforms) == 2
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_custom_transform")
    converter.read_conversion(
        "pet_custom_transform",
        check_transforms=False,
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
    with pytest.raises(ClinicaDLTensorConversionError):
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
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_transform")
    converter.read_conversion("pet_ref")
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
    converter.read_conversion("pet_ref")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=sub_data(
            [
                ("sub-000", "ses-M000"),
                ("sub-000", "ses-M003"),
                ("sub-010", "ses-M003"),
                ("sub-999", "ses-M999"),
            ]
        ),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion("pet_ref")


def test_convert_to_tensors():
    copy_dir(tmp_dir)
    delete_pt_files(tmp_dir)

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
        masks=["brain", "leftHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors("new_conversion", n_proc=2)
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
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    #       check old conversion updates
    with open(tmp_dir / "tensor_conversion" / "pet_masks.json", "r") as f:
        old_conversion_info = json.load(f)
    assert old_conversion_info["common_masks"] == ["rightHippocampus.nii.gz"]
    assert sorted(
        old_conversion_info["participants_sessions"]
    ) == sorted(  # not modified because not the same preprocessing
        [
            ["sub-000", "ses-M000"],
            ["sub-000", "ses-M003"],
            ["sub-010", "ses-M003"],
            ["sub-010", "ses-M012"],
        ]
    )

    tensors: dict = torch.load(
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
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
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert sorted(list(tensors.keys())) == sorted(["image", "seg", "brain", "affine"])
    for name in tensors.keys():
        assert (tensors[name] == true_tensors[name]).all()

    # test old json update
    data = sub_data(sub_ses)
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=data,
        label="age",
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors("new_conversion_pet")
    with open(tmp_dir / "tensor_conversion" / "new_conversion_pet.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["individual_masks"] == []
    assert conversion_info["transforms"] == []
    with open(tmp_dir / "tensor_conversion" / "pet_ref.json", "r") as f:
        old_conversion_info = json.load(f)
    assert sorted(old_conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M003"],
            ["sub-010", "ses-M012"],
        ]
    )

    # test save transforms
    delete_pt_files(tmp_dir)
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        transforms=Transforms(
            image_transforms=[get_transform_config("Crop", cropping=(0, 1, 0, 1, 0, 1))]
        ),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors("no_transforms", save_transforms=False)
    with open(tmp_dir / "tensor_conversion" / "no_transforms.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["transforms"] == []
    tensors = torch.load(
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert tensors["image"].shape == (1, 3, 3, 3)  # not cropped

    # spacing checked
    delete_pt_files(tmp_dir)
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
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors("check_spacing")
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
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file() != (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M012"
        / "t1_linear"
        / "tensors"
        / "sub-010_ses-M012_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    delete_pt_files(tmp_dir)
    converter.convert_to_tensors("not_check_spacing", ignore_spacing=True)
    with open(tmp_dir / "tensor_conversion" / "not_check_spacing.json", "r") as f:
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
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M012"
        / "t1_linear"
        / "tensors"
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
        converter.convert_to_tensors("check_shape", ignore_spacing=True)
    with open(tmp_dir / "tensor_conversion" / "check_shape.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["shape"] is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.convert_to_tensors(
            "not_check_shape", ignore_spacing=True, raise_warnings=False
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
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors("subject_consistency")

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        masks=["brain"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors("subject_consistency")

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        masks=["rightHemisphere.nii"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors("subject_consistency")

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        masks=["rightHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors("subject_consistency")
    converter.convert_to_tensors("subject_consistency", ignore_spacing=True)

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors("subject_consistency_control")

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
        masks=["brain"],
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors("custom_transform")
    with open(tmp_dir / "tensor_conversion" / "custom_transform.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["individual_masks"]) == sorted(["brain", "seg"])
    assert conversion_info["also"] == {
        "other_image": "image",
        "other_mask": "mask",
        "age": "other",
    }
    tensors: dict = torch.load(
        tmp_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    assert sorted(tensors.keys()) == sorted(
        [
            "image",
            "seg",
            "brain",
            "age",
            "other_image",
            "other_mask",
            "affine",
        ]
    )
    assert tensors["age"] == 55
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
        masks=["brain"],
    )
    converter = TensorConversion(caps_dataset)
    converter.convert_to_tensors("custom_transform_not_saved", save_transforms=False)
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

    shutil.rmtree(tmp_dir)


def test_merge_conversions():
    copy_dir(tmp_dir)
    (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).unlink()

    sub_ses = [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M003"),
    ]
    data = sub_data(sub_ses)
    preprocessing = T1Linear(use_uncropped_image=True)

    # control
    shutil.rmtree(tmp_dir / "tensor_conversion")
    (tmp_dir / "tensor_conversion").mkdir()
    shutil.copy(
        caps_dir / "tensor_conversion" / "t1_ref_interrupted.json",
        tmp_dir / "tensor_conversion" / "t1_ref_interrupted.json",
    )
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
    )
    converter = TensorConversion(caps_dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.convert_to_tensors("t1_ref_interrupted")
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
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    assert (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()
    (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).unlink()

    # check preprocessing
    data = sub_data(sub_ses)
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=data,
        transforms=Transforms(image_transforms=[]),
    )
    converter = TensorConversion(caps_dataset)
    copy_json_dir(tmp_dir)
    with pytest.raises(ClinicaDLArgumentError):
        converter.convert_to_tensors("t1_ref_interrupted")

    # check label
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
        label="brain",
        masks=["seg"],
    )
    converter = TensorConversion(caps_dataset)
    copy_json_dir(tmp_dir)
    with pytest.raises(ClinicaDLArgumentError):
        converter.convert_to_tensors("t1_ref_interrupted")
    converter.convert_to_tensors("t1_masks_interrupted")
    with open(tmp_dir / "tensor_conversion" / "t1_masks_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
    )

    # check masks
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
        masks=["brain", "seg", "leftHippocampus.nii.gz", "leftHemisphere.nii"],
    )
    converter = TensorConversion(caps_dataset)
    copy_json_dir(tmp_dir)
    converter.convert_to_tensors("t1_masks_interrupted")
    with open(tmp_dir / "tensor_conversion" / "t1_masks_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["common_masks"]) == sorted(
        ["leftHippocampus.nii.gz", "rightHippocampus.nii.gz", "leftHemisphere.nii"]
    )
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
    )
    assert (tmp_dir / "masks" / "tensors" / "leftHippocampus.pt").is_file()
    assert (tmp_dir / "masks" / "tensors" / "leftHemisphere.pt").is_file()

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
        masks=["seg"],
    )
    converter = TensorConversion(caps_dataset)
    copy_json_dir(tmp_dir)
    with pytest.raises(ClinicaDLArgumentError):
        converter.convert_to_tensors("t1_masks_interrupted")

    # check also
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[CustomTransformBis()]),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLArgumentError):
        converter.convert_to_tensors(
            "t1_custom_interrupted",
            check_transforms=False,
        )

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLArgumentError):
        converter.convert_to_tensors(
            "t1_custom_interrupted",
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
        "t1_custom_interrupted",
        check_transforms=False,
    )
    with open(tmp_dir / "tensor_conversion" / "t1_custom_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
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
    copy_json_dir(tmp_dir)
    converter.convert_to_tensors(
        "t1_transform_interrupted",
    )
    with open(
        tmp_dir / "tensor_conversion" / "t1_transform_interrupted.json", "r"
    ) as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
    )

    copy_json_dir(tmp_dir)
    with pytest.raises(ClinicaDLArgumentError):
        converter.convert_to_tensors("t1_ref_interrupted")
    with pytest.raises(ClinicaDLArgumentError):
        converter.convert_to_tensors(
            "t1_transform_interrupted",
            save_transforms=False,
        )
    converter.convert_to_tensors(
        "t1_ref_interrupted",
        save_transforms=False,
    )
    with open(tmp_dir / "tensor_conversion" / "t1_ref_interrupted.json", "r") as f:
        conversion_info = json.load(f)
    assert sorted(conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M000"],
            ["sub-010", "ses-M003"],
        ]
    )

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=preprocessing,
        data=data,
        transforms=Transforms(image_transforms=[]),
    )
    converter = TensorConversion(caps_dataset)
    copy_json_dir(tmp_dir)
    with pytest.raises(ClinicaDLArgumentError):
        converter.convert_to_tensors(
            "t1_transform_interrupted",
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
    copy_json_dir(tmp_dir)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.convert_to_tensors("t1_ref_interrupted")
    copy_json_dir(tmp_dir)
    with pytest.warns(match="Different image shapes found in the CAPS dataset:*"):
        converter.convert_to_tensors("t1_ref_interrupted", ignore_spacing=True)
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
    assert (
        tmp_dir
        / "subjects"
        / "sub-010"
        / "ses-M012"
        / "t1_linear"
        / "tensors"
        / "sub-010_ses-M012_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    ).is_file()

    # without shape warning
    copy_json_dir(tmp_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.convert_to_tensors(
            "t1_ref_interrupted", ignore_spacing=True, raise_warnings=False
        )

    # no checks in old conversion
    shutil.rmtree(tmp_dir / "tensor_conversion")
    (tmp_dir / "tensor_conversion").mkdir()
    shutil.copy(
        caps_dir / "tensor_conversion" / "t1_ref_interrupted_no_checks.json",
        tmp_dir / "tensor_conversion" / "t1_ref_interrupted_no_checks.json",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.convert_to_tensors("t1_ref_interrupted_no_checks")

    shutil.rmtree(tmp_dir)
