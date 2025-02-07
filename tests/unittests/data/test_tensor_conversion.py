import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data import CapsDataset
from clinicadl.data.datatype import PETLinear, T1Linear
from clinicadl.data.tensor_conversion import TensorConversion
from clinicadl.transforms import Slice, Transforms, get_transform_config
from clinicadl.utils.exceptions import ClinicaDLTensorConversionError

caps_dir = Path(__file__).parents[1] / "resources" / "caps_example"
full_data = pd.read_csv(caps_dir / "labels.tsv", sep="\t")


def sub_data(participants_sessions: list[tuple[str, str]]) -> pd.DataFrame:
    data = full_data.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


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
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_ref.json")
    info = converter.get_info()
    assert info.preprocessing == preprocessing
    assert info.label is None
    assert info.individual_masks == []
    assert info.common_masks == []
    assert info.transforms is None
    assert info.spacing == (1.3, 1.2, 1.1)
    assert info.shape == (1, 1, 1)
    assert info.participants_sessions == sub_ses

    # check json
    with pytest.raises(FileNotFoundError):
        converter.read_conversion(caps_dir / "tensor_conversion" / "abc.json")
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(
            caps_dir / "tensor_conversion" / "pet_ref_corrupted.json"
        )
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(
            caps_dir / "tensor_conversion" / "pet_ref_corrupted_bis.json"
        )
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(
            caps_dir / "tensor_conversion" / "pet_ref_missing_field.json"
        )

    # check preprocessing
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(caps_dir / "tensor_conversion" / "pet_ref.json")

    # check label
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        label="brain",
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(caps_dir / "tensor_conversion" / "pet_ref.json")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        label="age",
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(caps_dir / "tensor_conversion" / "pet_ref.json")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        label="brain",
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_mask_label.json")
    assert converter.get_info().label == "Mask('brain')"

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        label="age",
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_age_label.json")
    assert converter.get_info().label == "Column('age')"

    # check masks
    caps_dataset = CapsDataset(
        caps_dir, preprocessing=preprocessing, data=data, masks=["brain", "seg"]
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_masks.json")
    assert converter.get_info().individual_masks == ["brain", "seg"]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["leftHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_masks.json")
    assert converter.get_info().common_masks == ["leftHippocampus.nii.gz"]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["brain", "seg", "leftHippocampus.nii.gz"],
    )
    converter = TensorConversion(caps_dataset)
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_masks.json")
    assert converter.get_info().individual_masks == ["brain", "seg"]
    assert converter.get_info().common_masks == ["leftHippocampus.nii.gz"]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["brain", "seg", "hippocampus"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(caps_dir / "tensor_conversion" / "pet_masks.json")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=preprocessing,
        data=data,
        masks=["leftHippocampus.nii.gz", "rightHemisphere.nii"],
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(caps_dir / "tensor_conversion" / "pet_masks.json")

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
        caps_dir / "tensor_conversion" / "pet_transform.json",
    )
    assert len(converter.get_info().transforms) == 2
    with pytest.raises(ClinicaDLTensorConversionError):
        converter.read_conversion(
            caps_dir / "tensor_conversion" / "pet_custom_transform.json"
        )
    converter.read_conversion(
        caps_dir / "tensor_conversion" / "pet_custom_transform.json",
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
            caps_dir / "tensor_conversion" / "pet_transform.json",
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
        converter.read_conversion(caps_dir / "tensor_conversion" / "pet_transform.json")
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_no_transform.json")
    assert converter.get_info().transforms is None
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_ref.json")
    assert converter.get_info().transforms is None

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
    converter.read_conversion(caps_dir / "tensor_conversion" / "pet_ref.json")

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
        converter.read_conversion(caps_dir / "tensor_conversion" / "pet_ref.json")


def test_convert_to_tensors():
    tmp_dir = Path(__file__).parents[1] / "resources" / "caps_tmp"

    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    shutil.copytree(caps_dir, tmp_dir)
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted_bis.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_missing_field.json").unlink()

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
            image_transforms=[get_transform_config("Crop", cropping=(0, 1, 0, 1, 0, 1))]
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
    assert conversion_info["label"] == "Mask('seg')"
    assert conversion_info["individual_masks"] == ["brain"]
    assert conversion_info["common_masks"] == ["leftHippocampus.nii.gz"]
    assert conversion_info["transforms"] == [
        {
            "name": "Crop",
            "cropping": [0, 1, 0, 1, 0, 1],
        }
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
    assert sorted(list(tensors.keys())) == sorted(list(true_tensors.keys()))
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
    assert conversion_info["label"] == "Column('age')"
    with open(tmp_dir / "tensor_conversion" / "pet_ref.json", "r") as f:
        old_conversion_info = json.load(f)
    assert sorted(old_conversion_info["participants_sessions"]) == sorted(
        [
            ["sub-000", "ses-M003"],
            ["sub-010", "ses-M012"],
        ]
    )

    # test save transforms
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
    assert old_conversion_info["label"] is None
    assert conversion_info["transforms"] is None
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
    assert conversion_info["participants_sessions"] == [["sub-000", "ses-M000"]]
    converter.convert_to_tensors("not_check_spacing", ignore_spacing=True)
    with open(tmp_dir / "tensor_conversion" / "not_check_spacing.json", "r") as f:
        conversion_info = json.load(f)
    assert conversion_info["spacing"] is None
    assert conversion_info["participants_sessions"] == [
        ["sub-000", "ses-M000"],
        ["sub-010", "ses-M012"],
    ]

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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converter.convert_to_tensors(
            "not_check_shape", ignore_spacing=True, raise_warnings=False
        )

    # check json
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
    )
    converter = TensorConversion(caps_dataset)
    with pytest.raises(FileExistsError):
        converter.convert_to_tensors("not_check_shape")
    converter.convert_to_tensors("not_check_shape_")

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

    shutil.rmtree(tmp_dir)
