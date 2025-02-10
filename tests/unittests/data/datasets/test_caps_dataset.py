import shutil
from pathlib import Path

import pandas as pd
import pytest
import torchio as tio

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatype.preprocessing import PETLinear, T1Linear
from clinicadl.data.structures import Mask
from clinicadl.transforms import Patch, Slice, Transforms, get_transform_config
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLTSVError,
)

caps_dir = Path(__file__).parents[2] / "resources" / "caps_example"
full_data = pd.read_csv(caps_dir / "labels.tsv", sep="\t")


def sub_data(participants_sessions: list[tuple[str, str]]) -> pd.DataFrame:
    data = full_data.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


def test_good_caps_dataset():
    preprocessing = PETLinear(
        tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
    )
    transforms = Transforms(
        extraction=Slice(slices=[0]),
        image_transforms=[tio.RescaleIntensity()],
        sample_transforms=[
            get_transform_config("Pad", padding=1),
            tio.RemapLabels({1: 10}),
        ],
        augmentations=[get_transform_config("Crop", cropping=1)],
    )
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
        ]
    )
    label = "age"
    masks = ["brain"]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing,
        transforms,
        data,
        label,
        masks,
    )
    assert isinstance(caps_dataset.image_transform, tio.Compose)
    assert len(caps_dataset.image_transform.transforms) == 1
    assert isinstance(caps_dataset.image_transform.transforms[0], tio.RescaleIntensity)

    assert isinstance(caps_dataset.sample_transform, tio.Compose)
    assert len(caps_dataset.sample_transform.transforms) == 2
    assert isinstance(caps_dataset.sample_transform.transforms[0], tio.Pad)
    assert isinstance(caps_dataset.sample_transform.transforms[1], tio.RemapLabels)

    assert isinstance(caps_dataset.augmentation, tio.Compose)
    assert len(caps_dataset.augmentation.transforms) == 1
    assert isinstance(caps_dataset.augmentation.transforms[0], tio.Crop)

    assert (caps_dataset.df == data).all().all()
    assert caps_dataset.label == "age"
    assert len(caps_dataset.individual_masks) == 1
    assert isinstance(caps_dataset.individual_masks[0], Mask)
    assert caps_dataset.individual_masks[0].name == "brain"

    assert caps_dataset.common_masks == []
    assert caps_dataset.tensor_conversion.json is None


def test_checks():
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
        ]
    )

    # check preprocessing
    preprocessing = PETLinear(tracer="18FAV45", suvr_reference_region="pons2")
    with pytest.raises(ClinicaDLCAPSError):
        CapsDataset(
            caps_dir,
            preprocessing,
            data,
        )
    preprocessing.use_uncropped_image = True
    CapsDataset(
        caps_dir,
        preprocessing,
        data,
    )

    # check df
    data_path = Path(caps_dir / "only_pets.tsv")
    data.to_csv(data_path, sep="\t", index=False)
    with pytest.raises(FileNotFoundError):
        CapsDataset(
            caps_dir,
            preprocessing,
            caps_dir / "abc.tsv",
        )
    with pytest.raises(ClinicaDLArgumentError):
        CapsDataset(
            caps_dir,
            preprocessing,
            data=[("sub-000", "ses-M000")],
        )

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing,
        caps_dir / "only_pets.tsv",
    )
    assert (caps_dataset.df == data).all().all()
    data_path.unlink()

    caps_dataset = CapsDataset(
        caps_dir,
        T1Linear(use_uncropped_image=True),
        data=None,
    )
    tsv_path = caps_dir / "overview_t1-linear.tsv"
    tsv = pd.read_csv(tsv_path, sep="\t")
    assert (caps_dataset.df == data.drop(columns=["age"])).all().all()
    assert (tsv == data.drop(columns=["age"])).all().all()
    tsv_path.unlink()

    # check label
    with pytest.raises(ClinicaDLArgumentError):
        CapsDataset(caps_dir, T1Linear(use_uncropped_image=True), data=data, label=0)
    caps_dataset = CapsDataset(
        caps_dir, T1Linear(use_uncropped_image=True), data=data, label=None
    )
    assert caps_dataset.label is None
    caps_dataset = CapsDataset(
        caps_dir, T1Linear(use_uncropped_image=True), data=data, label="brain"
    )
    assert isinstance(caps_dataset.label, Mask)
    assert caps_dataset.label.name == "brain"

    # masks
    with pytest.raises(ClinicaDLArgumentError):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            masks="leftHippocampus.nii.gz",
        )
    caps_dataset = CapsDataset(
        caps_dir,
        T1Linear(use_uncropped_image=True),
        data=data,
        masks=["leftHippocampus.nii.gz"],
    )
    assert len(caps_dataset.common_masks) == 1
    assert len(caps_dataset.individual_masks) == 0
    assert isinstance(caps_dataset.common_masks[0], Mask)
    assert (
        caps_dataset.common_masks[0].path
        == caps_dir / "masks" / "leftHippocampus.nii.gz"
    )


def test_get_participant_session_couples():
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        caps_dir,
        T1Linear(use_uncropped_image=True),
        data=data,
    )
    assert caps_dataset.get_participant_session_couples() == [
        ("sub-000", "ses-M000"),
        ("sub-000", "ses-M003"),
    ]


def test_describe():
    tmp_dir = Path(__file__).parents[1] / "resources" / "caps_tmp"

    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    shutil.copytree(caps_dir, tmp_dir)
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted_bis.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_missing_field.json").unlink()

    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
        ]
    )

    caps_dataset = CapsDataset(
        tmp_dir,
        T1Linear(use_uncropped_image=True),
        data=data,
    )
    assert caps_dataset.describe()["total_samples"] == 3

    caps_dataset = CapsDataset(
        tmp_dir,
        T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(extraction=Slice()),
    )
    with pytest.raises(ClinicaDLCAPSError):
        caps_dataset.describe()
    caps_dataset.to_tensors("t1", ignore_spacing=True)
    description = caps_dataset.describe()
    assert description["total_samples"] == 7
    assert description["participant_session_pairs"] == [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M003"),
        ("sub-010", "ses-M012"),
    ]
    assert description["preprocessing"] == {
        "name": "t1-linear",
        "use_uncropped_image": True,
        "modality": "T1w",
        "file_type": {
            "pattern": "t1_linear/sub-*_ses-*_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii*",
            "description": "T1w images registered to MNI152NLin2009cSym space using Clinica's 't1-linear' pipeline",
            "needed_pipeline": "t1-linear",
        },
    }
    assert description["extraction"] == {
        "borders": None,
        "discarded_slices": None,
        "extract_method": "slice",
        "slice_direction": 0,
        "slices": None,
        "squeeze": True,
    }
    shutil.rmtree(tmp_dir)


def test_get_sample_info():
    tmp_dir = Path(__file__).parents[1] / "resources" / "caps_tmp"

    if tmp_dir.is_dir():
        shutil.rmtree(tmp_dir)
    shutil.copytree(caps_dir, tmp_dir)
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_corrupted_bis.json").unlink()
    Path(tmp_dir / "tensor_conversion" / "pet_ref_missing_field.json").unlink()

    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=data,
    )
    assert caps_dataset.get_sample_info(0, "age") == 1.0

    caps_dataset = CapsDataset(
        tmp_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(extraction=Patch(patch_size=2, stride=1)),
    )
    with pytest.raises(ClinicaDLCAPSError):
        caps_dataset.get_sample_info(8, "age")
    caps_dataset.read_tensor_conversion("t1")
    assert caps_dataset.get_sample_info(7, "age") == 1.0
    assert caps_dataset.get_sample_info(8, "age") == 2.0


def test_train_eval():
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
        ]
    )
    caps_dataset = CapsDataset(
        caps_dir,
        PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=data,
    )
    assert not caps_dataset.eval_mode
    caps_dataset.eval()
    assert caps_dataset.eval_mode
    caps_dataset.train()
    assert not caps_dataset.eval_mode


def test_subset():
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        caps_dir,
        PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=data,
    )
    subset = caps_dataset.subset(
        sub_data(
            [
                ("sub-000", "ses-M000"),
                ("sub-000", "ses-M003"),
            ]
        )
    )
    assert isinstance(subset, CapsDataset)
    assert len(subset) == 2

    with pytest.raises(ClinicaDLTSVError):
        caps_dataset.subset(
            sub_data(
                [
                    ("sub-000", "ses-M000"),
                    ("sub-000", "ses-M003"),
                    ("sub-010", "ses-M012"),
                ]
            )
        )
