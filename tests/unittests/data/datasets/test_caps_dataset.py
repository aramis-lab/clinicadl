import shutil
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes.preprocessing import PETLinear, T1Linear
from clinicadl.data.structures import DataPoint, Mask
from clinicadl.transforms import Transforms
from clinicadl.transforms.config import get_transform_config
from clinicadl.transforms.extraction import Patch, Slice
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLTSVError,
)

caps_dir = Path(__file__).parents[2] / "resources" / "caps_example"
full_data = pd.read_csv(caps_dir / "tsv" / "labels.tsv", sep="\t")


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
        transforms=transforms,
        data=data,
        label=label,
        masks=masks,
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
    assert (caps_dataset.df == data[["participant_id", "session_id"]]).all().all()
    assert (tsv == data[["participant_id", "session_id"]]).all().all()
    tsv_path.unlink()

    # check label
    with pytest.raises(ClinicaDLArgumentError):
        CapsDataset(
            caps_dir,
            preprocessing=PETLinear(
                use_uncropped_image=True,
                tracer="18FAV45",
                suvr_reference_region="pons2",
            ),
            data=full_data,
            label="category",
        )

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
    with pytest.raises(ClinicaDLArgumentError):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            masks=["affine"],
        )
    with pytest.raises(ClinicaDLArgumentError):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            masks=["leftHippocampus", "leftHippocampus.nii.gz"],
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
    tmp_dir = Path(__file__).parents[2] / "resources" / "caps_tmp"

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
    caps_dataset.to_tensors("t1_", ignore_spacing=True)
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
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
        data=data,
    )
    assert caps_dataset.get_sample_info(0, "age") == 1.0
    with pytest.raises(KeyError):
        caps_dataset.get_sample_info(0, "abc")
    with pytest.raises(IndexError):
        caps_dataset.get_sample_info(-1, "age")
    with pytest.raises(IndexError):
        caps_dataset.get_sample_info(2, "abc")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        label="seg",
        transforms=Transforms(
            extraction=Patch(patch_size=1, stride=1),
            image_transforms=[
                get_transform_config("Crop", cropping=(0, 1, 0, 1, 0, 1))
            ],
        ),
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
    tmp_dir = Path(__file__).parents[2] / "resources" / "caps_tmp"

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
        T1Linear(use_uncropped_image=True),
        transforms=Transforms(extraction=Slice(slices=[0, 1])),
        data=data,
    )
    caps_dataset.to_tensors("for_subset")
    subset = caps_dataset.subset(
        sub_data(
            [
                ("sub-010", "ses-M003"),
                (
                    "sub-999",
                    "ses-M099",
                ),  # not in the dataset, this shouldn't raise an error
            ]
        )
    )
    assert isinstance(subset, CapsDataset)
    assert len(subset) == 2
    assert subset[0].participant == "sub-010"
    assert subset[0].session == "ses-M003"

    with pytest.raises(
        ClinicaDLCAPSError,
        match=r"No \(participant, session\) pairs mentioned in 'data' are in the CapsDataset. This would lead to an empty dataset!",
    ):
        caps_dataset.subset(
            sub_data(
                [
                    ("sub-999", "ses-M099"),
                ]
            )
        )

    shutil.rmtree(tmp_dir)


def test__getitem__():
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        label="seg",
        masks=["brain"],
        transforms=Transforms(
            extraction=Slice(),
            image_transforms=[
                get_transform_config("Crop", cropping=(0, 1, 0, 1, 0, 1))
            ],
            sample_transforms=[tio.RescaleIntensity(masking_method="brain")],
            augmentations=[tio.RemapLabels({1: 10})],
        ),
    )

    with pytest.raises(ClinicaDLCAPSError):
        caps_dataset[0]

    caps_dataset.read_tensor_conversion("t1")

    tensors = torch.load(
        caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    out_sample = caps_dataset[0]
    assert out_sample.preprocessing == T1Linear(use_uncropped_image=True)
    assert out_sample.slice_position == 0
    assert out_sample.slice_direction == 0
    assert (
        out_sample.image.tensor
        == tio.RescaleIntensity(masking_method="brain")(
            tio.Subject(
                image=tio.ScalarImage(tensor=tensors["image"][:, 0:1]),
                brain=tio.LabelMap(tensor=tensors["brain"][:, 0:1]),
            )
        ).image.tensor[:, 0]
    ).all()
    assert (out_sample.affine == tensors["affine"]).all()
    assert out_sample.participant == "sub-000"
    assert out_sample.session == "ses-M000"
    assert out_sample.image_path == (
        caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    )
    assert (
        out_sample.label.tensor
        == tio.RemapLabels({1: 10})(tio.LabelMap(tensor=tensors["seg"])).tensor[:, 0]
    ).all()

    ###########
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        label="seg",
        masks=["brain", "leftHippocampus.nii.gz"],
        transforms=Transforms(
            extraction=Slice(),
            image_transforms=[tio.Crop((0, 0, 0, 0, 0, 1))],
            sample_transforms=[
                tio.RescaleIntensity(masking_method="brain"),
                tio.Mask(masking_method="leftHippocampus"),
            ],
            augmentations=[tio.RemapLabels({0: 10})],
        ),
    )
    caps_dataset.read_tensor_conversion("t1_without_transform")
    caps_dataset.eval()
    tensors = torch.load(
        caps_dir
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    common_mask = torch.load(
        caps_dir / "masks" / "tensors" / "leftHippocampus.pt",
        weights_only=True,
    )["mask"]
    out_sample = caps_dataset[3]
    assert out_sample.slice_position == 1
    assert out_sample.slice_direction == 0
    assert out_sample.participant == "sub-010"
    assert out_sample.session == "ses-M003"
    assert (
        out_sample.image.tensor
        == tio.Mask(masking_method="leftHippocampus")(
            tio.RescaleIntensity(masking_method="brain")(
                tio.Crop(cropping=(0, 0, 0, 0, 0, 1))(
                    tio.Subject(
                        image=tio.ScalarImage(tensor=tensors["image"][:, 1:2]),
                        brain=tio.LabelMap(tensor=tensors["brain"][:, 1:2]),
                        leftHippocampus=tio.LabelMap(tensor=common_mask[:, 1:2]),
                    )
                )
            )
        ).image.tensor[:, 0]
    ).all()
    assert (out_sample.affine == tensors["affine"]).all()
    assert out_sample.image_path == (
        caps_dir
        / "subjects"
        / "sub-010"
        / "ses-M003"
        / "t1_linear"
        / "tensors"
        / "sub-010_ses-M003_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    )
    assert (
        out_sample.label.tensor
        == tio.Crop(cropping=(0, 0, 0, 0, 0, 1))(
            tio.LabelMap(tensor=tensors["seg"])
        ).tensor[:, 1]
    ).all()

    # other label
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=data,
        label="age",
    )
    caps_dataset.read_tensor_conversion("pet_ref")
    out_sample = caps_dataset[0]
    assert out_sample.label == 1.0
    out_sample = caps_dataset[1]
    assert out_sample.label == 2.0
    with pytest.raises(IndexError):
        caps_dataset[2]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=data,
        label="category",
    )
    caps_dataset.read_tensor_conversion("pet_ref")
    assert caps_dataset.label_dict == {"A": 0, "C": 1}
    out_sample = caps_dataset[0]
    out_sample.label == 0

    # additional info
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(
            extraction=Slice(),
            sample_transforms=[CustomTransform()],
        ),
    )
    caps_dataset.read_tensor_conversion("t1_without_transform")
    out_sample = caps_dataset[0]
    assert "age" in out_sample.keys()

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(
            extraction=Slice(),
            image_transforms=[CustomTransform()],
        ),
    )
    caps_dataset.read_tensor_conversion("t1_without_transform")
    out_sample = caps_dataset[0]
    assert "age" in out_sample.keys()

    data = sub_data([("sub-000", "ses-M000")])
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(
            extraction=Slice(),
            image_transforms=[CustomTransform()],
        ),
    )
    caps_dataset.read_tensor_conversion(
        "t1_custom_interrupted", load_also=["age", "other_image", "other_mask"]
    )
    out_sample = caps_dataset[0]
    assert {"age", "other_image", "other_mask"}.difference(
        set(out_sample.keys())
    ) == set()
