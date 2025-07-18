import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes.preprocessing import PETLinear, T1Linear
from clinicadl.data.structures import DataPoint, Mask
from clinicadl.transforms import Transforms
from clinicadl.transforms.config import CropConfig, PadConfig
from clinicadl.transforms.extraction import Patch, Slice
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
)

caps_dir = Path(__file__).parents[2] / "resources" / "caps_example"
full_data = pd.read_csv(caps_dir / "tsv" / "labels.tsv", sep="\t")


class CustomTransform(tio.Transform):
    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        assert "coefficient" in datapoint
        assert "other_image" in datapoint
        assert "other_mask" in datapoint

        return datapoint


def sub_data(participants_sessions: list[tuple[str, str]]) -> pd.DataFrame:
    data = full_data.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


def encode_diagnosis(x: pd.Series) -> pd.Series:
    encoding = {"CN": 0, "AD": 2, "MCI": 1}
    return x.apply(lambda x: encoding[x])


def test_good_caps_dataset():
    preprocessing = T1Linear(use_uncropped_image=True)
    transforms = Transforms(
        extraction=Slice(slices=[0]),
        image_transforms=[tio.RescaleIntensity()],
        sample_transforms=[
            PadConfig(padding=1),
            tio.RemapLabels({1: 10}),
        ],
        augmentations=[CropConfig(cropping=1)],
    )
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    label = "age"
    masks = ["brain", "leftHippocampus.nii.gz"]
    columns = {"age": None, "diagnosis": encode_diagnosis}

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing,
        transforms=transforms,
        data=data,
        label=label,
        columns=columns,
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

    assert (
        (caps_dataset.df.drop(columns="diagnosis") == data.drop(columns="diagnosis"))
        .all()
        .all()
    )
    assert caps_dataset.df["diagnosis"].to_list() == [0, 2]
    assert caps_dataset.label == "age"
    assert len(caps_dataset.individual_masks) == 1
    assert isinstance(caps_dataset.individual_masks[0], Mask)
    assert caps_dataset.individual_masks[0].name == "brain"
    assert len(caps_dataset.common_masks) == 1
    assert caps_dataset.common_masks[0].name == "leftHippocampus"
    assert caps_dataset.tensor_conversion.json is None

    caps_dataset.read_tensor_conversion(conversion_name="t1_masks")
    output = caps_dataset[0]
    assert set(output.keys()) == {
        "image",
        "label",
        "brain",
        "leftHippocampus",
        "diagnosis",
        "extraction",
        "preprocessing",
        "image_path",
        "participant",
        "session",
        "slice_direction",
        "slice_position",
        "squeeze",
    }


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
    with pytest.raises(
        ClinicaDLArgumentError, match="Got 'category' for 'label', but there is no*"
    ):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            label="category",
        )
    with pytest.raises(
        ClinicaDLArgumentError,
        match="'category' was passed in 'label', but this column is not numeric!",
    ):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            label="category",
            columns=["category"],
        )
    with pytest.raises(
        ClinicaDLArgumentError,
        match="You passed a list in 'label', and this list can only contain*",
    ):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            label=["age", "brain"],
            columns=["age"],
            masks=["brain"],
        )
    with pytest.raises(
        ClinicaDLArgumentError,
        match="A segmentation mask must be specific to each image, but you passed*",
    ):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            label="leftHippocampus",
            masks=["leftHippocampus.nii.gz"],
        )

    caps_dataset = CapsDataset(
        caps_dir, T1Linear(use_uncropped_image=True), data=data, label=None
    )
    assert caps_dataset.label is None

    caps_dataset = CapsDataset(
        caps_dir,
        T1Linear(use_uncropped_image=True),
        data=data,
        label="age",
        columns=["age"],
    )
    assert caps_dataset.label == "age"

    caps_dataset = CapsDataset(
        caps_dir,
        T1Linear(use_uncropped_image=True),
        data=data,
        label=["age", "diagnosis"],
        columns={"age": None, "diagnosis": encode_diagnosis},
    )
    assert caps_dataset.label == ["age", "diagnosis"]

    caps_dataset = CapsDataset(
        caps_dir,
        T1Linear(use_uncropped_image=True),
        data=data,
        label="brain",
        masks=["brain"],
    )
    assert isinstance(caps_dataset.label, Mask)
    assert caps_dataset.label.name == "brain"

    # columns
    with pytest.raises(ClinicaDLArgumentError, match="A column cannot be named*"):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            columns=["affine"],
        )
    with pytest.raises(
        KeyError, match="'abc' was passed in 'columns', but there is no such column*"
    ):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            columns=["abc"],
        )

    # masks
    with pytest.raises(ClinicaDLArgumentError, match="'masks' should be a list*"):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            masks="leftHippocampus.nii.gz",
        )
    with pytest.raises(ClinicaDLArgumentError, match="Mask cannot be named*"):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            masks=["affine"],
        )
    with pytest.raises(
        ClinicaDLArgumentError,
        match="Conflict: 'age' has been passed in 'columns' AND 'masks'!",
    ):
        CapsDataset(
            caps_dir,
            T1Linear(use_uncropped_image=True),
            data=data,
            masks=["age"],
            columns=["age"],
        )
    with pytest.raises(
        ClinicaDLArgumentError, match="Duplicated mask names in 'masks'*"
    ):
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

    # load also
    caps_dataset = CapsDataset(caps_dir, preprocessing, data, columns=["age"])
    with pytest.raises(ClinicaDLArgumentError, match="Cannot load the element 'age'*"):
        caps_dataset.read_tensor_conversion(load_also=["age"])

    caps_dataset = CapsDataset(caps_dir, preprocessing, data, masks=["brain"])
    with pytest.raises(
        ClinicaDLArgumentError, match="Cannot load the element 'brain'*"
    ):
        caps_dataset.read_tensor_conversion(load_also=["brain"])


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
    with pytest.raises(
        ClinicaDLCAPSError, match="Needs tensors to compute the length of the dataset*"
    ):
        caps_dataset.describe()
    caps_dataset.to_tensors(conversion_name="t1_", ignore_spacing=True)
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
    with pytest.raises(KeyError, match="No column named 'abc'*"):
        caps_dataset.get_sample_info(0, "abc")
    with pytest.raises(IndexError, match="Index must be a non-negative integer*"):
        caps_dataset.get_sample_info(-1, "age")
    with pytest.raises(IndexError, match="Index out of range, there are only*"):
        caps_dataset.get_sample_info(2, "age")

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        label="seg",
        masks=["seg"],
        transforms=Transforms(
            extraction=Patch(patch_size=1, stride=1),
            image_transforms=[CropConfig(cropping=(0, 1, 0, 1, 0, 1))],
        ),
    )
    with pytest.raises(
        ClinicaDLCAPSError, match="Needs tensors to compute the length of the dataset*"
    ):
        caps_dataset.get_sample_info(8, "age")
    caps_dataset.read_tensor_conversion("t1_masks")
    assert caps_dataset.get_sample_info(7, "age") == 1.0
    assert caps_dataset.get_sample_info(8, "age") == 2.0


def test_train_eval():
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        caps_dir,
        T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(augmentations=[tio.RescaleIntensity()]),
    )
    caps_dataset.read_tensor_conversion()
    assert not caps_dataset.eval_mode

    caps_dataset.eval()
    assert caps_dataset.eval_mode
    out = caps_dataset[0]
    assert out.image.tensor.max() != 1

    caps_dataset.train()
    assert not caps_dataset.eval_mode
    out = caps_dataset[0]
    assert out.image.tensor.max() == 1


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
    caps_dataset.to_tensors(conversion_name="for_subset")
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
        masks=["brain", "seg", "leftHippocampus.nii.gz"],
        columns={"age": None, "diagnosis": encode_diagnosis},
        transforms=Transforms(
            extraction=Slice(squeeze=False),
            image_transforms=[
                CropConfig(cropping=(0, 0, 0, 1, 0, 1)),
                tio.OneHot(num_classes=7, include=["label"]),
            ],
            sample_transforms=[
                tio.RescaleIntensity(masking_method="brain"),
                tio.Mask(masking_method="leftHippocampus"),
            ],
            augmentations=[tio.RemapLabels({1: 10})],
        ),
    )

    with pytest.raises(ClinicaDLCAPSError, match="Cannot find tensor files.*"):
        caps_dataset[0]

    caps_dataset.read_tensor_conversion(conversion_name="t1_masks")

    tensors = torch.load(
        caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "t1_masks"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt",
        weights_only=True,
    )
    mask_tensor = torch.load(
        caps_dir / "masks" / "tensors" / "t1_masks" / "leftHippocampus.pt",
        weights_only=True,
    )
    out_sample = caps_dataset[0]
    assert out_sample["diagnosis"] == 0
    assert out_sample["age"] == 1
    assert out_sample.preprocessing == T1Linear(use_uncropped_image=True)
    assert out_sample.extraction == "slice"
    assert out_sample.slice_position == 0
    assert out_sample.slice_direction == 0
    assert not out_sample.squeeze
    assert out_sample.participant == "sub-000"
    assert out_sample.session == "ses-M000"
    assert out_sample.image_path == (
        caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "tensors"
        / "t1_masks"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    )

    assert (out_sample.affine == tensors["affine"]).all()
    assert out_sample.spatial_shape == (1, 2, 2)
    assert out_sample.label.shape == (7, 1, 2, 2)

    compose = tio.Compose(
        [
            tio.Crop(cropping=(0, 0, 0, 1, 0, 1)),
            tio.RescaleIntensity(masking_method="brain"),
            tio.Mask(masking_method="leftHippocampus"),
        ]
    )
    ref_subject = tio.Subject(
        image=tio.ScalarImage(tensor=tensors["image"][:, 0:1]),
        brain=tio.LabelMap(tensor=tensors["brain"][:, 0:1]),
        leftHippocampus=tio.LabelMap(tensor=mask_tensor["mask"][:, 0:1]),
    )
    assert (out_sample.image.tensor == (compose(ref_subject)).image.tensor).all()
    assert torch.unique(out_sample.label.tensor).tolist() == [0, 10]
    assert torch.unique(out_sample["brain"].tensor).tolist() == [10]
    assert torch.unique(out_sample["leftHippocampus"].tensor).tolist() == [0, 10]

    # check that image transform is not applied twice
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(
            image_transforms=[CropConfig(cropping=(0, 1, 0, 1, 0, 1))]
        ),
    )
    caps_dataset.read_tensor_conversion("t1_transform")
    out_sample = caps_dataset[0]
    assert out_sample.shape == (1, 2, 2, 2)

    # other label
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        label="age",
        columns=["age"],
    )
    caps_dataset.read_tensor_conversion()
    out_sample = caps_dataset[0]
    assert out_sample.label == 1.0
    out_sample = caps_dataset[1]
    assert out_sample.label == 2.0
    with pytest.raises(IndexError, match="Index out of range*"):
        caps_dataset[2]

    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=data,
        label=["age", "diagnosis"],
        columns={"age": None, "diagnosis": encode_diagnosis},
    )
    caps_dataset.read_tensor_conversion()
    out_sample = caps_dataset[0]
    out_sample.label == {"age": 1.0, "diagnosis": 0}

    # additional info
    caps_dataset = CapsDataset(
        caps_dir,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(
            sample_transforms=[CustomTransform()],
        ),
    )
    caps_dataset.read_tensor_conversion(
        conversion_name="t1_transform",
        load_also=["coefficient", "other_image", "other_mask"],
        check_transforms=False,
    )
    out_sample = caps_dataset[0]
    assert {"coefficient", "other_image", "other_mask"}.difference(
        set(out_sample.keys())
    ) == set()

    caps_dataset.read_tensor_conversion(
        conversion_name="t1_transform",
        check_transforms=False,
    )
    out_sample = caps_dataset[0]
    assert {"coefficient", "other_image", "other_mask"}.intersection(
        set(out_sample.keys())
    ) == set()
