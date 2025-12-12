import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatypes.preprocessing import PETLinear, T1Linear
from clinicadl.data.structures import DataPoint, Mask, Sample, Sample2D
from clinicadl.transforms.config import CropConfig, PadConfig
from clinicadl.transforms.extraction import Patch, Slice
from clinicadl.transforms.handlers import Transforms
from clinicadl.utils.exceptions import CannotReadJsonFieldError

from .utils import subset_df

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
DATAFRAME = pd.read_csv(CAPS_DIR / "tsv" / "labels.tsv", sep="\t")


class CustomImageTransform(tio.Transform):
    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        datapoint["coefficient"] = 0
        datapoint["other_image"] = 0
        datapoint["other_mask"] = 0

        return datapoint


class CustomSampleTransform(tio.Transform):
    def apply_transform(self, datapoint: DataPoint) -> DataPoint:
        assert "coefficient" in datapoint
        assert "other_image" in datapoint
        assert "other_mask" in datapoint

        return datapoint


def sub_data(participants_sessions: list[tuple[str, str]]) -> pd.DataFrame:
    return subset_df(DATAFRAME, participants_sessions)


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
        CAPS_DIR,
        preprocessing,
        transforms=transforms,
        data=data,
        label=label,
        columns=columns,
        masks=masks,
    )
    assert isinstance(caps_dataset.config.transforms.image_transforms, tio.Compose)
    assert len(caps_dataset.config.transforms.image_transforms.transforms) == 1
    assert isinstance(
        caps_dataset.config.transforms.image_transforms.transforms[0],
        tio.RescaleIntensity,
    )

    assert isinstance(caps_dataset.config.transforms.sample_transforms, tio.Compose)
    assert len(caps_dataset.config.transforms.sample_transforms.transforms) == 2
    assert isinstance(
        caps_dataset.config.transforms.sample_transforms.transforms[0], tio.Pad
    )
    assert isinstance(
        caps_dataset.config.transforms.sample_transforms.transforms[1],
        tio.RemapLabels,
    )

    assert isinstance(caps_dataset.config.transforms.augmentations, tio.Compose)
    assert len(caps_dataset.config.transforms.augmentations.transforms) == 1
    assert isinstance(
        caps_dataset.config.transforms.augmentations.transforms[0], tio.Crop
    )

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

    assert not caps_dataset.converted
    assert caps_dataset._tensor_conversion is None
    assert len(caps_dataset.transforms.image_transforms.transforms) == 1
    assert caps_dataset._initial_shape is None
    assert not caps_dataset._has_len

    caps_dataset.read_tensor_conversion(conversion_name="t1_masks")
    output = caps_dataset[0]
    assert set(output.keys()) == {
        "image",
        "label",
        "brain",
        "leftHippocampus",
        "diagnosis",
        "datatype",
        "image_path",
        "participant",
        "session",
        "sample_type",
        "slice_direction",
        "sample_position",
        "squeeze",
    }

    assert caps_dataset.converted
    assert not caps_dataset._tensor_conversion.interrupted
    assert len(caps_dataset.config.transforms.image_transforms.transforms) == 1
    assert caps_dataset._initial_shape == (1, 3, 3, 3)
    assert caps_dataset._has_len


def test_checks(tmp_path):
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
    with pytest.raises(RuntimeError):
        CapsDataset(
            CAPS_DIR,
            preprocessing,
            data,
        )
    preprocessing.use_uncropped_image = True
    CapsDataset(
        CAPS_DIR,
        preprocessing,
        data,
    )

    # check df
    data_path = tmp_path / "only_pets.tsv"
    data.to_csv(data_path, sep="\t", index=False)
    with pytest.raises(FileNotFoundError):
        CapsDataset(
            CAPS_DIR,
            preprocessing,
            CAPS_DIR / "abc.tsv",
        )
    with pytest.raises(ValidationError):
        CapsDataset(
            CAPS_DIR,
            preprocessing,
            data=[("sub-000", "ses-M000")],
        )

    caps_dataset = CapsDataset(
        CAPS_DIR,
        preprocessing,
        tmp_path / "only_pets.tsv",
    )
    assert (caps_dataset.df == data).all().all()

    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=None,
    )
    tsv_path = CAPS_DIR / "overview_t1-linear.tsv"
    tsv = pd.read_csv(tsv_path, sep="\t")
    assert (caps_dataset.df == data[["participant_id", "session_id"]]).all().all()
    assert (tsv == data[["participant_id", "session_id"]]).all().all()
    tsv_path.unlink()

    # check label
    with pytest.raises(
        ValidationError, match="Got 'category' for 'label', but there is no*"
    ):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            label="category",
        )
    with pytest.raises(
        ValidationError,
        match="'category' was passed in 'label', but this column is not numeric!",
    ):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            label="category",
            columns=["category"],
        )
    with pytest.raises(
        ValidationError,
        match="You passed a list in 'label', and this list can only contain*",
    ):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            label=["age", "brain"],
            columns=["age"],
            masks=["brain"],
        )
    with pytest.raises(
        ValidationError,
        match="A segmentation mask must be specific to each image, but you passed*",
    ):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            label="leftHippocampus",
            masks=["leftHippocampus.nii.gz"],
        )

    caps_dataset = CapsDataset(
        CAPS_DIR, datatype=T1Linear(use_uncropped_image=True), data=data, label=None
    )
    assert caps_dataset.label is None

    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        label="age",
        columns=["age"],
    )
    assert caps_dataset.label == "age"

    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        label=["age", "diagnosis"],
        columns={"age": None, "diagnosis": encode_diagnosis},
    )
    assert caps_dataset.label == ["age", "diagnosis"]
    assert caps_dataset.df["diagnosis"].iloc[0] == 0
    assert data["diagnosis"].iloc[0] == "CN"

    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        label="brain",
        masks=["brain"],
    )
    assert isinstance(caps_dataset.label, Mask)
    assert caps_dataset.label.name == "brain"

    # columns
    with pytest.raises(
        ValidationError, match=r"You passed \['age'\] in 'columns', but 'data' is None."
    ):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=None,
            columns=["age"],
        )
    with pytest.raises(ValidationError, match="A column cannot be named*"):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            columns=["affine"],
        )
    with pytest.raises(
        KeyError, match="'abc' was passed in 'columns', but there is no such column*"
    ):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            columns=["abc"],
        )

    # masks
    with pytest.raises(ValidationError):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            masks="leftHippocampus.nii.gz",
        )
    with pytest.raises(ValidationError, match="Mask cannot be named*"):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            masks=["affine"],
        )
    with pytest.raises(
        ValidationError,
        match="Conflict: 'age' has been passed in 'columns' AND 'masks'!",
    ):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            masks=["age"],
            columns=["age"],
        )
    with pytest.raises(ValidationError, match="Duplicated mask names in 'masks'*"):
        CapsDataset(
            CAPS_DIR,
            datatype=T1Linear(use_uncropped_image=True),
            data=data,
            masks=["leftHippocampus", "leftHippocampus.nii.gz"],
        )
    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        masks=["leftHippocampus.nii.gz"],
    )
    assert len(caps_dataset.common_masks) == 1
    assert len(caps_dataset.individual_masks) == 0
    assert isinstance(caps_dataset.common_masks[0], Mask)
    assert (
        caps_dataset.common_masks[0].path
        == CAPS_DIR / "masks" / "leftHippocampus.nii.gz"
    )

    # load also
    caps_dataset = CapsDataset(CAPS_DIR, preprocessing, data, columns=["age"])
    with pytest.raises(ValueError, match="Cannot load the element 'age'*"):
        caps_dataset.read_tensor_conversion(load_also=["age"])

    caps_dataset = CapsDataset(CAPS_DIR, preprocessing, data, masks=["brain"])
    with pytest.raises(ValueError, match="Cannot load the element 'brain'*"):
        caps_dataset.read_tensor_conversion(load_also=["brain"])


def test_get_participant_session_couples():
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
    )
    assert caps_dataset.get_participant_session_couples() == set(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
        ]
    )


def test_describe(tmp_path):
    shutil.copytree(CAPS_DIR, tmp_path, dirs_exist_ok=True)

    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
        ]
    )

    caps_dataset = CapsDataset(
        tmp_path,
        T1Linear(use_uncropped_image=True),
        data=data,
    )
    assert caps_dataset.describe()["total_samples"] == 3

    caps_dataset = CapsDataset(
        tmp_path,
        T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(extraction=Slice()),
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "The operation you are attempting to perform requires your data to be converted into tensors. "
            "Please use 'to_tensors', or 'read_tensor_conversion' if the conversion has already been performed."
        ),
    ):
        caps_dataset.describe()
    caps_dataset.to_tensors(conversion_name="t1_", ignore_spacing=True)
    description = caps_dataset.describe()
    assert description["total_samples"] == 7
    assert sorted(description["participant_session_pairs"]) == [
        ("sub-000", "ses-M000"),
        ("sub-010", "ses-M003"),
        ("sub-010", "ses-M012"),
    ]
    assert description["datatype"] == {
        "name": "T1Linear",
        "pattern": "t1_linear/sub-.*_ses-.*_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.*",
        "key": "t1-linear",
        "description": "T1w images registered to MNI152NLin2009cSym space using Clinica's 't1-linear' pipeline",
        "use_uncropped_image": True,
    }
    assert description["extraction"] == {
        "name": "Slice",
        "borders": None,
        "discarded_slices": None,
        "slice_direction": 0,
        "slices": None,
        "squeeze": True,
        "tsv_path": None,
    }


def test_get_sample_info():
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=PETLinear(
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
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        label="seg",
        masks=["seg"],
        transforms=Transforms(
            extraction=Patch(patch_size=1),
            image_transforms=[CropConfig(cropping=(0, 1, 0, 1, 0, 1))],
        ),
    )
    with pytest.raises(
        RuntimeError, match="The operation you are attempting to perform requires*"
    ):
        caps_dataset.get_sample_info(8, "age")
    caps_dataset.read_tensor_conversion("t1_transform")
    print(caps_dataset.df)
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
        CAPS_DIR,
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


def test_subset(tmp_path):
    shutil.copytree(CAPS_DIR, tmp_path, dirs_exist_ok=True)

    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        tmp_path,
        T1Linear(use_uncropped_image=True),
        transforms=Transforms(extraction=Slice(slices=[0, 1])),
        data=data,
    )
    caps_dataset.to_tensors(conversion_name="for_subset")
    subset = caps_dataset.subset(
        [
            ("sub-010", "ses-M003"),
            (
                "sub-999",
                "ses-M099",
            ),  # not in the dataset, this shouldn't raise an error
        ]
    )
    assert isinstance(subset, CapsDataset)
    assert len(subset) == 2
    assert subset[0].participant == "sub-010"
    assert subset[0].session == "ses-M003"

    with pytest.raises(
        RuntimeError,
        match=r"No \(participant, session\) pairs are in the dataset. This would lead to an empty dataset!",
    ):
        caps_dataset.subset(
            sub_data(
                [
                    ("sub-999", "ses-M099"),
                ]
            )
        )


def test__getitem__(tmp_path):
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
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

    with pytest.raises(
        RuntimeError,
        match="The operation you are attempting to perform requires your data to be converted into tensors.*",
    ):
        caps_dataset[0]

    caps_dataset.read_tensor_conversion(conversion_name="t1_masks")

    tensors = torch.load(
        CAPS_DIR
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
        CAPS_DIR / "masks" / "tensors" / "t1_masks" / "leftHippocampus.pt",
        weights_only=True,
    )
    out_sample = caps_dataset[0]
    assert out_sample["diagnosis"] == 0
    assert out_sample["age"] == 1
    assert out_sample.datatype[0] == T1Linear(use_uncropped_image=True)
    assert out_sample.sample_type == "slice"
    assert isinstance(out_sample, Sample2D)
    assert out_sample.sample_position == 0
    assert out_sample.slice_direction == 0
    assert not out_sample.squeeze
    assert out_sample.participant == "sub-000"
    assert out_sample.session == "ses-M000"
    assert out_sample.image_path[0] == (
        CAPS_DIR
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
    assert out_sample.label.shape == (1, 1, 2, 2)

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
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(
            image_transforms=[CropConfig(cropping=(0, 1, 0, 1, 0, 1))]
        ),
    )
    caps_dataset.read_tensor_conversion("t1_transform")
    assert len(caps_dataset.transforms.image_transforms.transforms) == 0
    out_sample = caps_dataset[0]
    assert out_sample.sample_type == "image"
    assert isinstance(out_sample, Sample)
    assert out_sample.sample_position is None
    assert out_sample.shape == (1, 2, 2, 2)

    caps_dataset.read_tensor_conversion(conversion_name="t1_masks")
    assert len(caps_dataset.transforms.image_transforms.transforms) == 1

    # other label
    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        transforms=Transforms(extraction=Patch(patch_size=3)),
        data=data,
        label="age",
        columns=["age"],
    )
    caps_dataset.read_tensor_conversion()
    out_sample = caps_dataset[0]
    assert isinstance(out_sample, Sample)
    assert out_sample.sample_type == "patch"
    assert out_sample.sample_position == (0, 0, 0)
    assert out_sample.label == 1.0
    out_sample = caps_dataset[1]
    assert out_sample.label == 2.0
    with pytest.raises(IndexError, match="Index out of range*"):
        caps_dataset[2]

    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=data,
        label=["age", "diagnosis"],
        columns={"age": None, "diagnosis": encode_diagnosis},
    )
    caps_dataset.read_tensor_conversion()
    out_sample = caps_dataset[0]
    out_sample.label == [1.0, 0.0]

    # additional info
    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(
            sample_transforms=[CustomSampleTransform()],
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

    shutil.copytree(CAPS_DIR, tmp_path, dirs_exist_ok=True)
    caps_dataset = CapsDataset(
        tmp_path,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        transforms=Transforms(
            image_transforms=[CustomImageTransform()],
            sample_transforms=[CustomSampleTransform()],
        ),
    )
    caps_dataset.to_tensors(conversion_name="new_conversion", save_transforms=True)
    out_sample = caps_dataset[0]
    assert {"coefficient", "other_image", "other_mask"}.difference(
        set(out_sample.keys())
    ) == set()


def test_from_json_to_json(tmp_path):
    data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    data_path = tmp_path / "data.tsv"
    data.to_csv(data_path, sep="\t", index=False)

    data.loc[0, "age"] = np.nan
    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        label="seg",
        masks=["brain", "seg", "leftHippocampus.nii.gz"],
        columns=["age", "diagnosis"],
        transforms=Transforms(
            extraction=Slice(squeeze=False),
            image_transforms=[
                CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
            ],
            sample_transforms=[
                PadConfig(padding=(0, 0, 0, 0, 0, 1)),
            ],
        ),
    )
    caps_dataset.to_json(tmp_path / "dataset.json")
    caps_dataset = CapsDataset.from_json(tmp_path / "dataset.json")
    assert caps_dataset.label.name == "seg"
    assert len(caps_dataset.individual_masks) == 2
    assert len(caps_dataset.common_masks) == 1
    assert len(caps_dataset.columns) == 2
    assert caps_dataset.get_participant_session_couples() == set(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    assert isinstance(caps_dataset.config.datatype, T1Linear)
    assert isinstance(
        caps_dataset.config.transforms.image_transforms.transforms[0], tio.Crop
    )
    assert caps_dataset.config.directory == CAPS_DIR

    # read conversion before
    caps_dataset.read_tensor_conversion("t1_masks")
    caps_dataset.to_json(tmp_path / "dataset.json", overwrite=True)
    caps_dataset = CapsDataset.from_json(tmp_path / "dataset.json")
    assert caps_dataset._tensor_conversion.conversion_name == "t1_masks"
    assert caps_dataset._initial_shape == (1, 3, 3, 3)
    assert caps_dataset.converted
    assert len(caps_dataset.transforms.image_transforms.transforms) == 1
    assert len(caps_dataset) == 4
    assert caps_dataset[0].participant == "sub-000"
    assert caps_dataset[1].session == "ses-M000"

    # read conversion with transforms
    caps_dataset.read_tensor_conversion("t1_transform")
    caps_dataset.to_json(tmp_path / "dataset.json", overwrite=True)
    caps_dataset = CapsDataset.from_json(tmp_path / "dataset.json")
    assert len(caps_dataset.transforms.image_transforms.transforms) == 0

    # to tensors before
    shutil.copytree(CAPS_DIR, tmp_path, dirs_exist_ok=True)
    caps_dataset = CapsDataset(
        tmp_path,
        datatype=T1Linear(use_uncropped_image=True),
        data=data,
        label="seg",
        masks=["brain", "seg", "leftHippocampus.nii.gz"],
        columns=["age", "diagnosis"],
        transforms=Transforms(
            extraction=Slice(squeeze=False),
            image_transforms=[
                CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
            ],
            sample_transforms=[
                tio.Pad(padding=(0, 0, 0, 0, 0, 1)),
            ],
        ),
    )
    caps_dataset.to_tensors(conversion_name="new_conversion", save_transforms=False)
    caps_dataset.to_json(tmp_path / "dataset.json", overwrite=True)
    caps_dataset = CapsDataset.from_json(
        tmp_path / "dataset.json",
        transforms=Transforms(
            extraction=Slice(squeeze=False),
            image_transforms=[
                CropConfig(cropping=(0, 1, 0, 1, 0, 1)),
            ],
            sample_transforms=[
                tio.Pad(padding=(0, 0, 0, 0, 0, 1)),
            ],
        ),
    )
    assert len(caps_dataset) == 4
    assert caps_dataset[0].participant == "sub-000"
    assert caps_dataset[1].session == "ses-M000"
    assert caps_dataset._tensor_conversion.conversion_name == "new_conversion"

    caps_dataset = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=data_path,
        columns={"age": None, "diagnosis": encode_diagnosis},
    )
    caps_dataset.to_json(tmp_path / "dataset.json", overwrite=True)
    with pytest.raises(
        CannotReadJsonFieldError,
        match=r"CapsDataset cannot read the field\(s\) \['columns'\] in .*",
    ):
        CapsDataset.from_json(tmp_path / "dataset.json")
    caps_dataset = CapsDataset.from_json(
        tmp_path / "dataset.json", columns={"age": None, "diagnosis": encode_diagnosis}
    )
    assert caps_dataset.df["diagnosis"].iloc[0] == 0
