import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset, ConcatDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.transforms import Transforms
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import ClinicaDLCAPSError

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
FULL_DATA = pd.read_csv(CAPS_DIR / "tsv" / "labels.tsv", sep="\t")


def sub_data(
    participants_sessions: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    if not participants_sessions:
        return deepcopy(FULL_DATA)
    data = FULL_DATA.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


def create_caps_datasets(pet_all: bool = False):
    t1_data = sub_data(
        [
            ("sub-000", "ses-M000"),
            ("sub-010", "ses-M003"),
        ]
    )
    if not pet_all:
        pet_data = sub_data(
            [
                ("sub-100", "ses-M000"),
                ("sub-100", "ses-M012"),
                ("sub-999", "ses-M099"),
                ("sub-999", "ses-M999"),
            ]
        )
    else:
        pet_data = sub_data()

    t1_data.loc[0, "age"] = np.nan
    t1_data = t1_data.drop(columns=["diagnosis", "category"])
    pet_data = pet_data.drop(columns="category")

    caps_t1 = CapsDataset(
        CAPS_DIR,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=t1_data,
        transforms=Transforms(extraction=Slice(squeeze=True)),
    )
    caps_pet = CapsDataset(
        CAPS_DIR,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=pet_data,
    )
    return caps_t1, caps_pet


def test_checks():
    caps_t1, caps_pet = create_caps_datasets()
    caps_pet.read_tensor_conversion("pet_spacing-1")
    with pytest.raises(ClinicaDLCAPSError):
        ConcatDataset([caps_t1, caps_pet])
    caps_t1.read_tensor_conversion("t1_all")
    with pytest.raises(ClinicaDLCAPSError):
        ConcatDataset([caps_t1, caps_pet])
    with pytest.warns(
        match="You are trying to concatenate datasets with different dimensionalities:*"
    ):
        ConcatDataset([caps_t1, caps_pet], ignore_spacing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ConcatDataset([caps_t1, caps_pet], ignore_spacing=True, raise_warnings=False)


def test_get_participant_session_couples():
    caps_t1, caps_pet = create_caps_datasets(pet_all=True)
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert sorted(multimodal_dataset.get_participant_session_couples()) == sorted(
        [
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
            ("sub-100", "ses-M000"),
            ("sub-100", "ses-M012"),
            ("sub-999", "ses-M099"),
            ("sub-999", "ses-M999"),
        ]
    )


def test_get_sample_info():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert multimodal_dataset.get_sample_info(7, "age") == 4
    with pytest.raises(IndexError):
        multimodal_dataset.get_sample_info(8, "age")
    with pytest.raises(IndexError):
        multimodal_dataset.get_sample_info(-1, "age")
    with pytest.raises(KeyError):
        multimodal_dataset.get_sample_info(0, "diagnosis")


def test_len():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert len(multimodal_dataset) == 8


def test_describe():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    description = multimodal_dataset.describe()
    assert len(description) == 2
    assert description[0]["total_samples"] == 4
    assert description[1]["total_samples"] == 4


def test_train_val():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    multimodal_dataset.eval()
    assert multimodal_dataset.datasets[0].eval_mode
    assert multimodal_dataset.datasets[1].eval_mode
    multimodal_dataset.train()
    assert not multimodal_dataset.datasets[0].eval_mode
    assert not multimodal_dataset.datasets[1].eval_mode


def test_subset():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    subset = multimodal_dataset.subset(
        sub_data(
            [
                ("sub-999", "ses-M999"),
                ("sub-010", "ses-M003"),
                ("sub-999", "ses-M099"),
            ]
        )
    )
    assert len((subset)) == 4
    assert subset[0].session == "ses-M003"
    assert "T1w" in str(subset[0].image_path)
    assert subset[3].session == "ses-M099"
    assert "pet" in str(subset[3].image_path)
    assert (
        (
            subset.df.fillna(-1)
            == (
                pd.DataFrame(
                    {
                        "dataset_id": [0, 1, 1],
                        "participant_id": ["sub-010", "sub-999", "sub-999"],
                        "session_id": ["ses-M003", "ses-M999", "ses-M099"],
                        "age": [2, 4, 4],
                        "n_samples": [2, 1, 1],
                        "first_idx": [0, 2, 3],
                        "last_idx": [1, 2, 3],
                        "diagnosis": [-1, "CN", "MCI"],
                    }
                )
            )
        )
        .all()
        .all()
    )
    assert (
        (
            subset.datasets[0].df[
                [
                    "participant_id",
                    "session_id",
                    "age",
                    "n_samples",
                    "first_idx",
                    "last_idx",
                ]
            ]
            == pd.DataFrame(
                {
                    "participant_id": ["sub-010"],
                    "session_id": ["ses-M003"],
                    "age": [2],
                    "n_samples": [2],
                    "first_idx": [0],
                    "last_idx": [1],
                }
            )
        )
        .all()
        .all()
    )

    subset = multimodal_dataset.subset(
        sub_data(
            [
                ("sub-999", "ses-M099"),
                ("sub-999", "ses-M999"),
            ]
        )
    )
    assert len(subset.datasets) == 1

    with pytest.raises(
        ClinicaDLCAPSError,
        match=r"No \(participant, session\) pairs mentioned in 'data' are in the ConcatDataset. This would lead to an empty dataset!",
    ):
        multimodal_dataset.subset(
            sub_data(
                [
                    ("sub-010", "ses-M012"),
                ]
            )
        )


def test_df():
    caps_t1 = CapsDataset(
        CAPS_DIR,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=sub_data([("sub-000", "ses-M000")]).drop(
            columns=["diagnosis", "category"]
        ),
        transforms=Transforms(extraction=Slice(squeeze=True)),
    )
    caps_pet = CapsDataset(
        CAPS_DIR,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=sub_data([("sub-999", "ses-M999"), ("sub-000", "ses-M000")]).drop(
            columns=["category"]
        ),
    )
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert multimodal_dataset.df.fillna(-1).equals(
        pd.DataFrame(
            {
                "dataset_id": [0, 1, 1],
                "participant_id": ["sub-000", "sub-999", "sub-000"],
                "session_id": ["ses-M000", "ses-M999", "ses-M000"],
                "age": [1, 4, 1],
                "n_samples": [2, 1, 1],
                "first_idx": [0, 2, 3],
                "last_idx": [1, 2, 3],
                "diagnosis": [-1, "CN", "CN"],
            }
        )
    )


def test__getitem__():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert multimodal_dataset[0].participant == "sub-000"
    assert multimodal_dataset[0].session == "ses-M000"
    assert multimodal_dataset[0].extraction == "slice"
    assert multimodal_dataset[2].participant == "sub-010"
    assert multimodal_dataset[2].session == "ses-M003"
    assert multimodal_dataset[2].extraction == "slice"
    assert multimodal_dataset[4].participant == "sub-100"
    assert multimodal_dataset[4].session == "ses-M000"
    assert multimodal_dataset[4].extraction == "image"
