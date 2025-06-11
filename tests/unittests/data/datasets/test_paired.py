from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset, ConcatDataset, PairedDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.transforms import Transforms
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import ClinicaDLCAPSError, ClinicaDLTSVError

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
            ("sub-010", "ses-M003"),
            ("sub-000", "ses-M000"),
        ]
    )
    if not pet_all:
        pet_data = deepcopy(t1_data)
    else:
        pet_data = sub_data()

    t1_data.loc[0, "age"] = np.nan
    t1_data = t1_data.drop(columns=["diagnosis", "category"])
    pet_data.loc[0, "diagnosis"] = np.nan
    pet_data = pet_data.drop(columns="category")

    caps_t1 = CapsDataset(
        CAPS_DIR,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=t1_data,
        transforms=Transforms(extraction=Slice(slices=[0])),
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
    caps_t1.read_tensor_conversion("t1_all")
    with pytest.raises(
        ClinicaDLCAPSError, match="Tensor conversion must be performed BEFORE pairing*"
    ):
        PairedDataset([caps_t1, caps_pet])

    caps_pet.read_tensor_conversion("pet_all")
    caps_pet_concat = ConcatDataset([caps_pet, caps_pet])
    with pytest.raises(
        ClinicaDLCAPSError,
        match="Datasets passed to 'PairedDataset' cannot contain duplicated*",
    ):
        PairedDataset([caps_t1, caps_pet_concat])

    _, caps_pet = create_caps_datasets(pet_all=True)
    caps_pet.read_tensor_conversion("pet_all")
    with pytest.raises(
        ClinicaDLCAPSError, match="To pair datasets, they must have exactly the same*"
    ):
        PairedDataset([caps_t1, caps_pet])

    _, caps_pet = create_caps_datasets()
    caps_t1 = CapsDataset(
        CAPS_DIR,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=caps_pet.df,
        transforms=Transforms(extraction=Slice()),
    )
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    with pytest.raises(
        ClinicaDLCAPSError,
        match=r"For \(sub-000, ses-M000\), different values found for 'n_samples'*",
    ):
        PairedDataset([caps_t1, caps_pet])


def test_df():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    assert (
        (
            caps_t1.df.fillna(-1)
            == pd.DataFrame(
                {
                    "participant_id": ["sub-010", "sub-000"],
                    "session_id": ["ses-M003", "ses-M000"],
                    "age": [-1, 1],
                    "n_samples": [1, 1],
                    "first_idx": [0, 1],
                    "last_idx": [0, 1],
                }
            )
        )
        .all()
        .all()
    )
    paired = PairedDataset([caps_t1, caps_pet])
    assert (
        (
            paired.df.fillna(-1)
            == pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [1, 2],
                    "n_samples": [1, 1],
                    "first_idx": [0, 1],
                    "last_idx": [0, 1],
                    "diagnosis": ["CN", -1],
                }
            )
        )
        .all()
        .all()
    )
    assert (
        (
            caps_t1.df.fillna(-1)
            == pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [1, -1],
                    "n_samples": [1, 1],
                    "first_idx": [0, 1],
                    "last_idx": [0, 1],
                }
            )
        )
        .all()
        .all()
    )


def test_get_participant_session_couples():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = PairedDataset([caps_t1, caps_pet])
    assert sorted(paired.get_participant_session_couples()) == sorted(
        [
            ("sub-010", "ses-M003"),
            ("sub-000", "ses-M000"),
        ]
    )


def test_get_sample_info():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = PairedDataset([caps_t1, caps_pet])
    assert paired.get_sample_info(0, "age") == 1
    with pytest.raises(IndexError):
        paired.get_sample_info(10, "age")
    with pytest.raises(IndexError):
        paired.get_sample_info(-1, "age")
    with pytest.raises(KeyError):
        paired.get_sample_info(0, "abc")


def test_describe():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = PairedDataset([caps_t1, caps_pet])
    description = paired.describe()
    assert len(description) == 2
    assert description[0]["total_samples"] == 2
    assert description[1]["total_samples"] == 2


def test_train_val():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = PairedDataset([caps_t1, caps_pet])
    paired.eval()
    assert paired.datasets[0].eval_mode
    assert paired.datasets[1].eval_mode
    paired.train()
    assert not paired.datasets[0].eval_mode
    assert not paired.datasets[1].eval_mode


def test_subset():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = PairedDataset([caps_t1, caps_pet])
    assert (
        (
            paired.datasets[0].df.fillna(-1)
            == pd.DataFrame(
                {
                    "participant_id": ["sub-000", "sub-010"],
                    "session_id": ["ses-M000", "ses-M003"],
                    "age": [1, -1],
                    "n_samples": [1, 1],
                    "first_idx": [0, 1],
                    "last_idx": [0, 1],
                }
            )
        )
        .all()
        .all()
    )

    subset = paired.subset(
        sub_data(
            [
                ("sub-010", "ses-M003"),
            ]
        )
    )
    assert len((subset)) == 1
    assert (
        (
            subset.df.fillna(-1)
            == pd.DataFrame(
                {
                    "participant_id": ["sub-010"],
                    "session_id": ["ses-M003"],
                    "age": [2],
                    "n_samples": [1],
                    "first_idx": [0],
                    "last_idx": [0],
                    "diagnosis": [-1],
                }
            )
        )
        .all()
        .all()
    )
    assert (
        (
            subset.datasets[0].df.fillna(-1)
            == pd.DataFrame(
                {
                    "participant_id": ["sub-010"],
                    "session_id": ["ses-M003"],
                    "age": [-1],
                    "n_samples": [1],
                    "first_idx": [0],
                    "last_idx": [0],
                }
            )
        )
        .all()
        .all()
    )

    with pytest.raises(
        ClinicaDLCAPSError,
        match=r"No \(participant, session\) pairs mentioned in 'data' are in the CapsDataset. This would lead to an empty dataset!",
    ):
        paired.subset(
            sub_data(
                [
                    ("sub-999", "ses-M099"),
                ]
            )
        )


def test__getitem__():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = PairedDataset([caps_t1, caps_pet])
    assert paired[0][0].participant == "sub-000"
    assert paired[0][0].session == "ses-M000"
    assert paired[0][0].extraction == "slice"
    assert paired[0][0].preprocessing == T1Linear(use_uncropped_image=True)
    assert paired[0][1].participant == "sub-000"
    assert paired[0][1].session == "ses-M000"
    assert paired[0][1].extraction == "image"
    assert paired[0][1].preprocessing == PETLinear(
        use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
    )


def test_paired_concat():
    caps_t1, _ = create_caps_datasets()
    caps_pet_1 = CapsDataset(
        CAPS_DIR,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=sub_data(
            [
                ("sub-010", "ses-M003"),
            ]
        ),
    )
    caps_pet_2 = CapsDataset(
        CAPS_DIR,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=sub_data(
            [
                ("sub-000", "ses-M000"),
            ]
        ),
    )
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet_1.read_tensor_conversion("pet_all")
    caps_pet_2.read_tensor_conversion("pet_all")
    caps_pet_concat = ConcatDataset([caps_pet_1, caps_pet_2])
    paired = PairedDataset([caps_t1, caps_pet_concat])
    assert (paired[1][1].participant, paired[1][1].session) == ("sub-000", "ses-M000")
