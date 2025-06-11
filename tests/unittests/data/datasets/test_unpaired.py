from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset, ConcatDataset, UnpairedDataset
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
        return FULL_DATA
    data = FULL_DATA.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


def create_caps_datasets():
    t1_data = sub_data(
        [
            ("sub-010", "ses-M003"),
            ("sub-000", "ses-M000"),
        ]
    )
    pet_data = sub_data(
        [
            ("sub-010", "ses-M003"),
            ("sub-999", "ses-M099"),
            ("sub-000", "ses-M000"),
        ]
    )
    t1_data = t1_data.drop(columns=["diagnosis", "category"])
    pet_data = pet_data.drop(columns="category")

    caps_t1 = CapsDataset(
        CAPS_DIR,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=t1_data,
        transforms=Transforms(extraction=Slice(slices=[0, 1])),
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
        ClinicaDLCAPSError, match="Tensor conversion must be performed BEFORE stacking*"
    ):
        UnpairedDataset([caps_t1, caps_pet])


def test_df():
    ref_df = pd.DataFrame(
        {
            (0, "participant_id"): ["sub-010", "sub-000", "nan"],
            (0, "session_id"): ["ses-M003", "ses-M000", "nan"],
            (0, "age"): [2, 1, "nan"],
            (0, "n_samples"): [2, 2, "nan"],
            (1, "participant_id"): ["sub-010", "sub-999", "sub-000"],
            (1, "session_id"): ["ses-M003", "ses-M099", "ses-M000"],
            (1, "age"): [2, 4, 1],
            (1, "diagnosis"): ["AD", "MCI", "CN"],
            (1, "n_samples"): [1, 1, 1],
        }
    ).rename_axis(columns=["dataset_id", None])
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=True)
    assert unpaired.df.fillna("nan").equals(ref_df)
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=False)
    assert unpaired.df.fillna("nan").equals(ref_df)


def test_get_participant_session_couples():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = UnpairedDataset([caps_t1, caps_pet], oversample=False)
    assert sorted(paired.get_participant_session_couples()) == sorted(
        [
            ("sub-010", "ses-M003"),
            ("sub-999", "ses-M099"),
            ("sub-000", "ses-M000"),
        ]
    )
    paired = UnpairedDataset([caps_t1, caps_pet], oversample=True)
    assert sorted(paired.get_participant_session_couples()) == sorted(
        [
            ("sub-010", "ses-M003"),
            ("sub-999", "ses-M099"),
            ("sub-000", "ses-M000"),
        ]
    )


def test_get_sample_info():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = UnpairedDataset([caps_t1, caps_pet])
    assert paired.get_sample_info(0, "age") == (1, 1)
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
    paired = UnpairedDataset([caps_t1, caps_pet])
    description = paired.describe()
    assert len(description) == 2
    assert description[0]["total_samples"] == 4
    assert description[1]["total_samples"] == 3


def test_train_val():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = UnpairedDataset([caps_t1, caps_pet])
    paired.eval()
    assert paired.datasets[0].eval_mode
    assert paired.datasets[1].eval_mode
    paired.train()
    assert not paired.datasets[0].eval_mode
    assert not paired.datasets[1].eval_mode


def test_len():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    paired = UnpairedDataset([caps_t1, caps_pet], oversample=True)
    assert len(paired) == 4
    paired = UnpairedDataset([caps_t1, caps_pet])
    assert len(paired) == 3


def test_subset():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    unpaired = UnpairedDataset([caps_t1, caps_pet])
    subset = unpaired.subset(
        sub_data(
            [
                ("sub-999", "ses-M099"),
                ("sub-000", "ses-M000"),
            ]
        )
    )
    assert len(subset) == 2
    assert subset.df.fillna("nan").equals(
        pd.DataFrame(
            {
                (0, "participant_id"): ["sub-000", "nan"],
                (0, "session_id"): ["ses-M000", "nan"],
                (0, "age"): [1, "nan"],
                (0, "n_samples"): [2, "nan"],
                (1, "participant_id"): ["sub-999", "sub-000"],
                (1, "session_id"): ["ses-M099", "ses-M000"],
                (1, "age"): [4, 1],
                (1, "diagnosis"): ["MCI", "CN"],
                (1, "n_samples"): [1, 1],
            }
        ).rename_axis(columns=["dataset_id", None])
    )

    with pytest.raises(
        ClinicaDLCAPSError,
        match=r"No \(participant, session\) pairs mentioned in 'data' are in the CapsDataset. This would lead to an empty dataset!",
    ):
        subset = unpaired.subset(
            sub_data(
                [
                    ("sub-999", "ses-M099"),
                ]
            )
        )


def test_set_epoch():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")

    # oversample
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=True)
    assert unpaired.mapping.equals(
        pd.DataFrame(
            {
                0: [2, 3, 1, 0],
                1: [2, 1, 0, 0],
            }
        ).rename_axis(columns="dataset_id", index="idx")
    )
    assert (
        unpaired[0][0].participant,
        unpaired[0][0].session,
        unpaired[0][0].slice_position,
    ) == ("sub-000", "ses-M000", 0)
    assert (unpaired[0][1].participant, unpaired[0][1].session) == (
        "sub-000",
        "ses-M000",
    )

    unpaired.set_epoch(1)
    assert unpaired.mapping.equals(
        pd.DataFrame(
            {
                0: [3, 2, 0, 1],
                1: [0, 2, 2, 1],
            }
        ).rename_axis(columns="dataset_id", index="idx")
    )
    assert (
        unpaired[0][0].participant,
        unpaired[0][0].session,
        unpaired[0][0].slice_position,
    ) == ("sub-000", "ses-M000", 1)
    assert (unpaired[0][1].participant, unpaired[0][1].session) == (
        "sub-010",
        "ses-M003",
    )

    # undersample
    unpaired = UnpairedDataset([caps_t1, caps_pet])
    assert unpaired.mapping.equals(
        pd.DataFrame(
            {
                0: [2, 3, 1],
                1: [2, 1, 0],
            }
        ).rename_axis(columns="dataset_id", index="idx")
    )
    assert (
        unpaired[0][0].participant,
        unpaired[0][0].session,
        unpaired[0][0].slice_position,
    ) == ("sub-000", "ses-M000", 0)
    assert (unpaired[0][1].participant, unpaired[0][1].session) == (
        "sub-000",
        "ses-M000",
    )

    unpaired.set_epoch(1)
    assert unpaired.mapping.equals(
        pd.DataFrame(
            {
                0: [3, 2, 0],
                1: [0, 2, 1],
            }
        ).rename_axis(columns="dataset_id", index="idx")
    )
    assert (
        unpaired[0][0].participant,
        unpaired[0][0].session,
        unpaired[0][0].slice_position,
    ) == ("sub-000", "ses-M000", 1)
    assert (unpaired[0][1].participant, unpaired[0][1].session) == (
        "sub-010",
        "ses-M003",
    )


def test_unpaired_concat():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion("t1_all")
    caps_pet.read_tensor_conversion("pet_all")
    caps_pet_concat = ConcatDataset([caps_pet, caps_pet])
    unpaired = UnpairedDataset([caps_t1, caps_pet_concat], oversample=True)
    assert len(unpaired) == 6
    assert (unpaired[3][1].participant, unpaired[3][1].session) == (
        "sub-000",
        "ses-M000",
    )
