from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset, UnpairedDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.transforms.extraction import Slice
from clinicadl.transforms.handlers import Transforms
from clinicadl.utils.exceptions import TensorConversionError

from .utils import subset_df

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
DATAFRAME = pd.read_csv(CAPS_DIR / "tsv" / "labels.tsv", sep="\t")


def sub_data(
    participants_sessions: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    return subset_df(DATAFRAME, participants_sessions)


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
        datatype=T1Linear(use_uncropped_image=True),
        data=t1_data,
        transforms=Transforms(extraction=Slice(slices=[0, 1])),
    )
    caps_pet = CapsDataset(
        CAPS_DIR,
        datatype=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=pet_data,
    )

    return caps_t1, caps_pet


def test_checks():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    with pytest.raises(
        TensorConversionError,
        match="Tensor conversion must be performed BEFORE joining*",
    ):
        UnpairedDataset([caps_t1, caps_pet])


def test_df():
    ref_df = pd.DataFrame(
        {
            "dataset_id": [0, 0, 1, 1, 1],
            "participant_id": ["sub-000", "sub-010", "sub-000", "sub-010", "sub-999"],
            "session_id": ["ses-M000", "ses-M003", "ses-M000", "ses-M003", "ses-M099"],
            "age": [1, 2, 1, 2, 4],
            "n_samples": [2, 2, 1, 1, 1],
            "diagnosis": ["nan", "nan", "CN", "AD", "MCI"],
        }
    )
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=True)
    print(unpaired.df)
    assert unpaired.df.fillna("nan").equals(ref_df)
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=False)
    assert unpaired.df.fillna("nan").equals(ref_df)


def test_get_participant_session_couples():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=False)
    assert unpaired.get_participant_session_couples() == set(
        [
            ("sub-010", "ses-M003"),
            ("sub-999", "ses-M099"),
            ("sub-000", "ses-M000"),
        ]
    )
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=True)
    assert unpaired.get_participant_session_couples() == set(
        [
            ("sub-010", "ses-M003"),
            ("sub-999", "ses-M099"),
            ("sub-000", "ses-M000"),
        ]
    )


def test_get_sample_info():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    unpaired = UnpairedDataset([caps_t1, caps_pet])
    assert unpaired.get_sample_info(0, "age") == (2, 4)
    assert unpaired.get_sample_info(2, "age") == (1, 1)
    with pytest.raises(IndexError):
        unpaired.get_sample_info(10, "age")
    with pytest.raises(IndexError):
        unpaired.get_sample_info(-1, "age")
    with pytest.raises(KeyError):
        unpaired.get_sample_info(0, "abc")


def test_describe():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    unpaired = UnpairedDataset([caps_t1, caps_pet])
    description = unpaired.describe()
    assert len(description) == 2
    assert description[0]["total_samples"] == 4
    assert description[1]["total_samples"] == 3


def test_train_val():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    unpaired = UnpairedDataset([caps_t1, caps_pet])
    unpaired.eval()
    assert unpaired.datasets[0].eval_mode
    assert unpaired.datasets[1].eval_mode
    unpaired.train()
    assert not unpaired.datasets[0].eval_mode
    assert not unpaired.datasets[1].eval_mode


def test_len():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=True)
    assert len(unpaired) == 4
    unpaired = UnpairedDataset([caps_t1, caps_pet])
    assert len(unpaired) == 3


def test_subset():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
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
                "dataset_id": [0, 1, 1],
                "participant_id": ["sub-000", "sub-999", "sub-000"],
                "session_id": ["ses-M000", "ses-M099", "ses-M000"],
                "age": [1, 4, 1],
                "n_samples": [2, 1, 1],
                "diagnosis": ["nan", "MCI", "CN"],
            }
        )
    )

    with pytest.raises(
        RuntimeError,
        match=r"No \(participant, session\) pairs are in the dataset. This would lead to an empty dataset!",
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
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()

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
        unpaired[2][0].participant,
        unpaired[2][0].session,
        unpaired[2][0].sample_position,
    ) == ("sub-000", "ses-M000", 1)
    assert (unpaired[2][1].participant, unpaired[2][1].session) == (
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
        unpaired[0][0].sample_position,
    ) == ("sub-010", "ses-M003", 1)
    assert (unpaired[0][1].participant, unpaired[0][1].session) == (
        "sub-000",
        "ses-M000",
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
        unpaired[0][0].sample_position,
    ) == ("sub-010", "ses-M003", 0)
    assert (unpaired[0][1].participant, unpaired[0][1].session) == (
        "sub-999",
        "ses-M099",
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


def test_from_json_to_json(tmp_path):
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    unpaired = UnpairedDataset([caps_t1, caps_pet], oversample=True)

    unpaired.to_json(tmp_path / "dataset.json")
    unpaired = UnpairedDataset.from_json(tmp_path / "dataset.json")

    assert len(unpaired) == 4
    assert (
        unpaired[2][0].participant,
        unpaired[2][0].session,
        unpaired[2][0].sample_position,
    ) == ("sub-000", "ses-M000", 1)
    assert (unpaired[2][1].participant, unpaired[2][1].session) == (
        "sub-000",
        "ses-M000",
    )


def test_custom():
    from .utils import CustomMultiSamplesDataset

    dataset_1 = CustomMultiSamplesDataset(
        sub_data(
            [
                ("sub-000", "ses-M000"),
                ("sub-010", "ses-M012"),
                ("sub-010", "ses-M003"),
            ]
        )
    )
    dataset_2 = CustomMultiSamplesDataset(
        sub_data(
            [
                ("sub-010", "ses-M003"),
                ("sub-000", "ses-M000"),
                ("sub-010", "ses-M012"),
            ]
        )
    )
    dataset_1.df["n_samples"] = [3, 3, 2]
    dataset_2.df["n_samples"] = [1, 1, 2]

    unpaired = UnpairedDataset([dataset_1, dataset_2], oversample=True)
    assert len(unpaired) == 8

    assert (
        unpaired[0][0].participant,
        unpaired[0][0].session,
        str(unpaired[0][0].image_path[0]),
    ) == ("sub-010", "ses-M003", "6")
    assert (
        unpaired[0][1].participant,
        unpaired[0][1].session,
        str(unpaired[0][1].image_path[0]),
    ) == ("sub-010", "ses-M012", "2")
