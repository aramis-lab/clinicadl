from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from clinicadl.data.datasets import (
    CapsDataset,
    ConcatDataset,
    PairedDataset,
)
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
        pet_data = sub_data()[::-1]

    t1_data.loc[0, "age"] = np.nan
    t1_data = t1_data.drop(columns=["diagnosis", "category"])
    pet_data.loc[0, "diagnosis"] = np.nan
    pet_data = pet_data.drop(columns="category")

    caps_t1 = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=t1_data,
        transforms=Transforms(extraction=Slice(slices=[0])),
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
        match="Tensor conversion must be performed BEFORE joining the datasets.*",
    ):
        PairedDataset([caps_t1, caps_pet])

    caps_pet.read_tensor_conversion()
    caps_pet_concat = ConcatDataset([caps_pet, caps_pet])
    with pytest.raises(
        ValueError,
        match="Datasets passed to PairedDataset cannot contain duplicated*",
    ):
        PairedDataset([caps_t1, caps_pet_concat])

    _, caps_pet = create_caps_datasets(pet_all=True)
    caps_pet.read_tensor_conversion()
    with pytest.raises(
        ValueError, match="To pair datasets, they must have exactly the same*"
    ):
        PairedDataset([caps_t1, caps_pet])

    _, caps_pet = create_caps_datasets()
    caps_t1 = CapsDataset(
        CAPS_DIR,
        datatype=T1Linear(use_uncropped_image=True),
        data=deepcopy(caps_pet.df),
        transforms=Transforms(extraction=Slice()),
    )
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    with pytest.raises(
        RuntimeError,
        match=r"For \(sub-000, ses-M000\), different values found for 'n_samples'.*",
    ):
        PairedDataset([caps_t1, caps_pet])


def test_df():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()

    paired = PairedDataset([caps_t1, caps_pet])
    print(paired.df)
    pd.testing.assert_frame_equal(
        paired.df.fillna(-1),
        pd.DataFrame(
            {
                "participant_id": ["sub-000", "sub-010"],
                "session_id": ["ses-M000", "ses-M003"],
                "age": [1, 2],
                "diagnosis": ["CN", -1],
                "n_samples": [1, 1],
            }
        ),
    )


def test_get_participant_session_couples():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    paired = PairedDataset([caps_t1, caps_pet])
    assert paired.get_participant_session_couples() == set(
        [
            ("sub-010", "ses-M003"),
            ("sub-000", "ses-M000"),
        ]
    )


def test_get_sample_info():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
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
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    paired = PairedDataset([caps_t1, caps_pet])
    description = paired.describe()
    assert len(description) == 2
    assert description[0]["total_samples"] == 2
    assert description[1]["total_samples"] == 2


def test_train_val():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    paired = PairedDataset([caps_t1, caps_pet])
    paired.eval()
    assert paired.datasets[0].eval_mode
    assert paired.datasets[1].eval_mode
    paired.train()
    assert not paired.datasets[0].eval_mode
    assert not paired.datasets[1].eval_mode


def test_subset():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    paired = PairedDataset([caps_t1, caps_pet])
    subset = paired.subset(
        sub_data(
            [
                ("sub-010", "ses-M003"),
            ]
        )
    )
    assert len((subset)) == 1
    pd.testing.assert_frame_equal(
        subset.df.fillna(-1),
        pd.DataFrame(
            {
                "participant_id": ["sub-010"],
                "session_id": ["ses-M003"],
                "age": [2],
                "diagnosis": [-1],
                "n_samples": [1],
            }
        ),
    )
    pd.testing.assert_frame_equal(
        subset.datasets[0].df.fillna(-1),
        pd.DataFrame(
            {
                "participant_id": ["sub-010"],
                "session_id": ["ses-M003"],
                "age": [-1.0],
                "n_samples": [1],
            }
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"No \(participant, session\) pairs are in the dataset. This would lead to an empty dataset!",
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
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    paired = PairedDataset([caps_t1, caps_pet])
    assert paired[0][0].participant == "sub-000"
    assert paired[0][0].session == "ses-M000"
    assert paired[0][0].sample_type == "slice"
    assert paired[0][0].datatype[0] == T1Linear(use_uncropped_image=True)
    assert paired[0][1].participant == "sub-000"
    assert paired[0][1].session == "ses-M000"
    assert paired[0][1].sample_type == "image"
    assert paired[0][1].datatype[0] == PETLinear(
        use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
    )


def test_from_json_to_json(tmp_path):
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    paired = PairedDataset([caps_t1, caps_pet])

    paired.to_json(tmp_path / "dataset.json")
    paired = PairedDataset.from_json(tmp_path / "dataset.json")

    assert len(paired) == 2
    assert paired[0][1].participant == "sub-000"
    assert paired[0][1].session == "ses-M000"


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
    dataset_2.df["n_samples"] = [2, 3, 3]

    paired = PairedDataset([dataset_1, dataset_2])
    assert len(paired) == 8

    assert paired[0][0].participant == "sub-000"
    assert paired[0][1].participant == "sub-000"
    assert paired[5][0].participant == "sub-010"
    assert paired[5][1].participant == "sub-010"
    assert paired[5][0].session == "ses-M012"
    assert paired[5][1].session == "ses-M012"
