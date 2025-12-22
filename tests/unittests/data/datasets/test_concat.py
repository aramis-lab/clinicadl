import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset, ConcatDataset, MultiSamplesDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.transforms.extraction import Image, Slice
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
        datatype=T1Linear(use_uncropped_image=True),
        data=t1_data,
        transforms=Transforms(extraction=Slice(squeeze=True)),
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
        ConcatDataset([caps_t1, caps_pet])
    caps_pet.read_tensor_conversion("pet_spacing-1")
    with pytest.warns(
        match="You are trying to concatenate datasets with different voxel spacings:*"
    ):
        ConcatDataset([caps_t1, caps_pet])
    with pytest.warns(
        match="You are trying to concatenate datasets with different dimensionalities:*"
    ):
        ConcatDataset([caps_t1, caps_pet])

    caps_t1.transforms.extraction = Image()
    caps_pet.read_tensor_conversion()
    with pytest.warns(
        match="You are trying to concatenate datasets with different image shapes:*"
    ):
        ConcatDataset([caps_t1, caps_pet])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ConcatDataset([caps_t1, caps_pet], raise_warnings=False)


def test_get_participant_session_couples():
    caps_t1, caps_pet = create_caps_datasets(pet_all=True)
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    multimodal_dataset = ConcatDataset(
        (d for d in [caps_t1, caps_pet]), raise_warnings=False
    )
    assert multimodal_dataset.get_participant_session_couples() == set(
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
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert multimodal_dataset.get_sample_info(6, "age") == 3
    with pytest.raises(IndexError):
        multimodal_dataset.get_sample_info(10, "age")
    with pytest.raises(IndexError):
        multimodal_dataset.get_sample_info(-1, "age")
    with pytest.raises(
        KeyError,
        match="No column named 'diagnosis' in the metadata DataFrame of the dataset from which the sample is taken.",
    ):
        multimodal_dataset.get_sample_info(0, "diagnosis")


def test_len():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert len(multimodal_dataset) == 10


def test_describe():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    description = multimodal_dataset.describe()
    assert len(description) == 2
    assert description[0]["total_samples"] == 6
    assert description[1]["total_samples"] == 4


def test_train_val():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    multimodal_dataset.eval()
    assert multimodal_dataset.datasets[0].eval_mode
    assert multimodal_dataset.datasets[1].eval_mode
    multimodal_dataset.train()
    assert not multimodal_dataset.datasets[0].eval_mode
    assert not multimodal_dataset.datasets[1].eval_mode


def test_subset():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
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
    assert len((subset)) == 5
    assert subset[0].session == "ses-M003"
    assert "T1w" in str(subset[0].image_path[0])
    assert subset[4].session == "ses-M099"
    assert "pet" in str(subset[4].image_path[0])
    pd.testing.assert_frame_equal(
        subset.df.fillna(-1),
        pd.DataFrame(
            {
                "dataset_id": [0, 1, 1],
                "participant_id": ["sub-010", "sub-999", "sub-999"],
                "session_id": ["ses-M003", "ses-M999", "ses-M099"],
                "age": [2.0, 4.0, 4.0],
                "n_samples": [3, 1, 1],
                "diagnosis": [-1, "CN", "MCI"],
            }
        ),
    )
    pd.testing.assert_frame_equal(
        subset.datasets[0].df[
            [
                "participant_id",
                "session_id",
                "age",
                "n_samples",
            ]
        ],
        pd.DataFrame(
            {
                "participant_id": ["sub-010"],
                "session_id": ["ses-M003"],
                "age": [2.0],
                "n_samples": [3],
            }
        ),
    )

    subset = multimodal_dataset.subset(
        [
            ("sub-999", "ses-M099"),
            ("sub-999", "ses-M999"),
        ]
    )
    assert len(subset.datasets) == 1

    with pytest.raises(
        RuntimeError,
        match=r"No \(participant, session\) pairs are in the dataset. This would lead to an empty dataset!",
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
        datatype=T1Linear(use_uncropped_image=True),
        data=sub_data([("sub-000", "ses-M000")]).drop(
            columns=["diagnosis", "category"]
        ),
        transforms=Transforms(extraction=Slice(squeeze=True)),
    )
    caps_pet = CapsDataset(
        CAPS_DIR,
        datatype=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=sub_data([("sub-999", "ses-M999"), ("sub-000", "ses-M000")]).drop(
            columns=["category"]
        ),
    )
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    pd.testing.assert_frame_equal(
        multimodal_dataset.df.fillna(-1),
        pd.DataFrame(
            {
                "dataset_id": [0, 1, 1],
                "participant_id": ["sub-000", "sub-000", "sub-999"],
                "session_id": ["ses-M000", "ses-M000", "ses-M999"],
                "age": [1, 1, 4],
                "n_samples": [3, 1, 1],
                "diagnosis": [-1, "CN", "CN"],
            }
        ),
    )


def test__getitem__():
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert multimodal_dataset[0].participant == "sub-000"
    assert multimodal_dataset[0].session == "ses-M000"
    assert multimodal_dataset[0].sample_type == "slice"
    assert multimodal_dataset[3].participant == "sub-010"
    assert multimodal_dataset[3].session == "ses-M003"
    assert multimodal_dataset[3].sample_type == "slice"
    assert multimodal_dataset[6].participant == "sub-100"
    assert multimodal_dataset[6].session == "ses-M000"
    assert multimodal_dataset[6].sample_type == "image"


def test_from_json_to_json(tmp_path):
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.read_tensor_conversion()
    caps_pet.read_tensor_conversion()
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet], raise_warnings=False)

    multimodal_dataset.to_json(tmp_path / "dataset.json")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multimodal_dataset = ConcatDataset.from_json(tmp_path / "dataset.json")

    assert len(multimodal_dataset) == 10
    assert multimodal_dataset[6].participant == "sub-100"
    assert multimodal_dataset[6].session == "ses-M000"


def test_custom_dataset():
    from .utils import CustomMultiSamplesDataset

    df = sub_data(
        [
            ("sub-100", "ses-M000"),
            ("sub-100", "ses-M012"),
            ("sub-999", "ses-M099"),
            ("sub-999", "ses-M999"),
        ]
    )
    dataset = CustomMultiSamplesDataset(df)
    with pytest.raises(
        ValueError,
        match="ConcatDataset needs the number of samples per image for each underlying dataset.*",
    ):
        ConcatDataset([dataset, dataset])

    dataset.df["n_samples"] = [1, 2, 2, 1]
    concat = ConcatDataset([dataset, dataset])
    assert len(concat) == 12
    assert concat.get_participant_session_couples() == set(
        [
            ("sub-100", "ses-M000"),
            ("sub-100", "ses-M012"),
            ("sub-999", "ses-M099"),
            ("sub-999", "ses-M999"),
        ]
    )

    concat.get_sample_info(8, "age") == 3
    with pytest.raises(
        NotImplementedError,
        match="'describe' not implemented in CustomMultiSamplesDataset",
    ):
        concat.describe()

    concat.eval()
    assert dataset.evaluation
    concat.train()
    assert not dataset.evaluation

    assert str(concat[8].image_path[0]) == "2"
