import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset, UnpairedDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.splitter.make_splits.utils import (
    extract_baseline,
    find_available_split_dir,
    read_and_format_data,
    write_to_tsv,
)
from clinicadl.transforms import Transforms
from clinicadl.transforms.extraction import Slice

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
FULL_DATA = pd.read_csv(CAPS_DIR / "tsv" / "labels.tsv", sep="\t")
TMP_DIR = Path(__file__).parents[2] / "resources" / "tmp"


def sub_data(
    participants_sessions: Optional[list[tuple[str, str]]] = None,
) -> pd.DataFrame:
    if not participants_sessions:
        return FULL_DATA
    data = FULL_DATA.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


caps_t1 = CapsDataset(
    CAPS_DIR,
    preprocessing=T1Linear(use_uncropped_image=True),
    label="age",
    data=sub_data([("sub-000", "ses-M000"), ("sub-010", "ses-M003")]),
    transforms=Transforms(extraction=Slice(slices=[0, 1])),
)
caps_pet = CapsDataset(
    CAPS_DIR,
    preprocessing=PETLinear(
        use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
    ),
    label="age",
    data=sub_data(
        [("sub-000", "ses-M000"), ("sub-010", "ses-M003"), ("sub-999", "ses-M099")]
    ),
)
caps_t1.read_tensor_conversion("t1_all")
caps_pet.read_tensor_conversion("pet_all")
UNPAIRED = UnpairedDataset([caps_t1, caps_pet], oversample=True)

CONCAT_DF = pd.DataFrame(
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

DF = pd.DataFrame(
    {
        "participant_id": [
            "sub-000",
            "sub-000",
            "sub-999",
            "sub-000",
            "sub-999",
            "sub-010",
        ],
        "session_id": [
            "ses-M000",
            "ses-M000",
            "ses-M999",
            "ses-M003",
            "ses-M099",
            "ses-M012",
        ],
        "age": [1, 1, 1, 1, 1, 1],
        "diagnosis": ["AD", "CN", "CN", "CN", "MCI", "AD"],
    }
)

BASELINE_DF = pd.DataFrame(
    {
        "participant_id": [
            "sub-000",
            "sub-010",
            "sub-999",
        ],
        "session_id": [
            "ses-M000",
            "ses-M012",
            "ses-M099",
        ],
        "age": [1, 1, 1],
    }
)


def test_read_and_format_data():
    df = read_and_format_data(CAPS_DIR / "tsv" / "small_test_df.tsv")
    assert set(df.columns) == {"participant_id", "session_id", "age", "diagnosis"}
    assert len(df) == 3

    df = read_and_format_data(CONCAT_DF)
    assert set(df.columns) == {
        "participant_id",
        "session_id",
        "dataset_id",
        "n_samples",
        "age",
        "first_idx",
        "last_idx",
        "diagnosis",
    }
    assert len(df) == 3

    df = read_and_format_data(UNPAIRED.df)
    assert set(df.columns) == {
        "participant_id",
        "session_id",
        "dataset_id",
        "n_samples",
        "age",
        "category",
        "diagnosis",
    }
    assert len(df) == 5


def test_extract_baseline():
    baseline = extract_baseline(DF)
    assert set(zip(baseline["participant_id"], baseline["session_id"])) == {
        ("sub-000", "ses-M000"),
        ("sub-999", "ses-M099"),
        ("sub-010", "ses-M012"),
    }
    assert set(baseline.columns) == {"participant_id", "session_id"}

    baseline = extract_baseline(DF, columns=["age"])
    assert (baseline == BASELINE_DF).all().all()

    with pytest.raises(
        ValueError, match="More than one value found in the dataframe for*"
    ):
        extract_baseline(DF, columns=["diagnosis"])


def test_write_to_tsv():
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    (TMP_DIR / "split").mkdir(parents=True)
    (TMP_DIR / "split_2").mkdir(parents=True)
    (TMP_DIR / "split_3").mkdir(parents=True)

    write_to_tsv(
        baseline_df=BASELINE_DF,
        split_dir=TMP_DIR / "split",
        subset_name="train",
        all_df=DF,
        longitudinal=False,
    )
    assert (TMP_DIR / "split" / "train_baseline.tsv").is_file()
    assert not (TMP_DIR / "split" / "train.tsv").is_file()

    with pytest.raises(FileExistsError):
        write_to_tsv(
            baseline_df=BASELINE_DF,
            split_dir=TMP_DIR / "split",
            subset_name="train",
            all_df=DF,
            longitudinal=False,
        )

    with pytest.raises(ValueError):
        write_to_tsv(
            baseline_df=BASELINE_DF,
            split_dir=TMP_DIR / "split_2",
            subset_name="train",
            longitudinal=True,
        )

    write_to_tsv(
        baseline_df=BASELINE_DF,
        split_dir=TMP_DIR / "split_3",
        subset_name="train",
        all_df=DF,
        longitudinal=True,
    )
    assert (TMP_DIR / "split" / "train_baseline.tsv").is_file()
    longitudinal = pd.read_csv(TMP_DIR / "split_3" / "train.tsv", sep="\t")
    assert (
        (
            longitudinal
            == DF.drop(columns=["diagnosis"])
            .drop_duplicates()
            .sort_values(["participant_id", "session_id"])
            .reset_index(drop=True)
        )
        .all()
        .all()
    )

    shutil.rmtree(TMP_DIR)


def test_find_available_split_dir():
    split_path = find_available_split_dir(CAPS_DIR / "splits", split_name="split")
    assert split_path == CAPS_DIR / "splits" / "split_2"
