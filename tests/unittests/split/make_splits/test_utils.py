from pathlib import Path

import pandas as pd
import pytest

from clinicadl.split.make_splits.utils import (
    extract_baseline,
    find_available_split_dir,
    write_to_tsv,
)

RESROUCES_DIR = Path(__file__).parents[2] / "resources"


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


def test_write_to_tsv(tmp_path):
    (tmp_path / "split").mkdir(parents=True)
    (tmp_path / "split_2").mkdir(parents=True)
    (tmp_path / "split_3").mkdir(parents=True)

    write_to_tsv(
        baseline_df=BASELINE_DF,
        split_dir=tmp_path / "split",
        subset_name="train",
        all_df=DF,
        longitudinal=False,
    )
    assert (tmp_path / "split" / "train_baseline.tsv").is_file()
    assert not (tmp_path / "split" / "train.tsv").is_file()

    with pytest.raises(FileExistsError):
        write_to_tsv(
            baseline_df=BASELINE_DF,
            split_dir=tmp_path / "split",
            subset_name="train",
            all_df=DF,
            longitudinal=False,
        )

    with pytest.raises(ValueError):
        write_to_tsv(
            baseline_df=BASELINE_DF,
            split_dir=tmp_path / "split_2",
            subset_name="train",
            longitudinal=True,
        )

    write_to_tsv(
        baseline_df=BASELINE_DF,
        split_dir=tmp_path / "split_3",
        subset_name="train",
        all_df=DF,
        longitudinal=True,
    )
    assert (tmp_path / "split" / "train_baseline.tsv").is_file()
    longitudinal = pd.read_csv(tmp_path / "split_3" / "train.tsv", sep="\t")
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


def test_find_available_split_dir():
    split_path = find_available_split_dir(RESROUCES_DIR, split_name="split")
    assert split_path == RESROUCES_DIR / "split_2"
