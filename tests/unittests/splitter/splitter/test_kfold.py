from pathlib import Path

import pandas as pd
import pytest

from clinicadl.data.datasets import (
    CapsDataset,
    ConcatDataset,
    PairedDataset,
    UnpairedDataset,
)
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.data.datatypes.preprocessing import PETLinear
from clinicadl.splitter.splitter import KFold

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
DATA = pd.read_csv(CAPS_DIR / "tsv" / "labels.tsv", sep="\t")

SPLIT_DIR = CAPS_DIR / "splits" / "split" / "2_fold"

CAPS = CapsDataset(
    CAPS_DIR,
    preprocessing=PETLinear(
        tracer="18FAV45",
        suvr_reference_region="pons2",
        use_uncropped_image=True,
    ),
    data=DATA,
)
CAPS_T1 = CapsDataset(
    CAPS_DIR,
    preprocessing=T1Linear(use_uncropped_image=True),
    data=pd.DataFrame.from_dict(
        {
            "participant_id": ["sub-000", "sub-010"],
            "session_id": ["ses-M000", "ses-M003"],
        }
    ),
)
CAPS_PET = CapsDataset(
    CAPS_DIR,
    preprocessing=PETLinear(
        use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
    ),
    data=pd.DataFrame.from_dict(
        {
            "participant_id": ["sub-000", "sub-100", "sub-999"],
            "session_id": ["ses-M003", "ses-M012", "ses-M099"],
        }
    ),
)
CAPS_T1.read_tensor_conversion("t1_all")
CAPS_PET.read_tensor_conversion("pet_all")

SPLITTER = KFold(SPLIT_DIR)


def test_kfold():
    splits = iter(SPLITTER.get_splits(CAPS))
    split = next(splits)
    assert split.index == 0
    assert split.split_dir == SPLIT_DIR
    assert len(split.train_dataset) == 2
    assert len(split.val_dataset) == 2
    assert set(split.val_dataset.get_participant_session_couples()) == {
        ("sub-100", "ses-M000"),
        ("sub-100", "ses-M012"),
    }

    split = next(splits)
    assert split.index == 1

    with pytest.raises(StopIteration):
        next(splits)

    # test "splits" arg
    splits = iter(SPLITTER.get_splits(CAPS, splits=[1]))
    split = next(splits)
    assert split.index == 1
    with pytest.raises(StopIteration):
        next(splits)

    # indexerror
    splits = iter(SPLITTER.get_splits(CAPS, splits=[0, 2]))
    next(splits)
    with pytest.raises(
        IndexError,
        match="Split '2' doesn't exist. There are 2 splits, numbered from 0 to 1.",
    ):
        next(splits)

    # errors
    with pytest.raises(FileNotFoundError, match="No such directory:*"):
        KFold(CAPS_DIR / "splits" / "bad_split" / "2_fold")


def test_kfold_concat():
    multimodal_dataset = ConcatDataset([CAPS_T1, CAPS_PET])
    splits = iter(SPLITTER.get_splits(multimodal_dataset))
    split = next(splits)
    assert len(split.train_dataset) == 2
    assert len(split.val_dataset) == 1


def test_kfold_paired():
    paired = PairedDataset([CAPS_PET, CAPS_PET])
    splits = iter(SPLITTER.get_splits(paired))
    split = next(splits)
    assert len(split.train_dataset) == 1
    assert len(split.val_dataset) == 1


def test_kfold_unpaired():
    unpaired = UnpairedDataset([CAPS_PET, CAPS_PET])
    splits = iter(SPLITTER.get_splits(unpaired))
    split = next(splits)
    assert len(split.train_dataset) == 1
    assert len(split.val_dataset) == 1
