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
from clinicadl.splitter.splitter import SingleSplit

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
DATA = pd.read_csv(CAPS_DIR / "tsv" / "labels.tsv", sep="\t")

SPLIT_DIR = CAPS_DIR / "splits" / "split"

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
            "participant_id": ["sub-000", "sub-100", "sub-999", "sub-999"],
            "session_id": ["ses-M000", "ses-M012", "ses-M099", "ses-M999"],
        }
    ),
)
CAPS_T1.read_tensor_conversion("t1_all")
CAPS_PET.read_tensor_conversion("pet_all")

SPLITTER = SingleSplit(SPLIT_DIR)


def test_single_split():
    split = SPLITTER.get_split(CAPS)
    assert split.index == 0
    assert split.split_dir == SPLIT_DIR
    assert len(split.train_dataset) == 4
    assert len(split.val_dataset) == 2
    assert set(split.val_dataset.get_participant_session_couples()) == {
        ("sub-010", "ses-M003"),
        ("sub-999", "ses-M099"),
    }

    # test errors
    with pytest.raises(FileNotFoundError, match="No such directory:*"):
        SingleSplit(CAPS_DIR / "splits" / "abc")

    with pytest.raises(FileNotFoundError, match="Required file missing:*"):
        SingleSplit(CAPS_DIR / "splits" / "bad_split")

    with pytest.raises(FileNotFoundError, match="No configuration file found in*"):
        SingleSplit(CAPS_DIR / "splits" / "bad_split_2")


def test_single_split_concat():
    multimodal_dataset = ConcatDataset([CAPS_T1, CAPS_PET])
    split = SPLITTER.get_split(multimodal_dataset)
    assert len(split.train_dataset) == 3
    assert len(split.val_dataset) == 2


def test_single_split_paired():
    paired = PairedDataset([CAPS_T1, CAPS_T1])
    split = SPLITTER.get_split(paired)
    assert len(split.train_dataset) == 1
    assert len(split.val_dataset) == 1


def test_single_split_unpaired():
    unpaired = UnpairedDataset([CAPS_T1, CAPS_PET])
    split = SPLITTER.get_split(unpaired)
    assert len(split.train_dataset) == 1
    assert len(split.val_dataset) == 1
