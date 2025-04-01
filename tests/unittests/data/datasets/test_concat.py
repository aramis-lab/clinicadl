import shutil
import warnings
from pathlib import Path

import pandas as pd
import pytest

from clinicadl.data.datasets import CapsDataset, ConcatDataset
from clinicadl.data.datatypes import PETLinear, T1Linear
from clinicadl.transforms import Transforms
from clinicadl.transforms.extraction import Slice
from clinicadl.utils.exceptions import ClinicaDLCAPSError, ClinicaDLTSVError

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"
FULL_DATA = pd.read_csv(CAPS_DIR / "labels.tsv", sep="\t")
TMP_DIR = Path(__file__).parents[2] / "resources" / "caps_tmp"


def sub_data(participants_sessions: list[tuple[str, str]]) -> pd.DataFrame:
    data = FULL_DATA.set_index(["participant_id", "session_id"])
    data = data.loc[participants_sessions]
    return data.reset_index()


def copy_caps():
    if TMP_DIR.is_dir():
        shutil.rmtree(TMP_DIR)
    shutil.copytree(CAPS_DIR, TMP_DIR)
    Path(TMP_DIR / "tensor_conversion" / "pet_ref_corrupted.json").unlink()
    Path(TMP_DIR / "tensor_conversion" / "pet_ref_corrupted_bis.json").unlink()
    Path(TMP_DIR / "tensor_conversion" / "pet_ref_missing_field.json").unlink()


def create_caps_datasets(t1_all: bool = False, pet_all: bool = False):
    if not t1_all:
        t1_data = sub_data(
            [
                ("sub-000", "ses-M000"),
                ("sub-010", "ses-M003"),
            ]
        )
    else:
        t1_data = None
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
        pet_data = None

    caps_t1 = CapsDataset(
        TMP_DIR,
        preprocessing=T1Linear(use_uncropped_image=True),
        data=t1_data,
        transforms=Transforms(extraction=Slice(squeeze=True)),
    )
    caps_pet = CapsDataset(
        TMP_DIR,
        preprocessing=PETLinear(
            use_uncropped_image=True, tracer="18FAV45", suvr_reference_region="pons2"
        ),
        data=pet_data,
    )
    return caps_t1, caps_pet


def test_checks():
    copy_caps()
    caps_t1, caps_pet = create_caps_datasets()
    caps_pet.to_tensors("pet_conversion")
    with pytest.raises(ClinicaDLCAPSError):
        ConcatDataset([caps_t1, caps_pet])
    caps_t1.to_tensors("t1_conversion")
    with pytest.raises(ClinicaDLCAPSError):
        ConcatDataset([caps_t1, caps_pet])
    with pytest.warns(
        match="You are trying to concatenate datasets with different dimensionalities:*"
    ):
        ConcatDataset([caps_t1, caps_pet], ignore_spacing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ConcatDataset([caps_t1, caps_pet], ignore_spacing=True, raise_warnings=False)
    shutil.rmtree(TMP_DIR)


def test_get_participant_session_couples():
    copy_caps()
    caps_t1, caps_pet = create_caps_datasets(pet_all=True)
    caps_t1.to_tensors("t1_conversion")
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
    shutil.rmtree(TMP_DIR)


def test_get_sample_info():
    copy_caps()
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.to_tensors("t1_conversion")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert multimodal_dataset.get_sample_info(0, "age") == 1
    assert multimodal_dataset.get_sample_info(9, "age") == 4
    with pytest.raises(IndexError):
        multimodal_dataset.get_sample_info(10, "age")
    shutil.rmtree(TMP_DIR)


def test_len():
    copy_caps()
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.to_tensors("t1_conversion")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert len(multimodal_dataset) == 10
    shutil.rmtree(TMP_DIR)


def test_describe():
    copy_caps()
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.to_tensors("t1_conversion")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    description = multimodal_dataset.describe()
    assert len(description) == 2
    assert description[0]["total_samples"] == 6
    assert description[1]["total_samples"] == 4
    shutil.rmtree(TMP_DIR)


def test_train_val():
    copy_caps()
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.to_tensors("t1_conversion")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    multimodal_dataset.eval()
    assert multimodal_dataset.datasets[0].eval_mode
    assert multimodal_dataset.datasets[1].eval_mode
    multimodal_dataset.train()
    assert not multimodal_dataset.datasets[0].eval_mode
    assert not multimodal_dataset.datasets[1].eval_mode
    shutil.rmtree(TMP_DIR)


def test_subset():
    copy_caps()
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.to_tensors("t1_conversion")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    subset = multimodal_dataset.subset(
        sub_data(
            [
                ("sub-000", "ses-M000"),
                ("sub-999", "ses-M099"),
                ("sub-999", "ses-M999"),
            ]
        )
    )
    assert len((subset)) == 5
    assert subset[0].session == "ses-M000"
    assert "T1w" in subset[0].image_path
    assert subset[3].session == "ses-M099"
    assert "pet" in subset[3].image_path

    subset = multimodal_dataset.subset(
        sub_data(
            [
                ("sub-999", "ses-M099"),
                ("sub-999", "ses-M999"),
            ]
        )
    )
    assert len(subset.datasets) == 1

    with pytest.raises(ClinicaDLTSVError):
        multimodal_dataset.subset(
            sub_data(
                [
                    ("sub-999", "ses-M099"),
                    ("sub-000", "ses-M003"),
                ]
            )
        )

    shutil.rmtree(TMP_DIR)


def test__getitem__():
    copy_caps()
    caps_t1, caps_pet = create_caps_datasets()
    caps_t1.to_tensors("t1_conversion")
    caps_pet.read_tensor_conversion("pet_all")
    multimodal_dataset = ConcatDataset([caps_t1, caps_pet])
    assert multimodal_dataset[0].participant == "sub-000"
    assert multimodal_dataset[0].session == "ses-M000"
    assert multimodal_dataset[0].extraction == "slice"
    assert multimodal_dataset[3].participant == "sub-010"
    assert multimodal_dataset[3].session == "ses-M003"
    assert multimodal_dataset[3].extraction == "slice"
    assert multimodal_dataset[6].participant == "sub-100"
    assert multimodal_dataset[6].session == "ses-M000"
    assert multimodal_dataset[6].extraction == "image"
