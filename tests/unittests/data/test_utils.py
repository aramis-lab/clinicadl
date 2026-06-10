import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchio as tio

from clinicadl.data.structures import DataPoint
from clinicadl.data.utils import DatasetChecker, remove_tensors

TENSORS = Path(__file__).parents[1] / "resources" / "bids" / "derivatives" / "tensors"


def test_remove_tensors(tmp_path):
    shutil.copytree(TENSORS, tmp_path, dirs_exist_ok=True)

    assert (
        tmp_path
        / "sub-000"
        / "ses-M000"
        / "tensors"
        / "sub-000_ses-M000_res-1d3x1d2x1d1_src-T1w_conv-T1Masks_tensors.pt"
    ).is_file()
    assert (
        tmp_path
        / "sub-000"
        / "ses-M000"
        / "tensors"
        / "sub-000_ses-M000_res-1d3x1d2x1d1_src-T1w_conv-T1Masks_tensors.json"
    ).is_file()
    assert (
        tmp_path / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_participantsXsessions.tsv"
    ).is_file()

    remove_tensors(tmp_path / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_description.json")

    assert not (
        tmp_path / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_tensors.json"
    ).is_file()
    assert not (
        tmp_path / "res-1d3x1d2x1d1_src-T1w_conv-T1Masks_participantsXsessions.tsv"
    ).is_file()
    assert not (
        tmp_path
        / "sub-000"
        / "ses-M000"
        / "tensors"
        / "sub-000_ses-M000_res-1d3x1d2x1d1_src-T1w_conv-T1Masks_tensors.pt"
    ).is_file()
    assert not (
        tmp_path
        / "sub-000"
        / "ses-M000"
        / "tensors"
        / "sub-000_ses-M000_res-1d3x1d2x1d1_src-T1w_conv-T1Masks_tensors.json"
    ).is_file()
    assert not (
        tmp_path
        / "sub-010"
        / "ses-M003"
        / "tensors"
        / "sub-010_ses-M003_res-1d3x1d2x1d1_src-T1w_conv-T1Masks_tensors.pt"
    ).is_file()
    assert not (
        tmp_path
        / "sub-010"
        / "ses-M003"
        / "tensors"
        / "sub-010_ses-M003_res-1d3x1d2x1d1_src-T1w_conv-T1Masks_tensors.json"
    ).is_file()
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "conversions.tsv", sep="\t"),
        pd.DataFrame.from_dict(
            {
                "conv_id": {0: "raw", 1: "T1Transform", 2: "PetSpacing1"},
                "description": {
                    0: "Raw images are converted without any transformation.",
                    1: "T1 images cropped",
                    2: "PET images with spacing 1.00x1.00x1.00mm",
                },
                "description_json": {
                    0: "*_conv-raw_description.json",
                    1: "src-T1w_conv-T1Transform_description.json",
                    2: "trc-18FAV45_res-0d8x0d8x0d8_src-pet_conv-PetSpacing1_description.json",
                },
            }
        ),
    )


class Dataset:
    def __init__(self, indices):
        self.indices = indices

    def __getitem__(self, idx):
        id_ = self.indices[idx]
        if id_ == 0:
            return DataPoint(
                participant_id="sub-000",
                session_id="ses-M000",
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
                mask=tio.LabelMap(tensor=torch.randn(1, 3, 3, 3)),
            )
        if id_ == 1:
            affine = np.eye(4) * 2
            affine_ = affine.copy()
            affine[:, 3] = [1, 1, 1, 1]
            return DataPoint(
                participant_id="sub-001",
                session_id="ses-M001",
                image=tio.ScalarImage(tensor=torch.randn(1, 4, 4, 4), affine=affine),
                mask=tio.LabelMap(tensor=torch.randn(1, 4, 4, 4), affine=affine_),
            )
        if id_ == 2:
            return DataPoint(
                participant_id="sub-002",
                session_id="ses-M000",
                image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
                mask=tio.LabelMap(tensor=torch.randn(1, 1, 1, 1), affine=np.eye(4) * 2),
            )
        if id_ == 3:
            return DataPoint(
                participant_id="sub-003",
                session_id="ses-M000",
                image=tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3)),
                mask=tio.LabelMap(tensor=torch.randn(1, 3, 3, 3)),
            )


class TestDatasetChecker:
    def test_spacing(self):
        dataset = Dataset(indices=[2])
        checker = DatasetChecker(spatial_checks=["spacing"])
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "An error occurred when checking (sub-002, ses-M000) (see above). If you don't care about voxel spacing consistency and want to ignore this error, please modify 'spatial_checks'."
            ),
        ):
            checker.check_data_point(dataset[0])
        checker.enabled = False
        checker.check_data_point(dataset[0])
        checker.reset()
        with pytest.raises(
            RuntimeError,
        ):
            checker.check_data_point(dataset[0])

    def test_shape(self):
        dataset = Dataset(indices=[2])
        checker = DatasetChecker(spatial_checks=["shape"])
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "If you don't care about spatial shape consistency and want to ignore this error, please modify 'spatial_checks'."
            ),
        ):
            checker.check(dataset)

    def test_affine(self):
        dataset = Dataset(indices=[1])
        checker = DatasetChecker(spatial_checks=["affine", "shape"])
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "If you don't care about affine matrix consistency and want to ignore this error, please modify 'spatial_checks'."
            ),
        ):
            checker.check(dataset)

    def test_global_shape(self):
        dataset = Dataset(indices=[0, 1])
        checker = DatasetChecker(spatial_checks=["global_shape", "spacing"])
        checker.check_data_point(dataset[0])
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Different spatial shape found in the dataset: for example, spatial shape is (4, 4, 4) for (sub-001, ses-M001), but (3, 3, 3) for (sub-000, ses-M000).\n"
                "If you don't care about spatial shape consistency and want to ignore this error, please modify 'spatial_checks'."
            ),
        ):
            checker.check_data_point(dataset[1])

        checker.reset()
        checker.check_data_point(dataset[1])
        checker.enabled = False
        checker.check_data_point(dataset[0])

    def test_global_spacing(self):
        dataset = Dataset(indices=[0, 1])
        checker = DatasetChecker(spatial_checks=["global_spacing", "shape"])
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Different voxel spacing found in the dataset: for example, voxel spacing is (2.0, 2.0, 2.0) for (sub-001, ses-M001), but (1.0, 1.0, 1.0) for (sub-000, ses-M000).\n"
                "If you don't care about voxel spacing consistency and want to ignore this error, please modify 'spatial_checks'."
            ),
        ):
            checker.check(dataset)

    def test_passes(self):
        dataset = Dataset(indices=[0, 3])
        checker = DatasetChecker(
            spatial_checks=["affine", "shape", "global_spacing", "global_shape"]
        )
        checker.check(dataset)
