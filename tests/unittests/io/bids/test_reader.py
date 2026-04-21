import re
from pathlib import Path

import pytest

from clinicadl.io import Bids, BidsFileType, T1Linear
from clinicadl.utils.json import write_json

DATA_DIR = Path(__file__).parents[2] / "resources"


class TestBidsReader:
    def test_init(self, tmp_path):
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(
                f"A BIDS (or a derivative) must contain a dataset_description.json. Nothing found at: {tmp_path / 'dataset_description.json'}"
            ),
        ):
            Bids(tmp_path)

        write_json(tmp_path / "dataset_description.json", {})
        with pytest.raises(
            AssertionError, match="dataset_description.json must contain 'DatasetType'"
        ):
            Bids(tmp_path)

        write_json(
            tmp_path / "dataset_description.json",
            {"DatasetType": "caps"},
            overwrite=True,
        )
        with pytest.raises(NotImplementedError):
            Bids(tmp_path)

        write_json(
            tmp_path / "dataset_description.json",
            {"DatasetType": "raw", "CAPSVersion": "0.11"},
            overwrite=True,
        )
        with pytest.raises(
            AssertionError,
            match="If the directory is a CAPS, DatasetType must be 'derivative' in dataset_description.json. Got: 'raw'",
        ):
            Bids(tmp_path)

    def test_study(self):
        bids = Bids(dir_ := DATA_DIR / "study")
        assert bids.participants_dir == dir_ / "sourcedata" / "raw"
        assert bids.tensors_dir == dir_ / "derivatives" / "tensors"

    def test_raw(self):
        bids = Bids(dir_ := DATA_DIR / "bids")
        assert bids.participants_dir == dir_
        assert bids.tensors_dir == dir_ / "derivatives" / "tensors"

    def test_caps(self):
        bids = Bids(dir_ := DATA_DIR / "bids" / "derivatives" / "caps")
        assert bids.participants_dir == dir_ / "subjects"
        assert bids.tensors_dir == dir_.parent / "tensors"

    def test_derivative(self):
        bids = Bids(dir_ := DATA_DIR / "bids" / "derivatives" / "resampling")
        assert bids.participants_dir == dir_
        assert bids.tensors_dir == dir_.parent / "tensors"

    @pytest.mark.parametrize(
        "bids_dir,file_type,participant,session,output",
        [
            (
                DATA_DIR / "bids" / "derivatives" / "caps",
                T1Linear(),
                "sub-999",
                "ses-M999",
                DATA_DIR
                / "bids"
                / "derivatives"
                / "caps"
                / "subjects"
                / "sub-999"
                / "ses-M999"
                / "t1_linear"
                / "sub-999_ses-M999_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.*",
            ),
            (
                DATA_DIR / "bids",
                BidsFileType(suffix="scans", extension=".tsv"),
                "sub-000",
                "ses-M000",
                DATA_DIR
                / "bids"
                / "sub-000"
                / "ses-M000"
                / "sub-000_ses-M000_scans.tsv",
            ),
            (
                DATA_DIR / "bids" / "derivatives" / "resampling",
                BidsFileType(suffix="sessions", extension=".tsv"),
                "sub-111",
                None,
                DATA_DIR
                / "bids"
                / "derivatives"
                / "resampling"
                / "sub-111"
                / "sub-111_sessions.tsv",
            ),
            (
                DATA_DIR / "bids",
                BidsFileType(suffix="participantsXsessions", extension=".tsv"),
                None,
                None,
                DATA_DIR / "bids" / "participantsXsessions.tsv",
            ),
        ],
    )
    def test_build_path(self, bids_dir, file_type, participant, session, output):
        bids = Bids(bids_dir)
        assert str(bids.build_path(file_type, participant, session)) == str(output)

    @pytest.mark.parametrize(
        "bids_dir,file_type,participant,session,output",
        [
            (
                DATA_DIR / "bids" / "derivatives" / "caps",
                T1Linear(use_uncropped_image=True),
                "sub-000",
                "ses-M000",
                DATA_DIR
                / "bids"
                / "derivatives"
                / "caps"
                / "subjects"
                / "sub-000"
                / "ses-M000"
                / "t1_linear"
                / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz",
            ),
            (
                DATA_DIR / "bids",
                BidsFileType(suffix="scans", extension=".tsv"),
                "sub-999",
                "ses-M999",
                DATA_DIR
                / "bids"
                / "sub-999"
                / "ses-M999"
                / "sub-999_ses-M999_scans.tsv",
            ),
            (
                DATA_DIR / "bids",
                BidsFileType(suffix="sessions", extension=".tsv"),
                "sub-999",
                None,
                DATA_DIR / "bids" / "sub-999" / "sub-999_sessions.tsv",
            ),
            (
                DATA_DIR / "bids",
                BidsFileType(suffix="participantsXsessions", extension=".tsv"),
                None,
                None,
                DATA_DIR / "bids" / "participantsXsessions.tsv",
            ),
        ],
    )
    def test_get_path_1(self, bids_dir, file_type, participant, session, output):
        bids = Bids(bids_dir)
        assert str(bids.get_path(file_type, participant, session)) == str(output)

    def test_get_path_2(self):
        caps = Bids(DATA_DIR / "bids" / "derivatives" / "caps")

        with pytest.raises(
            AssertionError,
            match="Cannot pass a session without a participant",
        ):
            caps.get_path(
                T1Linear(use_uncropped_image=False),
                session="ses-M000",
            )

        with pytest.raises(
            RuntimeError,
            match=r"For \(sub-000 | ses-M000\), an error occurred while trying to get .*: no file found",
        ):
            caps.get_path(
                T1Linear(use_uncropped_image=False),
                "sub-000",
                "ses-M000",
            )

        bids = Bids(DATA_DIR / "bids")
        with pytest.raises(
            RuntimeError,
            match=r"For \(sub-666 | ses-M666\), an error occurred while trying to get .*: more than one file found",
        ):
            bids.get_path(
                BidsFileType(data_type="anat", suffix="flair"),
                "sub-666",
                "ses-M666",
            )

    def test_has_datatype(self):
        caps = Bids(DATA_DIR / "bids" / "derivatives" / "caps")
        assert caps.has_file_type(
            "sub-000",
            "ses-M000",
            T1Linear(use_uncropped_image=True),
        )
        assert not caps.has_file_type(
            "sub-000",
            "ses-M000",
            T1Linear(use_uncropped_image=False),
        )

        bids = Bids(DATA_DIR / "bids")
        with pytest.raises(
            RuntimeError,
            match=r"For \(sub-100 | ses-M000\), an error occurred while trying to get .*: more than one file found",
        ):
            bids.has_file_type(
                "sub-100",
                "ses-M000",
                BidsFileType(
                    data_type="pet",
                    suffix="pet",
                    extension=".nii.*",
                    with_entities={"trc": "18FAV45"},
                ),
            )

    def test_get_participants_sessions(self):
        bids = Bids(DATA_DIR / "bids")
        assert bids.get_participants_sessions(
            BidsFileType(
                data_type="pet",
                suffix="pet",
                extension=".nii.*",
                with_entities={"trc": "18FAV45", "res": "1d3x1d2x1d1"},
                without_entities={"desc": "Crop"},
            )
        ) == {
            ("sub-000", "ses-M000"),
            ("sub-000", "ses-M003"),
            ("sub-010", "ses-M003"),
            ("sub-010", "ses-M012"),
        }

    def test_get_all_participants_sessions(self):
        bids = Bids(DATA_DIR / "bids" / "derivatives" / "resampling")
        assert bids.get_all_participants_sessions() == {
            ("sub-100", "ses-M000"),
            ("sub-100", "ses-M012"),
            ("sub-999", "ses-M099"),
            ("sub-999", "ses-M999"),
        }
